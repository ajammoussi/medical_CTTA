"""Component C1 - domain characterization (descriptor computation).

Produces the ``DomainDescriptor`` that keys the adapter bank (C2), feeds the
orchestrator (C3), and drives the freeze controller (C6). Fully deterministic:
loaders run with ``shuffle=False``, augmentation is replaced by a fixed
``Resize -> ToTensor`` transform, and the same seed is used for every pass.

Descriptor contract (``Agentic_CTTA_New_Architecture.md`` §11)::

    { "model_name", "centroid": [...D], "centroid_dim": D,
      "luminance", "contrast", "color_ratio_rg", "sharpness",
      "n_images", "classes_seen": [...] }

Semantics (fixed, so they are meaningful across runs):

- All image statistics are pooled over the *entire stream* of resized RGB
  pixels in ``[0, 1]`` (the same geometry the model consumes).
- ``luminance`` = global mean of all pixels; ``contrast`` = their population
  std.
- ``color_ratio_rg`` = mean(R) / mean(G), neutralized to ``1.0`` when G ~ 0.
- ``sharpness`` = population variance of a fixed 3x3 Laplacian response on the
  green channel, measured on the fully-valid interior (no border padding, so
  constant images score 0 and constant offsets are ignored).
- ``centroid`` = L2-normalized mean of per-sample L2-normalized CLS
  embeddings from ``model.extract_embedding`` (never ``extract_features``).
- ``classes_seen`` = class proportions over the stream; label-derived when
  ``use_labels_for_classes_seen`` (benchmark mode), otherwise estimated from
  the model's own prediction histogram (R-11 deployment mode).

Accuracy of the image statistics is verifiable analytically (known-value
synthetic images, linear-response transforms) and statistically (split-half
repeatability vs between-domain separation) in ``tests/test_characterize.py``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

logger = logging.getLogger(__name__)

_LAPLACIAN = torch.tensor(
    [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
    dtype=torch.float32,
).view(1, 1, 3, 3)


def _laplacian_response(green: torch.Tensor) -> torch.Tensor:
    """Laplacian of a ``(B, H, W)`` green channel, on the fully-valid interior.

    ``padding=0`` keeps the response translation-invariant (kernel sums to
    zero), so a constant image has a zero response and a constant offset leaves
    it unchanged. Returns ``(B, H-2, W-2)`` (empty for images < 3px).
    """
    return F.conv2d(green.unsqueeze(1), _LAPLACIAN.to(green.device)).squeeze(1)


@dataclass
class ImageStatistics:
    """Pooled image statistics over one stream of resized [0,1] RGB images."""

    luminance: float = 0.0
    contrast: float = 0.0
    color_ratio_rg: float = 1.0
    sharpness: float = 0.0
    luminance_rgb: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "luminance": float(self.luminance),
            "contrast": float(self.contrast),
            "color_ratio_rg": float(self.color_ratio_rg),
            "sharpness": float(self.sharpness),
            "luminance_rgb": [float(v) for v in self.luminance_rgb],
        }


@dataclass
class DomainDescriptor:
    """Serializable descriptor of a domain (C1 output; C2 bank key)."""

    model_name: str
    domain_name: str
    centroid: np.ndarray
    centroid_dim: int
    n_images: int
    num_classes: int
    classes_seen: List[float]
    classes_source: str = "labels"
    statistics: ImageStatistics = field(default_factory=ImageStatistics)
    partition_counts: Dict[str, int] = field(default_factory=dict)
    labeled_n: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "domain_name": self.domain_name,
            "centroid": [float(v) for v in self.centroid],
            "centroid_dim": int(self.centroid_dim),
            "n_images": int(self.n_images),
            "num_classes": int(self.num_classes),
            "classes_seen": [float(v) for v in self.classes_seen],
            "classes_source": self.classes_source,
            "labeled_n": int(self.labeled_n),
            "partition_counts": dict(self.partition_counts),
            "statistics": self.statistics.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DomainDescriptor":
        centroid = np.asarray(data["centroid"], dtype=np.float32)
        return cls(
            model_name=data["model_name"],
            domain_name=data["domain_name"],
            centroid=centroid,
            centroid_dim=int(data["centroid_dim"]),
            n_images=int(data["n_images"]),
            num_classes=int(data["num_classes"]),
            classes_seen=[float(v) for v in data["classes_seen"]],
            classes_source=data.get("classes_source", "labels"),
            labeled_n=int(data.get("labeled_n", 0)),
            partition_counts=dict(data.get("partition_counts", {})),
            statistics=ImageStatistics(**data.get("statistics", {})),
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True))

    @classmethod
    def load(cls, path: str | Path) -> "DomainDescriptor":
        return cls.from_dict(json.loads(Path(path).read_text()))


def descriptor_cosine(a: DomainDescriptor, b: DomainDescriptor) -> float:
    """Cosine similarity between two descriptors' normalized centroids (0..1)."""
    va = a.centroid
    vb = b.centroid
    return float((va @ vb) / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-12))


def descriptor_l2(a: DomainDescriptor, b: DomainDescriptor) -> float:
    """L2 distance between full descriptor feature vectors (stats + centroid)."""
    va = _descriptor_vector(a)
    vb = _descriptor_vector(b)
    return float(np.linalg.norm(va - vb))


def _descriptor_vector(d: DomainDescriptor) -> np.ndarray:
    stats = np.array(
        [d.statistics.luminance, d.statistics.contrast, d.statistics.color_ratio_rg,
         d.statistics.sharpness],
        dtype=np.float32,
    )
    return np.concatenate([d.centroid, stats])


def compute_image_statistics(x: torch.Tensor) -> ImageStatistics:
    """Pooled image statistics of a ``(B, 3, H, W)`` tensor in ``[0, 1]``.

    Pure and analytic: every returned number is a closed-form function of the
    input pixels, which is what makes the C1 accuracy evaluation factual.
    """
    if x.dim() != 4 or x.size(1) != 3:
        raise ValueError(f"expected (B,3,H,W) in [0,1], got {tuple(x.shape)}")
    flat = x.float()
    r, g, b = flat[:, 0], flat[:, 1], flat[:, 2]

    if flat.size(2) >= 3 and flat.size(3) >= 3:
        lap_interior = _laplacian_response(g)
        sharpness = float(lap_interior.var(unbiased=False).item())
    else:
        sharpness = 0.0  # images < 3px have no interior Laplacian response

    g_mean = float(g.mean().item())
    rg = float(r.mean().item() / g_mean) if g_mean > 1e-9 else 1.0

    return ImageStatistics(
        luminance=float(flat.mean().item()),
        contrast=float(flat.std(unbiased=False).item()),
        color_ratio_rg=rg,
        sharpness=sharpness,
        luminance_rgb=[
            float(r.mean().item()),
            float(g.mean().item()),
            float(b.mean().item()),
        ],
    )


class _StatsAccumulator:
    """Running pool statistics over the whole stream (single pass)."""

    def __init__(self) -> None:
        # Luminance/contrast are pooled over every channel value (3*B*H*W);
        # RGB means and Laplacian responses are pooled per-channel (B*H*W).
        self.sum_x = 0.0
        self.sum_x2 = 0.0
        self.sum_r = 0.0
        self.sum_g = 0.0
        self.sum_b = 0.0
        self.sum_lap = 0.0
        self.sum_lap2 = 0.0
        self.n_all = 0
        self.n_pixels = 0
        self.n_lap_pixels = 0
        self.n_images = 0

    def update(self, x: torch.Tensor) -> None:
        if x.dim() != 4 or x.size(1) != 3:
            raise ValueError(f"expected (B,3,H,W), got {tuple(x.shape)}")
        flat = x.float()
        b, _, h, w = flat.shape
        n_all = float(b * 3 * h * w)
        n_px = float(b * h * w)
        self.sum_x += float(flat.sum().item())
        self.sum_x2 += float((flat * flat).sum().item())
        self.sum_r += float(flat[:, 0].sum().item())
        self.sum_g += float(flat[:, 1].sum().item())
        self.sum_b += float(flat[:, 2].sum().item())
        lap = _laplacian_response(flat[:, 1])
        self.sum_lap += float(lap.sum().item())
        self.sum_lap2 += float((lap * lap).sum().item())
        self.n_all += n_all
        self.n_pixels += n_px
        self.n_lap_pixels += int(lap.numel())
        self.n_images += b

    def finalize(self) -> ImageStatistics:
        if self.n_pixels == 0:
            raise ValueError("cannot finalize empty statistics accumulator")
        mean = self.sum_x / self.n_all
        var = max(0.0, self.sum_x2 / self.n_all - mean * mean)
        if self.n_lap_pixels > 0:
            lap_mean = self.sum_lap / self.n_lap_pixels
            lap_var = max(0.0, self.sum_lap2 / self.n_lap_pixels - lap_mean * lap_mean)
        else:
            lap_var = 0.0
        r_mean = self.sum_r / self.n_pixels
        g_mean = self.sum_g / self.n_pixels
        b_mean = self.sum_b / self.n_pixels
        return ImageStatistics(
            luminance=float(mean),
            contrast=float(var ** 0.5),
            color_ratio_rg=float(r_mean / g_mean) if g_mean > 1e-9 else 1.0,
            sharpness=float(lap_var),
            luminance_rgb=[float(r_mean), float(g_mean), float(b_mean)],
        )


class Characterizer:
    """Offline descriptor computation for one domain (C1).

    ``normalize_mean`` / ``normalize_std`` are the image-normalization constants
    of the embedding model; characterization streams *raw* [0,1] images for the
    statistics and applies the normalization internally for extraction.
    """

    def __init__(
        self,
        model: Any,
        model_name: str,
        num_classes: int,
        normalize_mean: List[float],
        normalize_std: List[float],
        device: torch.device = torch.device("cpu"),
        use_labels_for_classes_seen: bool = True,
        seed: int = 42,
    ) -> None:
        self.model = model
        self.model_class = model_name
        self.num_classes = num_classes
        self.normalize_mean = torch.tensor(normalize_mean, dtype=torch.float32)
        self.normalize_std = torch.tensor(normalize_std, dtype=torch.float32)
        self.device = device
        self.use_labels_for_classes_seen = use_labels_for_classes_seen
        self.seed = seed

    def characterize(
        self,
        loader: DataLoader,
        domain_name: str,
        labeled_loader: Optional[DataLoader] = None,
        partition_counts: Optional[Dict[str, int]] = None,
        batch_size: int = 32,
        class_counts: Optional[np.ndarray] = None,
        pass_label: str = "",
    ) -> DomainDescriptor:
        """Stream ``loader`` (yields ``(raw_image, label)`` or ``(raw_image,)``)
        and return the domain descriptor.

        ``loader`` drives the centroid + pooled image statistics over the full
        stream. ``classes_seen`` is label-derived from ``labeled_loader`` when
        provided (a strictly-labeled subset of the stream, so placeholder
        labels on unlabeled competition splits never leak into the
        distribution); otherwise it falls back to the labels inside ``loader``
        (legacy behavior) or to the model's prediction histogram when
        ``use_labels_for_classes_seen`` is disabled.

        ``class_counts`` is an exact per-class label count (e.g. taken directly
        from a dataset manifest) that skips the extra ``labeled_loader`` image
        pass entirely — used by the C1 driver, which already knows the split
        label histogram.
        """
        torch.manual_seed(self.seed)
        self.model.eval()
        acc = _StatsAccumulator()
        emb_sum = None
        n_emb = 0

        use_labels = self.use_labels_for_classes_seen
        provided_counts = class_counts is not None
        class_counts = (
            None if class_counts is None
            else np.asarray(class_counts, dtype=np.float64).reshape(-1)
        )
        if class_counts is not None and class_counts.size != self.num_classes:
            raise ValueError(
                f"class_counts has {class_counts.size} entries, expected {self.num_classes}")
        if class_counts is None:
            class_counts = np.zeros(self.num_classes, dtype=np.float64)
        count_labels_in_loop = (
            use_labels and labeled_loader is None and not provided_counts
        )

        with torch.no_grad():
            for batch in tqdm(
                loader,
                total=len(loader) if hasattr(loader, "__len__") else None,
                desc=f"  characterize {domain_name}"
                + (f" [{pass_label}]" if pass_label else ""),
                leave=False,
                unit="batch",
            ):
                if isinstance(batch, (list, tuple)):
                    raw, labels = batch[0], batch[1]
                else:
                    raw, labels = batch, None
                raw = raw.to(self.device)
                acc.update(raw)
                x = self._normalize(raw)

                embeddings = self.model.extract_embedding(x)
                embeddings = F.normalize(embeddings, dim=1)
                if emb_sum is None:
                    emb_sum = embeddings.sum(dim=0)
                else:
                    emb_sum = emb_sum + embeddings.sum(dim=0)
                n_emb += embeddings.size(0)

                if count_labels_in_loop and labels is not None:
                    labels = labels.reshape(-1)
                    valid = labels >= 0
                    if valid.any():
                        for c in range(self.num_classes):
                            class_counts[c] += float((labels[valid] == c).sum().item())
                elif not use_labels:
                    logits = self.model(x)
                    preds = torch.argmax(logits, dim=-1).reshape(-1)
                    for c in range(self.num_classes):
                        class_counts[c] += float((preds == c).sum().item())

        if use_labels and labeled_loader is not None and not provided_counts:
            class_counts = self._count_labels(labeled_loader)

        if n_emb == 0:
            raise ValueError(f"characterized stream {domain_name!r} is empty")

        centroid = F.normalize(emb_sum.unsqueeze(0), dim=1).squeeze(0).cpu().numpy()
        total_labels = float(class_counts.sum())
        if total_labels > 0:
            classes_seen = (class_counts / total_labels).tolist()
        else:
            classes_seen = [0.0] * self.num_classes
        classes_source = "labels" if self.use_labels_for_classes_seen else "predictions"

        return DomainDescriptor(
            model_name=self.model_class,
            domain_name=domain_name,
            centroid=centroid.astype(np.float32),
            centroid_dim=int(centroid.shape[0]),
            n_images=int(acc.n_images),
            num_classes=self.num_classes,
            classes_seen=classes_seen,
            classes_source=classes_source,
            labeled_n=int(total_labels),
            partition_counts=dict(partition_counts or {}),
            statistics=acc.finalize(),
        )

    def _count_labels(self, loader: DataLoader) -> np.ndarray:
        """Count per-class labels over ``loader`` without any model forwards."""
        counts = np.zeros(self.num_classes, dtype=np.float64)
        for batch in loader:
            if isinstance(batch, (list, tuple)):
                raw, labels = batch[0], batch[1]
            else:
                raw, labels = batch, None
            if labels is None:
                continue
            labels = labels.reshape(-1)
            valid = labels >= 0
            if valid.any():
                for c in range(self.num_classes):
                    counts[c] += float((labels[valid] == c).sum().item())
        return counts

    def _normalize(self, raw: torch.Tensor) -> torch.Tensor:
        mean = self.normalize_mean.to(raw.device).view(3, 1, 1)
        std = self.normalize_std.to(raw.device).view(3, 1, 1)
        return (raw - mean) / std


def write_descriptors_json(descriptors: Dict[str, DomainDescriptor], path: str | Path) -> Path:
    """Write a ``{domain_name: descriptor}`` map as a JSON artifact (Phase 1
    exit criterion: a prototype run builds a ``descriptors.json``)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {name: d.to_dict() for name, d in sorted(descriptors.items())}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return path


def load_descriptors_json(path: str | Path) -> Dict[str, DomainDescriptor]:
    return {name: DomainDescriptor.from_dict(data) for name, data in
            json.loads(Path(path).read_text()).items()}