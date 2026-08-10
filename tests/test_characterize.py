"""Component C1 tests — domain characterization *accuracy*, evaluated
factually:

1. Image statistics are correct against *known-value* inputs (analytically
   closed-form luminance/contrast/RG ratio/sharpness).
2. Statistics respond to *controlled linear transforms* with the exact
   predicted scaling (brightness and contrast scale by alpha, sharpness by
   alpha^2, offsets leave contrast/sharpness invariant).
3. Sharpness decreases monotonically under Gaussian blur (energy removal is a
   ground-truth relationship).
4. The two-sample Phase 1 acceptance holds: within-domain split-half cosine
   dispersion < between-domain cosine separation (and retrieval top-1 = 1.0).
5. ``classes_seen`` is exact when label-derived and gated correctly to the
   prediction-histogram path; descriptors are deterministic and JSON round-trip.
"""

import json
import math

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.agents.characterize import (
    Characterizer,
    DomainDescriptor,
    ImageStatistics,
    compute_image_statistics,
    descriptor_cosine,
    descriptor_l2,
    load_descriptors_json,
    write_descriptors_json,
)
from src.agents.evaluation import (
    blur_sharpness_response,
    retrieval_accuracy,
    within_between_analysis,
)

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
NUM_CLASSES = 5


class MockEmbedModel(torch.nn.Module):
    """Deterministic fake foundation model.

    Embedding is a smooth, content-based function of (mean, std, R-channel
    mean) of the *normalized* batch: domains far apart in that statistic space
    get well-separated centroids, disjoint halves of one domain get nearly
    identical centroids. Forward logits commit to a class computed from the
    same features so the prediction-histogram path is also exercised.
    """

    def __init__(self, num_classes=NUM_CLASSES, dim=8, seed=0):
        super().__init__()
        self.num_classes = num_classes
        g = torch.Generator().manual_seed(seed)
        self.register_buffer("proj", torch.randn(dim, 3, generator=g) * 0.7)

    def extract_embedding(self, x):
        feats = torch.stack([
            x.mean(dim=(1, 2, 3)),
            x.std(dim=(1, 2, 3)),
            x[:, 0].mean(dim=(1, 2)),
        ], dim=-1)
        return torch.tanh(feats @ self.proj.t())

    def forward(self, x):
        g = x.mean(dim=(1, 2, 3))
        pred = (g * 2.0 + 2.0).long().clamp(0, self.num_classes - 1)
        logits = torch.full((x.shape[0], self.num_classes), -5.0)
        logits[torch.arange(x.shape[0]), pred] = 5.0
        return logits

    def get_num_classes(self):
        return self.num_classes


def make_characterizer(model, use_labels=True):
    return Characterizer(
        model=model,
        model_name="mock_foundation",
        num_classes=NUM_CLASSES,
        normalize_mean=MEAN,
        normalize_std=STD,
        device=torch.device("cpu"),
        use_labels_for_classes_seen=use_labels,
        seed=42,
    )


def run_characterize(images, labels=None, use_labels=True):
    model = MockEmbedModel()
    if labels is None:
        labels = torch.zeros(len(images), dtype=torch.long)
    ds = TensorDataset(images.float(), labels.long())
    loader = DataLoader(ds, batch_size=4, shuffle=False)
    return make_characterizer(model, use_labels=use_labels).characterize(
        loader, domain_name="synthetic")


# ---------------------------------------------------------------------------
# 1. Analytics image statistics vs known ground truth
# ---------------------------------------------------------------------------

def test_constant_image_exact_stats():
    c = 0.25
    x = torch.full((2, 3, 8, 8), c)
    stats = compute_image_statistics(x)
    assert stats.luminance == pytest.approx(c, abs=1e-6)
    assert stats.contrast == pytest.approx(0.0, abs=1e-6)
    assert stats.color_ratio_rg == pytest.approx(1.0, abs=1e-6)
    assert stats.sharpness == pytest.approx(0.0, abs=1e-6)


def test_binary_mix_contrast_analytic():
    x = torch.tensor([[[[0.0]], [[0.0]], [[0.0]]],
                      [[[1.0]], [[1.0]], [[1.0]]]], dtype=torch.float32)
    stats = compute_image_statistics(x)  # two pixels, p=0.5 ones
    assert stats.luminance == pytest.approx(0.5, abs=1e-6)
    assert stats.contrast == pytest.approx(math.sqrt(0.25), abs=1e-6)  # sqrt(p(1-p))
    assert stats.color_ratio_rg == pytest.approx(1.0, abs=1e-6)


def test_rg_ratio_analytic():
    x = torch.zeros(2, 3, 4, 4)
    x[:, 0] = 0.8
    x[:, 1] = 0.5
    x[:, 2] = 0.2
    stats = compute_image_statistics(x)
    assert stats.color_ratio_rg == pytest.approx(1.6, abs=1e-6)
    assert stats.luminance == pytest.approx(0.5, abs=1e-6)
    assert stats.contrast == pytest.approx(math.sqrt(0.06), abs=1e-6)


def test_rg_ratio_zero_green_neutralized():
    x = torch.zeros(2, 3, 4, 4)
    x[:, 0] = 0.5  # R>0, G=0
    stats = compute_image_statistics(x)
    assert stats.color_ratio_rg == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# 2. Controlled-linear-transform response (through the full pipeline)
# ---------------------------------------------------------------------------

def _textured_images(alpha_mult=1.0, offset=0.0, n=6, h=12, w=12):
    g = torch.Generator().manual_seed(7)
    phase = torch.arange(w, dtype=torch.float32).view(1, w)
    base = torch.sin(phase * math.pi / 3.0)  # (1, w) repeated per image
    images = []
    for i in range(n):
        img = base.repeat(3, h, 1) * 0.12 + 0.3
        img = img + torch.randn(3, h, w, generator=g) * 0.05
        images.append(img.clamp(0, 1))
    return torch.stack(images)


def test_linear_scale_alpha_response():
    x = _textured_images()
    xa = x * 0.5
    d = run_characterize(x)
    da = run_characterize(xa)
    lum_ratio = da.statistics.luminance / d.statistics.luminance
    con_ratio = da.statistics.contrast / d.statistics.contrast
    shp_ratio = da.statistics.sharpness / d.statistics.sharpness
    assert lum_ratio == pytest.approx(0.5, rel=1e-3)
    assert con_ratio == pytest.approx(0.5, rel=1e-3)
    assert shp_ratio == pytest.approx(0.25, rel=1e-2)  # Laplacian is linear -> var scales alpha^2
    assert da.statistics.color_ratio_rg == pytest.approx(
        d.statistics.color_ratio_rg, rel=1e-3)


def test_constant_offset_response():
    x = _textured_images()
    xb = (x + 0.2).clamp(0, 1)
    d = run_characterize(x)
    db = run_characterize(xb)
    assert db.statistics.luminance == pytest.approx(
        d.statistics.luminance + 0.2, abs=0.02)
    assert db.statistics.contrast == pytest.approx(d.statistics.contrast, rel=1e-3)
    assert db.statistics.sharpness == pytest.approx(d.statistics.sharpness, rel=1e-2)


def test_blur_sharpness_response_monotonic():
    g = np.random.default_rng(1)
    image = (g.uniform(0, 255, size=(48, 48, 3)) * (g.random((48, 48, 1)) > 0.5)).astype(np.float32)
    response = blur_sharpness_response(image, [0.5, 1.5, 3.0, 6.0])
    assert len(response) == 4
    assert all(math.isfinite(v) for v in response)
    assert all(response[i] > response[i + 1] for i in range(len(response) - 1))


# ---------------------------------------------------------------------------
# 3. Pipeline correctness: centroid, classes_seen, determinism, JSON
# ---------------------------------------------------------------------------

def test_centroid_unit_norm_and_dim():
    d = run_characterize(_textured_images())
    assert d.centroid_dim == d.centroid.shape[0]
    assert abs(float(np.linalg.norm(d.centroid)) - 1.0) < 1e-5
    assert d.model_name == "mock_foundation"
    assert d.n_images == _textured_images().shape[0]


def test_classes_seen_label_derived_exact():
    x = _textured_images(n=10)
    labels = torch.tensor([0, 0, 0, 1, 1, 2, 2, 3, 4, 4])
    d = run_characterize(x, labels=labels)
    assert d.classes_source == "labels"
    assert d.classes_seen == pytest.approx([0.3, 0.2, 0.2, 0.1, 0.2], abs=1e-6)


def test_classes_seen_prediction_derived_when_disabled():
    # constant 0.2 -> normalized global mean ~ -1.10 -> floor(2*g+2) clamps to 0
    # constant 0.6 -> normalized global mean ~ +0.669 -> class 3
    x = torch.cat([torch.full((3, 3, 8, 8), 0.2), torch.full((2, 3, 8, 8), 0.6)])
    d = run_characterize(x, use_labels=False)
    assert d.classes_source == "predictions"
    assert d.classes_seen == pytest.approx([0.6, 0.0, 0.0, 0.4, 0.0], abs=1e-6)


def test_determinism_identical_descriptors():
    x = _textured_images()
    a = run_characterize(x)
    b = run_characterize(x)
    assert np.array_equal(a.centroid, b.centroid)
    assert a.statistics == b.statistics
    assert a.classes_seen == b.classes_seen


def test_descriptor_json_roundtrip(tmp_path):
    d = run_characterize(_textured_images())
    path = tmp_path / "desc.json"
    d.save(path)
    loaded = DomainDescriptor.load(path)
    assert loaded.domain_name == d.domain_name
    assert np.array_equal(loaded.centroid, d.centroid)
    assert loaded.statistics.luminance == pytest.approx(d.statistics.luminance)

    map_path = tmp_path / "descriptors.json"
    write_descriptors_json({"syn": d}, map_path)
    restored = load_descriptors_json(map_path)
    assert list(restored) == ["syn"]
    assert restored["syn"].classes_seen == pytest.approx(d.classes_seen)


def test_cosine_l2_and_descriptor_roundtrip():
    a = run_characterize(torch.full((2, 3, 8, 8), 0.2))
    b = run_characterize(torch.full((2, 3, 8, 8), 0.2))
    c = run_characterize(torch.full((2, 3, 8, 8), 0.8))
    assert descriptor_cosine(a, b) > 0.999
    assert descriptor_cosine(a, c) < 0.999
    assert descriptor_l2(a, b) < descriptor_l2(a, c)


# ---------------------------------------------------------------------------
# 4. Two-sample Phase 1 acceptance + retrieval (C1 acceptance / risk R-1)
# ---------------------------------------------------------------------------

def _domain(mean, std, seed, n=48, h=16, w=16):
    g = torch.Generator().manual_seed(seed)
    phase = torch.arange(w, dtype=torch.float32).view(1, w)
    base = torch.sin(phase * 2.0 * math.pi * (0.5 + seed % 3) / w)
    images = []
    for _ in range(n):
        img = base.repeat(3, h, 1) * std
        img = img + (torch.randn(3, h, w, generator=g) * std * 0.5) + mean
        images.append(img.clamp(0, 1))
    return torch.stack(images)


def test_within_between_separation_acceptance():
    domains = {
        "dark": _domain(0.10, 0.03, seed=1),
        "mid": _domain(0.35, 0.08, seed=2),
        "bright": _domain(0.65, 0.04, seed=3),
    }
    reps = {}
    for name, images in domains.items():
        half_a = images[0::2]
        half_b = images[1::2]
        reps[name] = [run_characterize(half_a), run_characterize(half_b)]

    report = within_between_analysis(reps)
    assert report["pass"] is True
    assert report["within_mean"] > report["between_mean"]
    assert report["separation"] > 0.0

    lums = [run_characterize(img).statistics.luminance for img in domains.values()]
    names = list(domains)
    assert names[np.argmin(lums)] == "dark"
    assert names[np.argmax(lums)] == "bright"


def test_retrieval_provenance_top1():
    domains = {
        "dark": _domain(0.10, 0.03, seed=1),
        "mid": _domain(0.35, 0.08, seed=2),
        "bright": _domain(0.65, 0.04, seed=3),
    }
    reference, queries = [], []
    for name, images in domains.items():
        reference.append(run_characterize(images[0::2]))
        queries.append(run_characterize(images[1::2]))

    out = retrieval_accuracy(reference, queries)
    assert out["top1_accuracy"] == pytest.approx(1.0)
    assert out["n_queries"] == 3


def test_repeated_draw_separation_distribution():
    """Draw the same domain twice on disjoint samples -> closer than any
    cross-domain pair (the statistical version of the acceptance)."""
    d = _domain(0.35, 0.08, seed=2, n=64)
    splits = [d[0::4], d[1::4], d[2::4], d[3::4]]
    reps = {"mid": [run_characterize(d[0::2]), run_characterize(d[1::2])]}
    other = run_characterize(_domain(0.10, 0.03, seed=1, n=64))
    within = descriptor_cosine(*reps["mid"])
    cross_dists = [descriptor_cosine(reps["mid"][i], other) for i in range(2)]
    assert within > max(cross_dists)
    assert splits[0].shape[0] == 16 and len(splits) == 4