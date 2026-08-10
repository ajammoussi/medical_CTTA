"""Windowed telemetry helpers (Component C0 of the agentic CTTA architecture).

Pure, deterministic helpers that transform per-batch signals into the
statistically meaningful windowed artifacts (constraint C5 in
``Agentic_CTTA_New_Architecture.md``: ``window = min(config, n_stream)`` samples,
never single batches).

All functions are side-effect free and accept torch tensors, numpy arrays, or
plain lists where reasonable. They never gate adaptation; they only *log*
(per ``changes/2026-07-31_remove_shift_detection.md``, no per-batch gating).
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F


def _to_tensor(x: Any) -> torch.Tensor:
    """Coerce a tensor / numpy array / list-of-numbers into a float32 CPU tensor."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().float()
    if isinstance(x, np.ndarray):
        return torch.from_numpy(np.asarray(x, dtype=np.float32))
    return torch.as_tensor(x, dtype=torch.float32)


def l2_normalize(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """L2-normalize along ``dim`` with a zero-norm guard (returns zero vector)."""
    return F.normalize(x, p=2, dim=dim, eps=1e-12)


def windowed_class_distribution(
    pred_hist: Sequence[int], window: int = 200
) -> Dict[int, int]:
    """Count pred label occurrences over the *last* ``window`` predictions.

    Returns a mapping ``{class_id: count}``. ``window <= 0`` means "use the whole
    history". The function is total (never raises), returning ``{}`` for empty
    input.
    """
    if window is None or window <= 0 or window >= len(pred_hist):
        stats = pred_hist[-len(pred_hist):]
    else:
        stats = pred_hist[-window:]
    counts: Dict[int, int] = {}
    for p in stats:
        counts[int(p)] = counts.get(int(p), 0) + 1
    return counts


def class_distribution_proportions(
    pred_hist: Sequence[int], num_classes: int, window: int = 200
) -> List[float]:
    """Windowed class distribution as proportions over ``num_classes`` bins.

    The returned list sums to ~1.0 and is the per-batch ``class_distribution``
    telemetry field (C0).
    """
    counts = windowed_class_distribution(pred_hist, window)
    total = sum(counts.values())
    if total <= 0:
        return [0.0] * num_classes
    return [counts.get(c, 0) / total for c in range(num_classes)]


def cosine_drift(centroid_a: Any, centroid_b: Any) -> float:
    """Cosine drift ``1 - cos(a, b)`` for L2-normalized (or arbitrary) vectors.

    ``centroid_a`` / ``centroid_b`` are 1-D vectors. A zero vector is treated as
    orthogonal (drift = 1.0), matching torch's ``cosine_similarity`` convention
    for zero norms.
    """
    a = l2_normalize(_to_tensor(centroid_a).reshape(-1))
    b = l2_normalize(_to_tensor(centroid_b).reshape(-1))
    sim = float(F.cosine_similarity(a, b, dim=0).item())
    return float(1.0 - sim)


def window_centroid(batch_centroids: Sequence[Any], window: int = 200) -> torch.Tensor:
    """Mean of the last ``window`` per-batch centroids, re-L2-normalized.

    Each ``batch_centroids[i]`` is a 1-D centroid (already unit/any-norm) of one
    batch. Returns a float32, CPU, unit-norm tensor of the same dimension.
    ``window <= 0`` uses the whole history.
    """
    if len(batch_centroids) == 0:
        raise ValueError("window_centroid requires at least one batch centroid")
    if window is None or window <= 0 or window >= len(batch_centroids):
        recent = batch_centroids
    else:
        recent = batch_centroids[-window:]

    vecs = torch.stack([_to_tensor(c).reshape(-1) for c in recent], dim=0)
    mean = vecs.mean(dim=0)
    return l2_normalize(mean, dim=0)