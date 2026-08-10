"""Factual accuracy evaluation for C1 domain characterization.

Follows the methodology used in the dataset-shift detection literature
(Rabanser et al., NeurIPS 2019 "Failing Loudly"; Roschewitz et al., 2024):

- ``within_between_analysis``: two-sample design. Characterize each domain on
  two disjoint halves; the within-domain cosine dispersion (replicate A vs B)
  must be smaller than the between-domain cosine distance (replicate of domain
  A vs replicate of domain B). This is exactly the Phase 1 acceptance criterion
  of ``Agentic_CTTA_New_Architecture.md`` §5 and Risk R-1.
- ``retrieval_accuracy``: provenance test. Every characterized half must be
  retrieved to its own domain under the C2 retrieval rule (centroid cosine),
  reporting top-1 hit-rate over all trials.
- ``blur_sharpness_response``: ground-truth transform response. Gaussian blur
  strictly removes high-frequency energy, so measured Laplacian-variance
  sharpness must decrease monotonically with ``sigma``. This is a controlled
  factual check on the *sharpness statistic itself*.

Every metric here is deterministic and computed without any learned baseline,
so a PASS/FAIL is fully reproducible.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from src.agents.characterize import DomainDescriptor, descriptor_cosine

SEPARATION_MARGIN = 0.02  # between must exceed within by this much (cosine units)
DEFAULT_MATCH_THRESHOLD = 0.95  # C2 "same domain" threshold from the architecture
DEFAULT_MIN_SHARE = 0.9  # fraction of within-half pairs that must clear the threshold


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def within_between_analysis(
    descriptors_by_domain: Dict[str, List[DomainDescriptor]],
    match_threshold: float = DEFAULT_MATCH_THRESHOLD,
    min_share: float = DEFAULT_MIN_SHARE,
) -> Dict[str, float]:
    """Two-sample split-half acceptance for the centroid bank key.

    ``descriptors_by_domain[d]`` must contain at least two descriptors (two
    disjoint halves of the same domain). Computes, in cosine space:

    - ``within_mean`` / ``min_within`` / ``p10_within`` / ``p50_within``:
      cosine between halves of the same domain.
    - ``between_mean`` / ``max_between``: cosine between halves of different
      domains.
    - ``separation``: ``within_mean - between_mean``.
    - ``share_within_ge_threshold``: fraction of within-half pairs with cosine
      ``>= match_threshold`` — the *absolute* operating point the C2 bank will
      actually use (the 0.95 "same domain" gate). Separation alone can pass
      while 0.95 is unreachable, which silently disables M1/M3 (the G1 trap).
    - ``pass``: separation holds **and** the absolute threshold is reachable.
    """
    within: List[float] = []
    between: List[float] = []
    domains = list(descriptors_by_domain)
    for i, da in enumerate(domains):
        reps_a = descriptors_by_domain[da]
        for a0, a1 in zip(reps_a[:-1], reps_a[1:]):
            within.append(descriptor_cosine(a0, a1))
        for db in domains[i + 1:]:
            reps_b = descriptors_by_domain[db]
            for a_rep in reps_a:
                for b_rep in reps_b:
                    between.append(descriptor_cosine(a_rep, b_rep))

    within_mean = float(np.mean(within)) if within else 0.0
    between_mean = float(np.mean(between)) if between else 0.0
    min_within = float(np.min(within)) if within else 0.0
    max_between = float(np.max(between)) if between else 1.0
    separation = within_mean - between_mean
    p10_within = float(np.percentile(within, 10)) if within else 0.0
    p50_within = float(np.percentile(within, 50)) if within else 0.0
    n_within = len(within)
    share_within_ge = (
        float(sum(1.0 for c in within if c >= match_threshold)) / n_within
        if n_within else 0.0
    )
    pass_sep = bool(min_within > max_between + SEPARATION_MARGIN)
    pass_abs = bool(share_within_ge >= min_share)
    return {
        "within_mean": within_mean,
        "between_mean": between_mean,
        "min_within": min_within,
        "max_between": max_between,
        "p10_within": p10_within,
        "p50_within": p50_within,
        "n_within_pairs": n_within,
        "separation": separation,
        "match_threshold": float(match_threshold),
        "share_within_ge_threshold": share_within_ge,
        "pass_sep": pass_sep,
        "pass_abs": pass_abs,
        "pass": bool(pass_sep and pass_abs),
    }


def similarity_matrix(
    descriptors: Dict[str, DomainDescriptor],
) -> "np.ndarray":
    """Pairwise centroid-cosine similarity across domains (readable table).

    Returns a NumPy matrix indexed in the same order as ``list(descriptors)``;
    the caller formats it (e.g. with pandas) for human-readable output.
    """
    names = list(descriptors)
    n = len(names)
    mat = np.zeros((n, n), dtype=np.float64)
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            mat[i, j] = descriptor_cosine(descriptors[a], descriptors[b])
    return mat


def retrieval_accuracy(
    reference: List[DomainDescriptor],
    queries: List[DomainDescriptor],
) -> Dict[str, float]:
    """Top-1 provenance accuracy under the C2 rule (centroid cosine, same-model
    filter applied implicitly by using one model per call)."""
    if not reference or not queries:
        return {"top1_accuracy": 0.0, "n_queries": 0}
    hits = 0
    for q in queries:
        best = None
        best_sim = -2.0
        for r in reference:
            sim = descriptor_cosine(q, r)
            if sim > best_sim:
                best_sim = sim
                best = r
        if best is not None and best.domain_name == q.domain_name:
            hits += 1
    return {
        "top1_accuracy": float(hits) / float(len(queries)),
        "n_queries": len(queries),
    }


def blur_sharpness_response(
    image: np.ndarray,
    sigmas: List[float],
    interpolate: bool = False,
) -> List[float]:
    """Measured Laplacian-variance sharpness of ``image`` blurred at each sigma.

    A controlled ground-truth check: values must be strictly decreasing in
    ``sigma`` (blur removes high-frequency energy). ``image`` is ``(H, W, 3)``
    in ``[0, 255]`` (PIL convention is applied inside).

    The Laplacian-response variance is measured on the interior region only:
    ``gaussian_blur``'s reflect padding injects boundary energy that otherwise
    inflates the variance and breaks monotonicity at large ``sigma``.
    """
    import torch
    import torch.nn.functional as F
    from torchvision.transforms.functional import gaussian_blur

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("expected (H, W, 3) uint8-or-float array")
    t = torch.from_numpy(np.asarray(image, dtype=np.float32)).permute(2, 0, 1) / 255.0
    lap = torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]]).view(1, 1, 3, 3)
    results = []
    for sigma in sigmas:
        if sigma <= 0:
            kernel_size = 0  # identity
            blurred = t
        else:
            k = int(round(sigma * 6)) | 1  # odd kernel, ~6*sigma spans
            k = max(k, 3)
            blurred = gaussian_blur(t.unsqueeze(0), kernel_size=k, sigma=sigma).squeeze(0)
        green = blurred[1].unsqueeze(0).unsqueeze(0)
        response = F.conv2d(green, lap, padding=1)
        pad = (max(0, int(round(sigmas[0] * 6) if sigma > 0 else 0)) // 2) + 2
        interior = response[:, :, pad:-pad or None, pad:-pad or None]
        results.append(float(interior.var(unbiased=False).item()))
    return results