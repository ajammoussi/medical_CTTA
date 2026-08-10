"""C0 drift-latency verification harness (fast-fourier style).

Proves the *core C0 exit criterion* quantitatively: telemetry signals sampled
over a ``min(config_window, n_stream)`` window detect a distributional shift
with **bounded** latency, while the same signal measured as a whole-stream
statistic has time-to-detect that grows with the prior history length ``P``
and never shrinks.

Three signal families are covered, mirroring the C0 windowing semantics of
``src/utils/metrics`` (helper *numerics* are validated separately in
``tests/test_phase0.py``; this harness proves the *latency* semantics):

1. **Class distribution**   -> ``min(200, n_samples)`` window over predicted
   samples (the ``class_distribution_proportions`` field semantics)
2. **Embedding centroid**   -> ``min(200, n_batches)`` window over per-batch
   centroids (the ``window_centroid`` + ``cosine_drift`` semantics)
3. **Confidence scalar**    -> windowed/global mean (entropy / mean-max-prob
   telemetry family; same batch-granularity windowing as the centroid)

Design (fast-fourier style): fully deterministic, incremental single-pass (no
repeated slicing, no RNG). Streams are built from *expected* per-batch
counts/proportions, so latency is an exact integer batch count and every
scenario is analytically checkable.

Exact analytic predictions (mid-jump detection threshold):

- class-dist shift (p0=0.3 -> p1=0.8, batch=20, window=200 samples = 10 batches):
  windowed latency = 5 batches (independent of ``P``), global = ``P`` exactly.
- confidence scalars (1.20 -> 0.80): windowed = ``k/2`` = 100 batches,
  global = ``P`` exactly.
- centroid shift (0 -> 30 deg): windowed <= ``k`` (bounded), global grows
  ~2.3x per unit of ``P``.
- no-shift controls: neither detector ever fires.
- early stream (``n_stream < window``): ``min(k, n_stream)`` == whole stream,
  so windowed and global are identical -- this is why the ``min()`` constraint
  exists.

Run directly (prints table, non-zero exit on violation):

    python scripts/verify_c0_drift_latency.py

The same scenarios are asserted in ``tests/test_c0_drift_latency.py``.
Behavior-change guarantee is tracked separately: with ``agents_enabled=false``
the runners emit legacy entries byte-identically (``tests/test_phase0.py``).
"""

from __future__ import annotations

import math
from collections import deque

import torch

from src.utils.metrics import cosine_drift

CONFIG_WINDOW = 200  # the C0 config default (streaming_window)

CLASS_BATCH = 20
CLASS_PRE = 6   # p0 = 0.30 -> 6 ones per 20-sample batch
CLASS_POST = 16  # p1 = 0.80 -> 16 ones per 20-sample batch
CLASS_TAU = 0.55  # mid jump

CONF_PRE, CONF_POST, CONF_TAU = 1.2, 0.8, 1.0  # mid jump of a supposedly-stable metric

EPS = 1e-9


# ---------------------------------------------------------------------------
# class-distribution detector (sample-granularity window, per C0 helper)
# ---------------------------------------------------------------------------

def _class_track(labels: list[int], *, windowed: bool) -> int:
    """First 1-based batch index where class-1 proportion >= CLASS_TAU.

    ``windowed=True`` counts over the last ``min(200, n_stream)`` *samples*
    (the C0 rule); ``windowed=False`` over the whole stream so far.
    """
    dq: deque[int] = deque(maxlen=CONFIG_WINDOW)
    ones_seen = 0
    samples = 0
    for i, lbl in enumerate(labels):
        ones_seen += int(lbl)
        samples += 1
        dq.append(int(lbl))
        if (i + 1) % CLASS_BATCH == 0:  # only ever check at batch boundaries
            cell = sum(dq) / len(dq) if windowed else ones_seen / samples
            if cell >= CLASS_TAU - EPS:
                return (i + 1) // CLASS_BATCH
    return -1


def class_shift_latency(prior: int, *, windowed: bool) -> int:
    """Post-shift batches until class-1 shift is detected (mid-jump threshold)."""
    pre = [0] * (CLASS_BATCH - CLASS_PRE) + [1] * CLASS_PRE
    post = [0] * (CLASS_BATCH - CLASS_POST) + [1] * CLASS_POST
    labels = pre * prior + post * (prior + CONFIG_WINDOW)
    return _class_track(labels, windowed=windowed) - prior


# ---------------------------------------------------------------------------
# confidence-scalar detector (batch-granularity window, entropy / max-prob)
# ---------------------------------------------------------------------------

def _confidence_track(values: list[float], *, windowed: bool) -> int:
    """First 1-based batch index where mean confidence drops to <= CONF_TAU."""
    dq: deque[float] = deque(maxlen=CONFIG_WINDOW)
    running = 0.0
    n = 0
    for i, v in enumerate(values):
        if len(dq) == CONFIG_WINDOW:
            dq.popleft()
        dq.append(v)
        running += v
        n += 1
        cell = sum(dq) / len(dq) if windowed else running / n
        if cell <= CONF_TAU + EPS:
            return i + 1
    return -1


def confidence_shift_latency(prior: int, *, windowed: bool) -> int:
    """Post-shift batches until the confidence drop is detected (mid-jump)."""
    values = [CONF_PRE] * prior + [CONF_POST] * (prior + CONFIG_WINDOW)
    return _confidence_track(values, windowed=windowed) - prior


# ---------------------------------------------------------------------------
# centroid detector (batch-granularity window)
# ---------------------------------------------------------------------------

def _centroid_basis() -> tuple[torch.Tensor, torch.Tensor]:
    dim = 8
    e0 = torch.zeros(dim)
    e0[0] = 1.0
    u = torch.zeros(dim)
    u[0] = math.cos(math.radians(30.0))
    u[1] = math.sin(math.radians(30.0))
    return e0, u


def _centroid_track(centroids: list[torch.Tensor], *, windowed: bool) -> int:
    """First 1-based batch index where drift(centroid vs e0) >= half-saturation.

    Incremental single-pass: keeps a running vector sum (and, when windowed, a
    ``min(200, n_stream)`` queue to subtract the oldest on eviction).
    ``cosine_drift`` re-normalizes internally, so the un-normalized running sum
    is a valid centroid argument.
    """
    e0, _ = _centroid_basis()
    tau = (1.0 - math.cos(math.radians(30.0))) / 2.0  # 0.5 * saturation drift
    dq: deque[torch.Tensor] = deque()
    total = torch.zeros_like(e0)
    for n, v in enumerate(centroids, 1):
        if windowed:
            if len(dq) == CONFIG_WINDOW:
                total = total - dq.popleft()
            dq.append(v)
            total = total + v
        else:
            total = total + v
        if total.norm().item() > 0.0 and cosine_drift(total, e0) >= tau - EPS:
            return n
    return -1


def centroid_shift_latency(prior: int, *, windowed: bool) -> int:
    """Post-shift batches until the centroid rotation is detected."""
    e0, u = _centroid_basis()
    post_budget = 3 * prior + CONFIG_WINDOW  # global needs ~2.32*P here
    centroids = [e0] * prior + [u] * post_budget
    return _centroid_track(centroids, windowed=windowed) - prior


# ---------------------------------------------------------------------------
# no-shift controls and early-stream constraint
# ---------------------------------------------------------------------------

def no_shift_class(prior: int = 300) -> bool:
    pre = [0] * (CLASS_BATCH - CLASS_PRE) + [1] * CLASS_PRE
    return _class_track(pre * (2 * prior), windowed=True) >= 0


def no_shift_confidence(prior: int = 300) -> bool:
    return _confidence_track([CONF_PRE] * (2 * prior), windowed=True) >= 0


def no_shift_centroid(prior: int = 300) -> bool:
    e0, _ = _centroid_basis()
    return _centroid_track([e0] * (2 * prior), windowed=True) >= 0


def early_stream_latency(prior: int) -> tuple[int, int]:
    """P < window: ``min(k, n_stream)`` == whole stream -> identical latencies."""
    w = confidence_shift_latency(prior, windowed=True)
    g = confidence_shift_latency(prior, windowed=False)
    return w, g


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------

def _row(name: str, prior: int, windowed_lat: int, global_lat: int) -> str:
    ratio = "--" if windowed_lat <= 0 else (
        "inf" if global_lat <= 0 else f"{global_lat / windowed_lat:.1f}x")
    return (f"  {name:<44} P={prior:<5} | windowed={str(windowed_lat):<5} | "
            f"global={str(global_lat):<5} | ratio={ratio}")


def run_panel() -> list[str]:
    lines = [
        "C0 drift-latency proof (config window k=200, mid-jump threshold):",
        "  windowed = min(k, n_stream)  |  global = whole-stream running mean",
        "-" * 78,
    ]
    for p in [200, 400, 1000]:
        w = class_shift_latency(p, windowed=True)
        g = class_shift_latency(p, windowed=False)
        lines.append(_row("class distribution (0.3 -> 0.8, sample-window)", p, w, g))
    for p in [200, 400, 1000]:
        w = confidence_shift_latency(p, windowed=True)
        g = confidence_shift_latency(p, windowed=False)
        lines.append(_row("confidence scalar (1.20 -> 0.80)", p, w, g))
    for p in [200, 400, 800]:
        w = centroid_shift_latency(p, windowed=True)
        g = centroid_shift_latency(p, windowed=False)
        lines.append(_row("embedding centroid (0 deg -> 30 deg)", p, w, g))

    lines.append("-" * 78)
    lines.append("No-shift controls (must never fire):")
    lines.append(f"  class       -> fired={no_shift_class()}")
    lines.append(f"  confidence  -> fired={no_shift_confidence()}")
    lines.append(f"  centroid    -> fired={no_shift_centroid()}")
    w_early, g_early = early_stream_latency(50)
    lines.append(
        f"  early stream P=50 < k: windowed={w_early} == global={g_early} "
        f"(documents the min() constraint)"
    )
    return lines


def check_expectations() -> list[str]:
    """Structural assertions (returned as failure messages)."""
    failures: list[str] = []

    def fail(msg: str) -> None:
        failures.append(msg)

    # class: windowed == 5 batches, independent of P; global == P
    for p in [200, 400, 1000]:
        w = class_shift_latency(p, windowed=True)
        g = class_shift_latency(p, windowed=False)
        if w != 5:
            fail(f"class windowed latency {w} != 5 at P={p}")
        if g != p:
            fail(f"class global latency {g} != P={p}")
        if w >= g:
            fail(f"class windowed {w} not < global {g} at P={p}")

    # confidence: windowed == k/2 == 100; global == P
    for p in [200, 400, 1000]:
        w = confidence_shift_latency(p, windowed=True)
        g = confidence_shift_latency(p, windowed=False)
        if w != CONFIG_WINDOW // 2:
            fail(f"confidence windowed latency {w} != k/2 at P={p}")
        if g != p:
            fail(f"confidence global latency {g} != P={p}")

    # centroid: windowed bounded (~ const, <= k), global monotone and slower
    cen_w, cen_g = [], []
    for p in [200, 400, 800]:
        w = centroid_shift_latency(p, windowed=True)
        g = centroid_shift_latency(p, windowed=False)
        cen_w.append(w)
        cen_g.append(g)
        if not (0 < w <= CONFIG_WINDOW):
            fail(f"centroid windowed latency {w} not in (0, k] at P={p}")
        if not (g > w):
            fail(f"centroid global {g} not slower than windowed {w} at P={p}")
    if not (cen_w[0] == cen_w[1] == cen_w[2]):
        fail(f"centroid windowed latency not P-independent: {cen_w}")
    if not (cen_g[0] < cen_g[1] < cen_g[2]):
        fail(f"centroid global latency not monotone in P: {cen_g}")

    # no-shift controls
    if no_shift_class():
        fail("class no-shift control fired")
    if no_shift_confidence():
        fail("confidence no-shift control fired")
    if no_shift_centroid():
        fail("centroid no-shift control fired")

    # early-stream min() constraint
    w_early, g_early = early_stream_latency(50)
    if w_early != g_early:
        fail(f"min() constraint broken: windowed={w_early} global={g_early}")

    return failures


if __name__ == "__main__":
    print("\n".join(run_panel()))
    print()
    failures = check_expectations()
    if failures:
        print("FAILURES:")
        for msg in failures:
            print(f"  - {msg}")
        raise SystemExit(1)
    print("All C0 drift-latency expectations hold.")