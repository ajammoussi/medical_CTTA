"""C0 drift-latency verification (pytest wrapper around the counterfactual
harness in ``scripts/verify_c0_drift_latency.py``).

Exit criterion: window-local telemetry (``window = min(config_window, n_stream)``)
detects a distributional shift with bounded latency, while whole-stream
statistics need ~P (the full prior history) batches. All scenarios are
deterministic (expected-value streams, no RNG) so latencies are exact integers.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from verify_c0_drift_latency import (  # noqa: E402
    CONFIG_WINDOW,
    check_expectations,
    class_shift_latency,
    confidence_shift_latency,
    centroid_shift_latency,
    early_stream_latency,
    no_shift_centroid,
    no_shift_class,
    no_shift_confidence,
    run_panel,
)


def test_all_drift_latency_expectations_hold():
    assert check_expectations() == []


def test_panel_is_reproducible():
    assert run_panel() == run_panel()  # fully deterministic, no RNG


def test_class_windowed_latency_bounded_and_p_independent():
    for p in [200, 400, 1000]:
        w = class_shift_latency(p, windowed=True)
        assert w == 5  # 200-sample window / 20 samples per batch * 0.5 mid jump


def test_class_global_latency_equals_prior_history():
    for p in [200, 400, 1000]:
        assert class_shift_latency(p, windowed=False) == p


def test_confidence_windowed_is_half_window_global_is_prior():
    for p in [200, 400, 1000]:
        w = confidence_shift_latency(p, windowed=True)
        g = confidence_shift_latency(p, windowed=False)
        assert w == CONFIG_WINDOW // 2
        assert g == p


def test_centroid_windowed_bounded_global_grows():
    wins, globs = [], []
    for p in [200, 400, 800]:
        w = centroid_shift_latency(p, windowed=True)
        g = centroid_shift_latency(p, windowed=False)
        assert 0 < w <= CONFIG_WINDOW
        assert g > w
        wins.append(w)
        globs.append(g)
    assert wins[0] == wins[1] == wins[2]
    assert globs[0] < globs[1] < globs[2]


def test_no_shift_controls_never_fire():
    assert not no_shift_class()
    assert not no_shift_confidence()
    assert not no_shift_centroid()


def test_early_stream_min_constraint():
    # P < k: min(k, n_stream) == n_stream, so windowed == global
    w, g = early_stream_latency(50)
    assert w == g
    assert w == 50