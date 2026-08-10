"""Phase 0 tests — C0 telemetry: windowed helpers + per-batch telemetry dict.

Exit criteria covered here:
- ``src/utils/metrics.py`` window helpers are numerically correct.
- Telemetry is a pure extension: a legacy run (``agents_enabled=false``) produces
  steps with exactly the historical ``{batch_idx, signals, metrics}`` keys and
  performs no extra forward pass (byte-identical behavior).
- ``agents_enabled=true`` adds per-batch ``telemetry`` dicts with the C0 fields.
"""

import math

import numpy as np
import pytest
import torch

from src.utils.metrics import (
    windowed_class_distribution,
    class_distribution_proportions,
    cosine_drift,
    window_centroid,
)
from src.config import CTTAConfig
from src.evaluation.runner.base import BaseRunnerMixin
from src.evaluation.metrics import quadratic_weighted_kappa


# ---------------------------------------------------------------------------
# windowed_class_distribution
# ---------------------------------------------------------------------------

def test_wcd_counts_over_window():
    assert windowed_class_distribution([0, 0, 1, 2], window=2) == {1: 1, 2: 1}


def test_wcd_window_larger_than_history():
    assert windowed_class_distribution([0, 1, 2, 3]) == {0: 1, 1: 1, 2: 1, 3: 1}


def test_wcd_window_zero_uses_all():
    assert windowed_class_distribution([0, 0, 1], window=0) == {0: 2, 1: 1}


def test_wcd_negative_window_uses_all():
    assert windowed_class_distribution([0, 0, 1], window=-5) == {0: 2, 1: 1}


def test_wcd_empty():
    assert windowed_class_distribution([]) == {}


def test_wcd_class_ids_are_ints():
    out = windowed_class_distribution([0, 0, 1], window=2)
    assert all(isinstance(k, int) for k in out)


def test_proportions_sum_to_one():
    props = class_distribution_proportions([0, 0, 1, 2], num_classes=4, window=4)
    assert props == [0.5, 0.25, 0.25, 0.0]
    assert math.isclose(sum(props), 1.0)


def test_proportions_empty_returns_zeros():
    assert class_distribution_proportions([], num_classes=5) == [0.0] * 5


# ---------------------------------------------------------------------------
# cosine_drift
# ---------------------------------------------------------------------------

def test_cosine_drift_identical():
    a = torch.tensor([1.0, 0.0, 0.0])
    assert abs(cosine_drift(a, a)) < 1e-6


def test_cosine_drift_orthogonal():
    a = torch.tensor([1.0, 0.0])
    b = torch.tensor([0.0, 1.0])
    assert abs(cosine_drift(a, b) - 1.0) < 1e-6


def test_cosine_drift_antiparallel():
    a = torch.tensor([1.0, 0.0, 0.0])
    b = torch.tensor([-1.0, 0.0, 0.0])
    assert abs(cosine_drift(a, b) - 2.0) < 1e-6


def test_cosine_drift_numpy_input():
    a = np.array([1.0, 0.0], dtype=np.float32)
    b = np.array([0.0, 1.0], dtype=np.float32)
    assert abs(cosine_drift(a, b) - 1.0) < 1e-6


def test_cosine_drift_partial_overlap():
    a = torch.tensor([1.0, 0.0])
    b = torch.tensor([1.0, 1.0])
    expected = 1.0 - 1.0 / math.sqrt(2.0)
    assert abs(cosine_drift(a, b) - expected) < 1e-6


# ---------------------------------------------------------------------------
# window_centroid
# ---------------------------------------------------------------------------

def test_window_centroid_mean_and_unit_norm():
    c = window_centroid([torch.tensor([1.0, 0.0, 0.0]),
                         torch.tensor([0.0, 1.0, 0.0])], window=2)
    assert c.shape == (3,)
    assert abs(c.norm().item() - 1.0) < 1e-6
    assert abs(c[0].item() - math.sqrt(0.5)) < 1e-5
    assert abs(c[1].item() - math.sqrt(0.5)) < 1e-5


def test_window_centroid_honors_window():
    c = window_centroid([
        torch.tensor([1.0, 0.0, 0.0]),
        torch.tensor([0.0, 1.0, 0.0]),
        torch.tensor([0.0, 0.0, 1.0]),
    ], window=2)
    # mean of last two ((0,1,0)+(0,0,1))/2 -> normalized
    assert abs(c[0].item()) < 1e-6
    assert abs(c[1].item() - math.sqrt(0.5)) < 1e-5
    assert abs(c[2].item() - math.sqrt(0.5)) < 1e-5


def test_window_centroid_empty_raises():
    with pytest.raises(ValueError):
        window_centroid([])


def test_window_centroid_numpy_vectors():
    c = window_centroid([np.array([1.0, 0.0]), np.array([0.0, 1.0])], window=2)
    assert abs(c.norm().item() - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# _step_entry — byte-identical historical shape
# ---------------------------------------------------------------------------

class _FakeConfig:
    def __init__(self, agents_enabled=False, method="cotta", streaming_window=200):
        self.model = None
        self.ctta = CTTAConfig(
            method=method,
            agents_enabled=agents_enabled,
            streaming_window=streaming_window,
        )


class _RunnerHarness(BaseRunnerMixin):
    def __init__(self, cfg):
        self.config = cfg
        self.device = torch.device("cpu")


class _CallCountingModel:
    """Regression guard: when telemetry is disabled no embedding forward happens."""

    def __init__(self, num_classes=5):
        self.num_classes = num_classes
        self.embedding_calls = 0

    def extract_embedding(self, x):
        self.embedding_calls += 1
        b = x.size(0)
        rows = torch.arange(b, dtype=torch.float32).view(b, 1).repeat(1, 8)
        return rows * 0.1

    def get_num_classes(self):
        return self.num_classes


def test_step_entry_legacy_shape():
    cfg = _FakeConfig(agents_enabled=False)
    h = _RunnerHarness(cfg)
    entry = h._step_entry(0, {"loss": 1.0}, None)
    assert list(entry.keys()) == ["batch_idx", "signals", "metrics"]


def test_step_entry_with_telemetry_appends_key():
    cfg = _FakeConfig(agents_enabled=True)
    h = _RunnerHarness(cfg)
    entry = h._step_entry(0, {"loss": 1.0}, {"batch_accuracy": 0.5})
    assert list(entry.keys()) == ["batch_idx", "signals", "metrics", "telemetry"]


def test_telemetry_disabled_is_none_and_does_no_forward():
    cfg = _FakeConfig(agents_enabled=False)
    h = _RunnerHarness(cfg)
    model = _CallCountingModel()
    images = torch.randn(4, 3, 8, 8)
    logits = torch.randn(4, 5)
    with torch.no_grad():
        tel = h._record_telemetry(
            model, batch_idx=0, images=images, logits=logits,
            batch_acc=0.25, pred_window=__import__("collections").deque(),
        )
    assert tel is None
    assert model.embedding_calls == 0


def test_telemetry_enabled_fields():
    torch.manual_seed(0)
    cfg = _FakeConfig(agents_enabled=True, method="cotta")
    h = _RunnerHarness(cfg)
    model = _CallCountingModel(num_classes=5)
    images = torch.randn(4, 3, 8, 8)
    logits = torch.randn(4, 5)
    from collections import deque
    with torch.no_grad():
        tel = h._record_telemetry(
            model, batch_idx=3, images=images, logits=logits,
            batch_acc=0.5, pred_window=deque(), adapter_metrics={},
        )

    assert tel["batch_idx"] == 3
    assert tel["n_samples"] == 4
    assert len(tel["cls_centroid_l2"]) == 8
    # cls_centroid_l2 is unit-norm (L2-normalized batch mean embedding)
    arr = torch.tensor(tel["cls_centroid_l2"])
    assert abs(float(arr.norm()) - 1.0) < 1e-5
    assert len(tel["class_distribution"]) == 5
    assert math.isclose(sum(tel["class_distribution"]), 1.0, abs_tol=1e-5)
    assert tel["batch_accuracy"] == 0.5

    # entropy / max-prob must match a direct hand computation
    probs = torch.softmax(logits, dim=-1)
    expected_ent = float((-torch.sum(probs * torch.log(probs.clamp_min(1e-12)), dim=1)).mean())
    expected_mmp = float(probs.max(dim=1).values.mean())
    assert abs(tel["pred_entropy_mean"] - expected_ent) < 1e-5
    assert abs(tel["mean_max_prob"] - expected_mmp) < 1e-5


def test_telemetry_windowed_class_distribution():
    """The rolling pred window accumulates across batches and respects window."""
    from collections import deque
    cfg = _FakeConfig(agents_enabled=True, method="cotta", streaming_window=200)
    h = _RunnerHarness(cfg)
    model = _CallCountingModel(num_classes=5)
    window = deque()

    # batch 0: logits predict classes [0, 0, 0, 1]
    logits0 = torch.zeros(4, 5)
    logits0[:, 0] = 2.0
    logits0[3, 1] = 2.0
    logits0[:, 0] = 1.0
    logits0[3, 1] = 3.0
    logits0[:, 0] = 1.0
    logits0[3, 1] = 3.0
    images = torch.randn(4, 3, 8, 8)
    with torch.no_grad():
        tel0 = h._record_telemetry(model, batch_idx=0, images=images,
                                   logits=logits0, batch_acc=0.5,
                                   pred_window=window)
    assert tel0["class_distribution"][0] == 0.75
    assert tel0["class_distribution"][1] == 0.25

    # batch 1: logits predict classes [2, 2, 2, 2]
    logits1 = torch.zeros(4, 5)
    logits1[:, 2] = 3.0
    with torch.no_grad():
        tel1 = h._record_telemetry(model, batch_idx=1, images=images,
                                   logits=logits1, batch_acc=0.5,
                                   pred_window=window)
    # window = all 8 preds: {0:3, 1:1, 2:4}
    assert tel1["class_distribution"][0] == pytest.approx(3 / 8)
    assert tel1["class_distribution"][1] == pytest.approx(1 / 8)
    assert tel1["class_distribution"][2] == pytest.approx(4 / 8)


def test_telemetry_window_truncation():
    from collections import deque
    cfg = _FakeConfig(agents_enabled=True, method="cotta", streaming_window=4)
    h = _RunnerHarness(cfg)
    model = _CallCountingModel(num_classes=2)
    window = deque()
    images = torch.randn(2, 3, 8, 8)
    for _ in range(3):
        logits = torch.zeros(2, 2); logits[:, 0] = 3.0
        with torch.no_grad():
            h._record_telemetry(model, batch_idx=0, images=images,
                                logits=logits, batch_acc=0.5,
                                pred_window=window)
    logits = torch.zeros(2, 2); logits[:, 1] = 3.0
    with torch.no_grad():
        tel = h._record_telemetry(model, batch_idx=0, images=images,
                                  logits=logits, batch_acc=0.5,
                                  pred_window=window)
    # 3 batches of class-0 (6) then 1 batch of class-1 (2); window=4 keeps last 4
    assert tel["class_distribution"][1] == pytest.approx(0.5)


def test_telemetry_vida_scales():
    from collections import deque
    cfg = _FakeConfig(agents_enabled=True, method="vida")
    h = _RunnerHarness(cfg)
    model = _CallCountingModel(num_classes=5)
    images = torch.randn(4, 3, 8, 8)
    logits = torch.randn(4, 5)
    metrics = {"lambda_low": 0.95, "lambda_high": 1.05}
    with torch.no_grad():
        tel = h._record_telemetry(
            model, batch_idx=0, images=images, logits=logits,
            batch_acc=0.5, pred_window=deque(), adapter_metrics=metrics,
        )
    assert tel["vida_scales"] == {"lambda_low": 0.95, "lambda_high": 1.05}


def test_telemetry_cotta_confident_and_restored():
    from collections import deque
    cfg = _FakeConfig(agents_enabled=True, method="cotta")
    h = _RunnerHarness(cfg)
    model = _CallCountingModel(num_classes=5)
    images = torch.randn(4, 3, 8, 8)
    logits = torch.randn(4, 5)
    metrics = {"num_confident_samples": 7, "restoration_count": 120}
    with torch.no_grad():
        tel = h._record_telemetry(
            model, batch_idx=0, images=images, logits=logits,
            batch_acc=0.5, pred_window=deque(), adapter_metrics=metrics,
        )
    assert tel["n_confident"] == 7
    assert tel["restored"] == 120


def test_adapt_log_line_telemetry_suffix():
    cfg = _FakeConfig(agents_enabled=True, method="cotta")
    h = _RunnerHarness(cfg)
    line = h._adapt_log_line(1, {"loss": 0.5}, {"pred_entropy_mean": 1.1,
                                                "mean_max_prob": 0.8,
                                                "batch_accuracy": 0.4})
    assert "[tel]" in line
    assert "ent=1.1000" in line

    legacy = h._adapt_log_line(1, {"loss": 0.5}, None)
    assert "Adapt step 1" in legacy
    assert "[tel]" not in legacy