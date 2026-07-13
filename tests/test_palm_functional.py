"""Functional test for PALM adapter with a mock model."""

import torch
import torch.nn as nn
from src.adapters.palm import PALMAdapter


class MockModel:
    def __init__(self):
        self.n = nn.Parameter(torch.randn(5, 5))
        self.p = nn.Parameter(torch.randn(5))
        self.q = nn.Parameter(torch.randn(5, 5))
        self.frozen = nn.Parameter(torch.randn(3), requires_grad=False)

    def named_parameters(self):
        yield 'backbone.n', self.n
        yield 'backbone.p', self.p
        yield 'backbone.q', self.q
        yield 'classifier.w', self.frozen

    def train(self):
        pass

    def eval(self):
        pass

    def forward(self, x):
        return x @ self.n.T + self.p[:x.shape[0], None] + x @ self.q.T * 0.01

    def get_num_classes(self):
        return 5


def test_palm_step():
    model = MockModel()
    adapter = PALMAdapter(
        lr=1e-3, temperature=5.0, selection_percentile=0.5,
        consistency_lambda=0.0, num_augmentations=0,
        entropy_threshold_factor=10.0,  # high threshold → all samples confident
    )
    adapter.setup(model)

    batch = torch.randn(4, 5)
    logits = model.forward(batch)
    metrics = adapter.adapt_step(batch, logits)

    assert isinstance(metrics, dict)
    assert 'loss' in metrics
    assert 'kl_loss' in metrics
    assert 'selected_fraction' in metrics
    assert 0 <= metrics['selected_fraction'] <= 1.0
    assert metrics['n_selected_params'] > 0
    assert metrics['n_total_params'] > 0
    assert metrics['loss'] >= 0


def test_palm_multiple_steps():
    model = MockModel()
    adapter = PALMAdapter(
        lr=1e-3, selection_percentile=0.5,
        consistency_lambda=0.0, num_augmentations=0,
    )
    adapter.setup(model)

    for step in range(3):
        batch = torch.randn(4, 5)
        logits = model.forward(batch)
        _ = adapter.adapt_step(batch, logits)
    # Verify running_sensitivity was updated across steps
    assert len(adapter.running_sensitivity) == 3


def test_palm_state_dict():
    model = MockModel()
    adapter = PALMAdapter(
        lr=1e-3, selection_percentile=0.5,
        consistency_lambda=0.0, num_augmentations=0,
    )
    adapter.setup(model)

    batch = torch.randn(4, 5)
    logits = model.forward(batch)
    adapter.adapt_step(batch, logits)

    state = adapter.state_dict()
    assert 'running_sensitivity' in state
    assert 'optimizer' in state
    assert len(state['running_sensitivity']) > 0


def test_palm_restore_noop():
    model = MockModel()
    adapter = PALMAdapter(
        lr=1e-3, consistency_lambda=0.0, num_augmentations=0,
    )
    adapter.setup(model)
    adapter.restore_parameters()
    # Should not crash — PALM doesn't use restore


def test_palm_all_params_selected():
    """Verify that with high percentile, all params get selected."""
    model = MockModel()
    adapter = PALMAdapter(
        lr=1e-3, selection_percentile=1.0,
        consistency_lambda=0.0, num_augmentations=0,
    )
    adapter.setup(model)

    batch = torch.randn(4, 5)
    logits = model.forward(batch)
    metrics = adapter.adapt_step(batch, logits)

    # All 3 trainable params participate in forward → all selected
    assert metrics['n_selected_params'] == 3
    assert metrics['selected_fraction'] == 1.0
