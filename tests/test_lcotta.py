"""Functional tests for the LCoTTA adapter.

Uses mock ViT-style backbones (RETFound-style nn.Sequential blocks + VisionFM-style
nn.ModuleList blocks with get_intermediate_layers) to verify:
  - only normalization affine params train (everything else frozen)
  - an entropy signal fires every batch (quota fallback) with finite loss
  - norm params actually change across steps (non-zero adaptation)
  - the gradient queue fills, respects its cap, and the subspace activates
  - projected updates keep a bounded projection-norm ratio
  - fallback reverts on QWK/acc regression
  - VisionFM-style backbone works unchanged
  - state_dict round-trip restores params + queue
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.adapters.lcotta import LCoTTAAdapter


class MockBlock(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.Linear(dim, dim)
        self.mlp1 = nn.Linear(dim, dim * 2)
        self.mlp2 = nn.Linear(dim * 2, dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        x = self.norm1(x)
        x = self.attn(x)
        x = self.norm2(x)
        x = F.relu(self.mlp1(x))
        x = self.mlp2(x)
        return x


class RetFoundMock(nn.Module):
    """RETFound-style: backbone.blocks is nn.Sequential (timm ViT)."""

    def __init__(self, n_blocks=24, dim=128, num_classes=5):
        super().__init__()
        self.backbone = nn.Module()
        self.backbone.blocks = nn.Sequential(*[MockBlock(dim) for _ in range(n_blocks)])
        self.backbone.embed_dim = dim
        self.classifier = nn.Linear(dim, num_classes)
        self.num_classes = num_classes

    def forward(self, x):
        h = self.backbone.blocks(x)
        feats = F.normalize(h, dim=1)
        return self.classifier(feats) * 10.0   # temperature-scaled prototype head

    def get_num_classes(self):
        return self.num_classes


class VisionFMMock(nn.Module):
    """VisionFM-style: backbone.blocks is nn.ModuleList; classifier consumes
    concat of the last 4 block outputs (n=4 taps, 4*dim input)."""

    def __init__(self, n_blocks=12, dim=128, num_classes=5):
        super().__init__()
        self.backbone = nn.Module()
        self.backbone.blocks = nn.ModuleList([MockBlock(dim) for _ in range(n_blocks)])
        self.backbone.embed_dim = dim
        self.classifier = nn.Linear(dim * 4, num_classes)
        self.num_classes = num_classes

    def _get_intermediate_layers(self, x, n=4):
        out = []
        for i, blk in enumerate(self.backbone.blocks):
            x = blk(x)
            if i >= len(self.backbone.blocks) - n:
                out.append(x)
        return out

    def forward(self, x):
        inter = self._get_intermediate_layers(x, n=4)
        feats = F.normalize(torch.cat(inter, dim=-1), dim=1)
        return self.classifier(feats) * 10.0

    def get_num_classes(self):
        return self.num_classes


def _count_trainable(model):
    return sum(1 for p in model.parameters() if p.requires_grad)


def test_setup_trains_only_norm_params():
    torch.manual_seed(0)
    model = RetFoundMock(n_blocks=8)
    adapter = LCoTTAAdapter(lr=1e-4, lcotta_subspace_dim=3,
                            lcotta_queue_length=5, lcotta_sample_interval=1)
    adapter.setup(model)

    # 8 blocks x 2 LayerNorms x 2 tensors (weight+bias) = 32 trainable tensors
    assert len(adapter._params) == 32
    assert all(p.requires_grad for p in adapter._params)
    assert _count_trainable(model) == 32, "only norm-layer affine params may train"
    # classifier / linear weights stay frozen
    assert not model.classifier.weight.requires_grad
    assert not model.backbone.blocks[0].attn.weight.requires_grad


def test_adapt_step_signal_fires():
    torch.manual_seed(1)
    model = RetFoundMock(n_blocks=8)
    adapter = LCoTTAAdapter(lr=1e-4, lcotta_subspace_dim=3,
                            lcotta_queue_length=5, lcotta_sample_interval=1)
    adapter.setup(model)

    metrics = adapter.adapt_step(torch.randn(16, 128))
    assert metrics['num_confident_samples'] >= 1, "entropy signal must fire every batch"
    assert torch.isfinite(torch.tensor(metrics['loss']))
    assert metrics['updated'] is True


def test_norm_params_change_across_steps():
    torch.manual_seed(2)
    model = RetFoundMock(n_blocks=8)
    adapter = LCoTTAAdapter(lr=5e-3, lcotta_subspace_dim=3,
                            lcotta_queue_length=5, lcotta_sample_interval=1)
    adapter.setup(model)
    before = {id(p): p.detach().clone() for p in adapter._params}
    for _ in range(5):
        adapter.adapt_step(torch.randn(8, 128))
    delta = sum((p.detach() - before[id(p)]).abs().sum().item()
                for p in adapter._params)
    assert delta > 0, "norm params did not change -> dead adaptation"


def test_subspace_activates_and_queue_is_capped():
    torch.manual_seed(3)
    model = RetFoundMock(n_blocks=8)
    adapter = LCoTTAAdapter(lr=1e-4, lcotta_subspace_dim=3,
                            lcotta_queue_length=5, lcotta_sample_interval=1)
    adapter.setup(model)

    saw_active = False
    for i in range(10):
        m = adapter.adapt_step(torch.randn(8, 128))
        assert m['queue_len'] <= 5, "queue must be capped at queue_length"
        if m['subspace_active']:
            saw_active = True
            assert 0.0 <= m['proj_norm_ratio'] <= 1.0 + 1e-3, \
                "projection cannot amplify the gradient norm"
    assert saw_active, "subspace never activated after queue filled"
    # first batch has no queue entry yet (official skips batch 0)
    assert adapter._batch_num == 10


def test_projection_suppresses_out_of_subspace_component():
    """Direct unit check of the subspace math: projecting a vector that lies in
    the span of the queue reconstructs it; an orthogonal vector is nulled."""
    torch.manual_seed(4)
    model = RetFoundMock(n_blocks=2)
    adapter = LCoTTAAdapter(lr=1e-4, lcotta_subspace_dim=2,
                            lcotta_queue_length=4, lcotta_sample_interval=1)
    adapter.setup(model)

    n = adapter._numel
    v1 = torch.zeros(n); v1[0] = 1.0
    v2 = torch.zeros(n); v2[1] = 1.0
    adapter._grad_queue = [v1.clone(), v2.clone()]
    P = adapter._compute_subspace()
    assert P.shape == (2, n)
    in_span = v1 + v2
    out_of_span = torch.zeros(n); out_of_span[2] = 1.0
    proj_in = P.T @ (P @ in_span)
    proj_out = P.T @ (P @ out_of_span)
    assert torch.allclose(proj_in, in_span, atol=1e-4), "in-span component must be preserved"
    assert proj_out.norm() < 1e-4, "orthogonal (ED-like) component must be suppressed"


def test_visionfm_mock_works():
    torch.manual_seed(5)
    model = VisionFMMock(n_blocks=12)
    adapter = LCoTTAAdapter(lr=1e-4, lcotta_subspace_dim=3,
                            lcotta_queue_length=5, lcotta_sample_interval=1)
    adapter.setup(model)
    x = torch.randn(8, 128)
    logits = model(x)
    assert logits.shape == (8, 5)
    metrics = adapter.adapt_step(x)
    assert metrics['num_confident_samples'] >= 1
    assert metrics['updated'] is True


def test_fallback_on_qwk_and_acc_regression():
    def run(b, c):
        m = RetFoundMock(n_blocks=4)
        a = LCoTTAAdapter()
        a.setup(m)
        return a.check_fallback(b, c)
    assert run({"qwk": 0.8, "overall_accuracy": 0.5},
               {"qwk": 0.6, "overall_accuracy": 0.5}) is True
    assert run({"qwk": 0.8, "overall_accuracy": 0.5},
               {"qwk": 0.81, "overall_accuracy": 0.47}) is True
    assert run({"qwk": 0.8, "overall_accuracy": 0.5},
               {"qwk": 0.82, "overall_accuracy": 0.52}) is False


def test_state_dict_roundtrip():
    torch.manual_seed(6)
    model = RetFoundMock(n_blocks=6)
    adapter = LCoTTAAdapter(lr=1e-4, lcotta_subspace_dim=2,
                            lcotta_queue_length=4, lcotta_sample_interval=1)
    adapter.setup(model)
    for _ in range(6):
        adapter.adapt_step(torch.randn(8, 128))
    state = adapter.state_dict()
    assert len(state['params']) == len(adapter._params)
    assert len(state['grad_queue']) > 0

    model2 = RetFoundMock(n_blocks=6)
    adapter2 = LCoTTAAdapter(lr=1e-4, lcotta_subspace_dim=2,
                             lcotta_queue_length=4, lcotta_sample_interval=1)
    adapter2.setup(model2)
    adapter2.load_state_dict(state)
    for p, p2 in zip(adapter._params, adapter2._params):
        assert torch.allclose(p.detach().cpu(), p2.detach().cpu()), "state roundtrip mismatch"
    assert len(adapter2._grad_queue) == len(adapter._grad_queue)
    assert adapter2._batch_num == adapter._batch_num


def test_per_class_cap_limits_confident_set():
    torch.manual_seed(7)
    model = RetFoundMock(n_blocks=6)
    adapter = LCoTTAAdapter(lr=1e-4, lcotta_per_class_cap=1,
                            lcotta_subspace_dim=2, lcotta_queue_length=4,
                            lcotta_sample_interval=1)
    adapter.setup(model)
    for _ in range(3):
        metrics = adapter.adapt_step(torch.randn(16, 128))
        assert metrics['num_confident_samples'] >= 1, "cap must keep an entropy signal"
        assert metrics['num_confident_samples'] <= 5, "cap=1 -> at most one per class (5 classes)"
        assert metrics['updated'] is True


def test_adapt_head_includes_classifier_with_lr_group():
    torch.manual_seed(9)
    model = RetFoundMock(n_blocks=4)
    adapter = LCoTTAAdapter(lr=1e-3, lcotta_adapt_head=True,
                            lcotta_head_lr_multiplier=10.0,
                            lcotta_subspace_dim=2, lcotta_queue_length=4,
                            lcotta_sample_interval=1)
    adapter.setup(model)
    assert any(p is model.classifier.weight for p in adapter._params), \
        "classifier weight must be trainable when lcotta_adapt_head=True"
    assert model.classifier.weight.requires_grad
    lrs = [g["lr"] for g in adapter._optimizer.param_groups]
    assert len(lrs) == 2, "expected separate norm/head param groups"
    assert lrs[1] == pytest.approx(lrs[0] * 10.0)
    before = model.classifier.weight.detach().clone()
    for _ in range(3):
        m = adapter.adapt_step(torch.randn(8, 128))
        assert m['updated'] is True
    delta = (model.classifier.weight.detach() - before).abs().sum().item()
    assert delta > 0, "head did not move -> dead head adaptation"


def test_quota_enforced_even_when_margin_selects_everything():
    """Overconfident models sit entirely below the margin -> the quota must
    still guarantee a rich confident set every batch (dead-adaptation fix)."""
    torch.manual_seed(10)
    model = RetFoundMock(n_blocks=6)
    # margin ~0 -> margin filter selects nothing; only the quota tops up
    adapter = LCoTTAAdapter(lr=1e-4, lcotta_entropy_margin=0.0,
                            lcotta_min_confident_fraction=0.25,
                            lcotta_subspace_dim=2, lcotta_queue_length=4,
                            lcotta_sample_interval=1)
    adapter.setup(model)
    for _ in range(3):
        m = adapter.adapt_step(torch.randn(16, 128))
        assert m['num_confident_samples'] >= 4, \
            "quota of 25% of a 16-sample batch must be enforced"


def test_adam_optimizer_option():
    torch.manual_seed(11)
    model = RetFoundMock(n_blocks=6)
    adapter = LCoTTAAdapter(lr=1e-4, lcotta_optimizer="adam",
                            lcotta_adapt_head=True,
                            lcotta_subspace_dim=2, lcotta_queue_length=4,
                            lcotta_sample_interval=1)
    adapter.setup(model)
    import torch.optim as optim
    assert isinstance(adapter._optimizer, optim.Adam)
    before = [p.detach().clone() for p in adapter._params]
    for _ in range(3):
        m = adapter.adapt_step(torch.randn(8, 128))
        assert m['updated'] is True
    delta = sum((p.detach() - b).abs().sum().item()
                for p, b in zip(adapter._params, before))
    assert delta > 0, "adam optimizer produced no updates"


def test_balanced_quota_spans_classes():
    """Global top-k would pick all majority-grade samples; the round-robin
    quota must spread selections across predicted classes."""
    entropies = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    preds = torch.tensor([0, 0, 0, 0, 0, 1, 2, 3])   # 5 lowest all class 0
    chosen = LCoTTAAdapter._balanced_quota_indices(entropies, preds, 4)
    labels = {preds[i].item() for i in chosen}
    assert len(chosen) == 4
    assert labels == {0, 1, 2, 3}, \
        f"round-robin must cover all predicted classes, got {labels}"
    # majority class contributes first (lowest entropy), then one per class
    assert chosen[0] == 0


def test_prior_alignment_term_in_loss():
    torch.manual_seed(12)
    model = RetFoundMock(n_blocks=6)
    adapter = LCoTTAAdapter(lr=1e-4, lcotta_prior_alignment_weight=0.3,
                            lcotta_subspace_dim=2, lcotta_queue_length=4,
                            lcotta_sample_interval=1)
    adapter.setup(model)
    m = adapter.adapt_step(torch.randn(16, 128))
    assert 'prior_loss' in m
    assert m['prior_loss'] >= 0.0
    assert m['updated'] is True


def test_head_anchor_bounds_head_drift():
    """With a strong anchor, even aggressive lr must keep the head near its
    source weights (the per-class-erosion protection mechanism)."""
    def run(anchor_w):
        torch.manual_seed(13)
        model = RetFoundMock(n_blocks=6)
        # de-saturate the head (*10 -> *3) so entropies are non-zero and
        # gradients actually flow (mirrors real DR confidence levels)
        with torch.no_grad():
            model.classifier.weight.mul_(0.3)
        adapter = LCoTTAAdapter(lr=1e-2, lcotta_adapt_head=True,
                                lcotta_optimizer="adam",
                                lcotta_head_anchor_weight=anchor_w,
                                lcotta_subspace_dim=2, lcotta_queue_length=4,
                                lcotta_sample_interval=1)
        adapter.setup(model)
        source = [p.detach().clone() for p in model.classifier.parameters()]
        for _ in range(5):
            adapter.adapt_step(torch.randn(16, 128))
        return sum((p.detach() - s).abs().sum().item()
                   for p, s in zip(model.classifier.parameters(), source))

    unanchored = run(0.0)
    anchored = run(0.3)
    assert unanchored > 0
    # steady-state drift under a per-step pull lam is ~step*(1-lam)/lam,
    # i.e. well below the unanchored accumulation over the same steps
    assert anchored < 0.5 * unanchored, \
        f"anchor failed to bound head drift: {anchored} vs {unanchored}"


def test_restore_parameters_reverts_to_source():
    torch.manual_seed(8)
    model = RetFoundMock(n_blocks=4)
    adapter = LCoTTAAdapter(lr=5e-3, lcotta_subspace_dim=2,
                            lcotta_queue_length=4, lcotta_sample_interval=1)
    adapter.setup(model)
    source = [p.detach().clone() for p in adapter._params]
    for _ in range(3):
        adapter.adapt_step(torch.randn(8, 128))
    delta = sum((p.detach() - s).abs().sum().item()
                for p, s in zip(adapter._params, source))
    assert delta > 0
    adapter.restore_parameters()
    for p, s in zip(adapter._params, source):
        assert torch.allclose(p.detach(), s), "restore must revert to source weights"


if __name__ == "__main__":
    test_setup_trains_only_norm_params()
    test_adapt_step_signal_fires()
    test_norm_params_change_across_steps()
    test_subspace_activates_and_queue_is_capped()
    test_projection_suppresses_out_of_subspace_component()
    test_visionfm_mock_works()
    test_fallback_on_qwk_and_acc_regression()
    test_state_dict_roundtrip()
    test_per_class_cap_limits_confident_set()
    test_adapt_head_includes_classifier_with_lr_group()
    test_quota_enforced_even_when_margin_selects_everything()
    test_adam_optimizer_option()
    test_balanced_quota_spans_classes()
    test_prior_alignment_term_in_loss()
    test_head_anchor_bounds_head_drift()
    test_restore_parameters_reverts_to_source()
    print("ALL LCOTTA SMOKE TESTS PASSED")
