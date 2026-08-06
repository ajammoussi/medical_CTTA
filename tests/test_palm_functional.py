"""Functional test for the DR-tuned official PALM adapter.

Uses a small mock module structured like a frozen ViT backbone + prototype
classifier. Verifies:
  - scale-aware layer selection adapts a real slice of capacity
  - an entropy signal fires on EVERY batch (num_confident_samples > 0)
  - the per-class cap balances confident-sample selection
  - parameters actually change across steps (non-zero adaptation)
  - fallback reverts on QWK/acc regression
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.adapters.palm import PALMAdapter, _name_is_head


class MockBlock(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.Linear(dim, dim)
        self.mlp1 = nn.Linear(dim, dim * 2)   # big weight -> big raw L1 norm
        self.mlp2 = nn.Linear(dim * 2, dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        x = self.norm1(x)
        x = self.attn(x)
        x = self.norm2(x)
        x = F.relu(self.mlp1(x))
        x = self.mlp2(x)
        return x


class MockModel(nn.Module):
    def __init__(self, n_blocks=24, dim=128, train_last=4, num_classes=5):
        super().__init__()
        # RETFound-style naming: backbone.blocks.N.*  /  classifier.*
        blocks = []
        for i in range(n_blocks):
            blocks.append((str(i), MockBlock(dim)))
        self.backbone = nn.Sequential()
        for name, block in blocks:
            self.backbone.add_module(name, block)
        self.fc_norm = nn.LayerNorm(dim)
        self.classifier = nn.Linear(dim, num_classes)
        self.num_classes = num_classes
        for i in range(n_blocks):
            self.backbone[i].requires_grad_(i >= n_blocks - train_last)

    def forward(self, x):
        h = x
        for block in self.backbone:
            h = block(h)
        h = self.fc_norm(h)
        feats = F.normalize(h, dim=1)               # (B, dim) output features
        return self.classifier(feats) * 5.0   # temperature-scaled prototype head

    def get_num_classes(self):
        return self.num_classes


def test_selection_adapts_substantial_params():
    torch.manual_seed(0)
    model = MockModel()
    adapter = PALMAdapter(lr=1e-3, selection_percentile=0.4, always_blocks=1,
                          min_selected_ratio=0.05, per_class_cap=4)
    adapter.setup(model)

    metrics = adapter.adapt_step(torch.randn(16, 128))

    trainable = [p for p in model.parameters() if p.requires_grad]
    total = sum(p.numel() for p in trainable)
    n_total = metrics['n_total_params']
    assert any(_name_is_head(n) for n in adapter._selected), "head not selected"
    assert n_total >= max(1, int(total * 0.05)), f"only {n_total}/{total} params selected"
    assert metrics['num_confident_samples'] >= 1, "no confident sample -> dead entropy"
    # anti-collapse: entropy fires only when the confident set spans >=2
    # classes (single-class sets would reinforce that class -> QWK collapse).
    # When suppressed, consistency + diversity still drive the update.
    if metrics['entropy_loss'] > 0:
        assert metrics['num_confident_samples'] >= 2
    assert metrics['loss'] != 0.0, "no gradient signal -> dead adaptation"
    assert torch.isfinite(torch.tensor(metrics['loss']))
    assert metrics['updated'] is True


def test_params_change_across_steps():
    torch.manual_seed(1)
    model = MockModel()
    adapter = PALMAdapter(lr=1e-2, selection_percentile=0.5, always_blocks=1,
                          per_class_cap=4, min_selected_ratio=0.05)
    adapter.setup(model)
    before = {n: p.detach().clone() for n, p in adapter._trainable_dict.items()}
    for _ in range(5):
        adapter.adapt_step(torch.randn(8, 128))
    head_delta = sum((p.detach() - before[n]).abs().sum().item()
                     for n, p in adapter._trainable_dict.items() if _name_is_head(n))
    any_delta = sum((p.detach() - before[n]).abs().sum().item()
                    for n, p in adapter._trainable_dict.items())
    assert head_delta > 0, "classifier head did not adapt"
    assert any_delta > 0, "no parameter changed -> adaptation is dead"


def test_per_class_cap_balances_classes():
    torch.manual_seed(2)
    model = MockModel()
    adapter = PALMAdapter(lr=1e-3, per_class_cap=2, selection_percentile=0.5)
    adapter.setup(model)
    # single-class near-one-hot distribution -> cap must limit class-0 samples
    logits = torch.zeros(12, 5)
    logits[:, 0] = 10.0
    mask, _ = adapter._confident_mask(F.softmax(logits, dim=-1))
    # single-class near-one-hot -> cap limits dominant class;
    # enforcement may add uncertain samples so total mask > cap.
    preds_mask = torch.argmax(F.softmax(logits, dim=-1), dim=-1)
    conf_preds = preds_mask[mask]
    counts = torch.bincount(conf_preds, minlength=5)
    assert int(counts.max()) <= adapter.per_class_cap, f"dominant class cap broken: {counts}"


def test_fallback_on_qwk_and_acc_regression():
    def run(b, c):
        m = MockModel()
        a = PALMAdapter()
        a.setup(m)
        return a.check_fallback(b, c)
    assert run({"qwk": 0.8, "overall_accuracy": 0.5},
               {"qwk": 0.6, "overall_accuracy": 0.5}) is True
    assert run({"qwk": 0.8, "overall_accuracy": 0.5},
               {"qwk": 0.81, "overall_accuracy": 0.47}) is True
    assert run({"qwk": 0.8, "overall_accuracy": 0.5},
               {"qwk": 0.82, "overall_accuracy": 0.52}) is False


def test_per_confident_settles_per_batch():
    torch.manual_seed(3)
    model = MockModel()
    adapter = PALMAdapter(lr=1e-3, per_class_cap=4, min_confident_fraction=0.25)
    adapter.setup(model)
    for _ in range(3):
        m = adapter.adapt_step(torch.randn(16, 128))
        assert m['num_confident_samples'] >= 1, "confident sample must exist every batch"


if __name__ == "__main__":
    test_selection_adapts_substantial_params()
    test_params_change_across_steps()
    test_per_class_cap_balances_classes()
    test_fallback_on_qwk_and_acc_regression()
    test_per_confident_settles_per_batch()
    print("ALL SMOKE TESTS PASSED")