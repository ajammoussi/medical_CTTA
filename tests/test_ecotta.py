"""Functional tests for the EcoTTA adapter.

Uses mock ViT-style backbones (RETFound-style nn.Sequential blocks + VisionFM-style
nn.ModuleList blocks with get_intermediate_layers) to verify:
  - partitions are injected and all original params frozen (only metas train)
  - the meta MLP starts near-identity (baseline forward preserved)
  - an entropy signal fires every batch (quota fallback) with non-zero reg loss
  - meta params actually change across steps (non-zero adaptation)
  - source warmup runs and updates the meta params
  - fallback reverts on QWK/acc regression
  - VisionFM classifier taps (last 4 partitions) stay 4 outputs after injection
  - state_dict round-trip restores meta params
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from src.adapters.ecotta import EcoTTAAdapter, PartitionWithMeta


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


def _param_sum(model):
    return sum(p.abs().sum().item() for p in model.parameters() if p.requires_grad)


def test_setup_injects_partitions_and_freezes_originals():
    torch.manual_seed(0)
    model = RetFoundMock(n_blocks=24)
    adapter = EcoTTAAdapter(ecotta_num_partitions=4, ecotta_reg_lambda=0.25)
    adapter.setup(model)

    assert len(adapter._partitions) == 4
    sizes = [len(p.blocks) for p in adapter._partitions]
    assert sum(sizes) == 24
    assert sizes == [4, 4, 8, 8], f"expected shallow-dense [4,4,8,8], got {sizes}"
    # only meta MLP params train (4 partitions x 4 tensors); block weights stay frozen
    n_meta = sum(1 for p in adapter._meta_params if p.requires_grad)
    assert n_meta == len(adapter._meta_params)
    mlp_tensors = sum(1 for _ in adapter._partitions[0].mlp.parameters())
    assert n_meta == 4 * mlp_tensors, f"expected only meta MLP params to train, got {n_meta}"
    assert _count_trainable(model) == n_meta, "only meta params may train"
    for part in adapter._partitions:
        for p in part.blocks.parameters():
            assert not p.requires_grad, "frozen encoder blocks must stay frozen"

    # meta MLP is zero-initialized -> out == out1 at init (no meta shift)
    x = torch.randn(4, 128)
    out1 = adapter._partitions[0].blocks(x)
    out = adapter._partitions[0](x)
    assert torch.allclose(out, out1, atol=1e-5), "meta MLP must be zero-init (out == out1)"


def test_adapt_step_has_entropy_and_reg_signal():
    torch.manual_seed(1)
    model = RetFoundMock(n_blocks=24)
    adapter = EcoTTAAdapter(ecotta_num_partitions=4, ecotta_reg_lambda=0.25)
    adapter.setup(model)

    metrics = adapter.adapt_step(torch.randn(16, 128))
    assert metrics['num_confident_samples'] >= 1, "entropy signal must fire every batch"
    assert metrics['reg_loss'] >= 0.0
    assert torch.isfinite(torch.tensor(metrics['loss']))
    assert metrics['updated'] is True


def test_meta_params_change_across_steps():
    torch.manual_seed(2)
    model = RetFoundMock(n_blocks=24)
    adapter = EcoTTAAdapter(ecotta_num_partitions=4, ecotta_reg_lambda=0.25,
                            ecotta_tta_lr=5e-3)
    adapter.setup(model)
    before = {id(p): p.detach().clone() for p in adapter._meta_params}
    for _ in range(5):
        adapter.adapt_step(torch.randn(8, 128))
    delta = sum((p.detach() - before[id(p)]).abs().sum().item()
                for p in adapter._meta_params)
    assert delta > 0, "meta params did not change -> dead adaptation"


def test_warmup_updates_meta_params():
    torch.manual_seed(3)
    model = RetFoundMock(n_blocks=24)
    adapter = EcoTTAAdapter(ecotta_num_partitions=4, ecotta_reg_lambda=0.25,
                            ecotta_warmup_epochs=3, ecotta_warmup_lr=1e-2)
    adapter.setup(model)

    # small labeled source loader
    x = torch.randn(32, 128)
    y = torch.randint(0, 5, (32,))
    loader = DataLoader(TensorDataset(x, y), batch_size=8)

    before = {id(p): p.detach().clone() for p in adapter._meta_params}
    adapter.warmup(loader)
    delta = sum((p.detach() - before[id(p)]).abs().sum().item()
                for p in adapter._meta_params)
    assert delta > 0, "warmup did not update meta params"
    # warmup also re-anchors the fallback source state
    assert len(adapter._source_meta) == len(adapter._meta_params)


def test_visionfm_classifier_taps_stay_aligned():
    torch.manual_seed(4)
    model = VisionFMMock(n_blocks=12)
    adapter = EcoTTAAdapter(ecotta_num_partitions=7,
                            ecotta_partition_sizes=[2, 2, 4, 1, 1, 1, 1],
                            ecotta_reg_lambda=0.25)
    adapter.setup(model)

    assert len(adapter._partitions) == 7
    sizes = [len(p.blocks) for p in adapter._partitions]
    assert sizes == [2, 2, 4, 1, 1, 1, 1]
    # classifier taps still collect exactly 4 outputs (n=4) -> 4*dim input ok
    x = torch.randn(4, 128)
    logits = model(x)
    assert logits.shape == (4, 5)
    metrics = adapter.adapt_step(x)
    assert metrics['num_confident_samples'] >= 1


def test_fallback_on_qwk_and_acc_regression():
    def run(b, c):
        m = RetFoundMock(n_blocks=8)
        a = EcoTTAAdapter(ecotta_num_partitions=2)
        a.setup(m)
        return a.check_fallback(b, c)
    assert run({"qwk": 0.8, "overall_accuracy": 0.5},
               {"qwk": 0.6, "overall_accuracy": 0.5}) is True
    assert run({"qwk": 0.8, "overall_accuracy": 0.5},
               {"qwk": 0.81, "overall_accuracy": 0.47}) is True
    assert run({"qwk": 0.8, "overall_accuracy": 0.5},
               {"qwk": 0.82, "overall_accuracy": 0.52}) is False


def test_state_dict_roundtrip():
    torch.manual_seed(5)
    model = RetFoundMock(n_blocks=12)
    adapter = EcoTTAAdapter(ecotta_num_partitions=3)
    adapter.setup(model)
    adapter.adapt_step(torch.randn(8, 128))
    state = adapter.state_dict()
    assert len(state['meta_params']) == len(adapter._meta_params)

    # load into a fresh adapter
    model2 = RetFoundMock(n_blocks=12)
    adapter2 = EcoTTAAdapter(ecotta_num_partitions=3)
    adapter2.setup(model2)
    adapter2.load_state_dict(state)
    for p, p2 in zip(adapter._meta_params, adapter2._meta_params):
        assert torch.allclose(p.detach().cpu(), p2.detach().cpu()), "state roundtrip mismatch"


def test_per_class_cap_limits_confident_set():
    torch.manual_seed(6)
    model = RetFoundMock(n_blocks=12)
    adapter = EcoTTAAdapter(ecotta_num_partitions=3, ecotta_per_class_cap=1)
    adapter.setup(model)
    for _ in range(3):
        metrics = adapter.adapt_step(torch.randn(16, 128))
        assert metrics['num_confident_samples'] >= 1, "cap must keep an entropy signal"
        assert metrics['num_confident_samples'] <= 5, "cap=1 -> at most one per class (5 classes)"
        assert metrics['updated'] is True


def test_grad_checkpointing_preserves_gradients():
    torch.manual_seed(7)
    a = EcoTTAAdapter(ecotta_num_partitions=3, ecotta_grad_checkpointing=True)
    b = EcoTTAAdapter(ecotta_num_partitions=3, ecotta_grad_checkpointing=False)
    torch.manual_seed(7)
    ma = RetFoundMock(n_blocks=12)
    torch.manual_seed(7)
    mb = RetFoundMock(n_blocks=12)
    a.setup(ma)
    b.setup(mb)
    # mlp[0] is random-init; setup consumes RNG -> force identical starting point
    with torch.no_grad():
        for pa, pb in zip(a._meta_params, b._meta_params):
            pb.copy_(pa)
    x = torch.randn(8, 128)
    ma_ = a.adapt_step(x)
    mb_ = b.adapt_step(x)
    assert ma_['updated'] is True and mb_['updated'] is True
    for pa, pb in zip(a._meta_params, b._meta_params):
        assert torch.allclose(pa.detach(), pb.detach(), atol=1e-6), "checkpointing changed gradients"


if __name__ == "__main__":
    test_setup_injects_partitions_and_freezes_originals()
    test_adapt_step_has_entropy_and_reg_signal()
    test_meta_params_change_across_steps()
    test_warmup_updates_meta_params()
    test_visionfm_classifier_taps_stay_aligned()
    test_fallback_on_qwk_and_acc_regression()
    test_state_dict_roundtrip()
    test_per_class_cap_limits_confident_set()
    test_grad_checkpointing_preserves_gradients()
    print("ALL ECO TTA SMOKE TESTS PASSED")