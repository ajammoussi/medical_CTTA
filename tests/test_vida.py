import pytest
import torch
import torch.nn as nn
from src.adapters.registry import AdapterRegistry
from src.adapters.base import CTTAAdapter
from src.adapters.vida import ViDAAdapter, ViDAInjectedLinear, inject_vida_adapters, symmetric_kl_loss


def test_adapter_registry():
    adapters = AdapterRegistry.list_adapters()
    assert 'vida' in adapters


def test_get_vida_class():
    vida_class = AdapterRegistry.get('vida')
    assert issubclass(vida_class, CTTAAdapter)


def test_vida_initialization():
    adapter = ViDAAdapter()
    assert adapter.vida_rank1 == 1
    assert adapter.vida_rank2 == 128
    assert adapter.uncertainty_threshold == 0.2
    assert adapter.num_augmentations == 10
    assert adapter.uncertainty_scale == 0.1
    assert adapter.alpha_teacher == 0.99
    assert adapter.alpha_vida == 0.8
    assert adapter.vida_lr == 5e-4
    assert adapter.model_lr == 5e-7
    assert adapter.restore_prob == 0.0
    assert adapter.max_grad_norm == 5.0
    assert adapter.ce_loss_weight == 1.0
    assert adapter.pseudo_label_threshold == 0.5


def test_vida_custom_params():
    adapter = ViDAAdapter(
        vida_rank1=2,
        vida_rank2=256,
        uncertainty_threshold=0.3,
        num_augmentations=5,
        vida_lr=5e-5,
        restore_prob=0.01,
        max_grad_norm=2.0,
    )
    assert adapter.vida_rank1 == 2
    assert adapter.vida_rank2 == 256
    assert adapter.uncertainty_threshold == 0.3
    assert adapter.num_augmentations == 5
    assert adapter.vida_lr == 5e-5
    assert adapter.restore_prob == 0.01
    assert adapter.max_grad_norm == 2.0


def test_vida_state_dict():
    adapter = ViDAAdapter()
    state = adapter.state_dict()
    assert 'source_state' in state
    assert 'optimizer' in state


def test_vida_injected_linear_shapes():
    in_f, out_f, r, r2 = 64, 128, 1, 16
    layer = ViDAInjectedLinear(in_f, out_f, bias=True, r=r, r2=r2)
    x = torch.randn(2, in_f)
    out = layer(x)
    assert out.shape == (2, out_f)


def test_vida_injected_linear_zero_init():
    """Adapter starts as identity (zero adapter output)."""
    in_f, out_f, r, r2 = 64, 128, 1, 16
    layer = ViDAInjectedLinear(in_f, out_f, bias=True, r=r, r2=r2)
    x = torch.randn(4, in_f)
    out = layer(x)
    expected = layer.linear_vida(x)
    assert torch.allclose(out, expected, atol=1e-6)


def test_inject_vida_adapters():
    """Test injection into a small ViT-like model."""

    class FakeAttn(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.qkv = nn.Linear(dim, dim * 3)
            self.proj = nn.Linear(dim, dim)

        def forward(self, x):
            return x

    class FakeBlock(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.attn = FakeAttn(dim)
            self.norm1 = nn.LayerNorm(dim)

        def forward(self, x):
            return x

    class FakeViT(nn.Module):
        def __init__(self, dim=32):
            super().__init__()
            self.blocks = nn.ModuleList([FakeBlock(dim) for _ in range(2)])
            self.head = nn.Linear(dim, 5)

        def forward(self, x):
            return x

    model = FakeViT(dim=32)
    injected = inject_vida_adapters(model, r=1, r2=8, target_modules=["FakeAttn"])

    assert len(injected) > 0
    # Check that qkv and proj are now ViDAInjectedLinear
    for block in model.blocks:
        assert isinstance(block.attn.qkv, ViDAInjectedLinear)
        assert isinstance(block.attn.proj, ViDAInjectedLinear)


def test_symmetric_kl_loss():
    student = torch.randn(4, 5)
    teacher = torch.randn(4, 5)
    loss = symmetric_kl_loss(student, teacher)
    assert loss.shape == (4,)
    assert (loss >= 0).all()


def test_symmetric_kl_loss_same_dist():
    """Same distribution should give zero loss."""
    logits = torch.randn(4, 5)
    loss = symmetric_kl_loss(logits, logits)
    assert torch.allclose(loss, torch.zeros_like(loss), atol=1e-6)


def test_inject_vida_adapters_device_consistency():
    """Injected adapter params must be on the same device as the model."""

    class FakeAttn(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.qkv = nn.Linear(dim, dim * 3)
            self.proj = nn.Linear(dim, dim)

        def forward(self, x):
            return x

    class FakeBlock(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.attn = FakeAttn(dim)
            self.norm1 = nn.LayerNorm(dim)

        def forward(self, x):
            return x

    class FakeViT(nn.Module):
        def __init__(self, dim=32):
            super().__init__()
            self.blocks = nn.ModuleList([FakeBlock(dim) for _ in range(2)])
            self.head = nn.Linear(dim, 5)

        def forward(self, x):
            return x

    model = FakeViT(dim=32)
    # Simulate: model already on some device, inject adapters
    inject_vida_adapters(model, r=1, r2=8, target_modules=["FakeAttn"])

    # After injection, all parameters should be on the same device
    devices = {p.device for p in model.parameters()}
    assert len(devices) == 1, f"Mixed devices: {devices}"

    # Verify forward pass works without device errors
    x = torch.randn(2, 32)
    # No assertion needed — just ensure no RuntimeError
    for block in model.blocks:
        block.attn.qkv(x)
