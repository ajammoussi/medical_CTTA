import pytest
import torch
from src.adapters.registry import AdapterRegistry
from src.adapters.base import CTTAAdapter
from src.adapters.cotta import CoTTAAdapter
from src.adapters.palm import PALMAdapter


def test_adapter_registry():
    adapters = AdapterRegistry.list_adapters()
    assert 'cotta' in adapters
    assert 'palm' in adapters


def test_get_cotta_class():
    cotta_class = AdapterRegistry.get('cotta')
    assert issubclass(cotta_class, CTTAAdapter)


def test_get_palm_class():
    palm_class = AdapterRegistry.get('palm')
    assert issubclass(palm_class, CTTAAdapter)


def test_invalid_adapter():
    with pytest.raises(ValueError):
        AdapterRegistry.get('invalid_adapter')


def test_cotta_initialization():
    adapter = CoTTAAdapter()
    assert adapter.ema_alpha == 0.999
    assert adapter.restore_prob == 0.01
    assert adapter.num_augmentations == 6
    assert adapter.confidence_threshold == 0.65
    assert adapter.lr == 5e-4
    assert adapter.weight_decay == 0.0
    assert adapter.max_grad_norm == 1.0
    assert adapter.teacher_temperature == 0.5
    assert adapter.entropy_weight == 0.1
    assert adapter.diversity_weight == 0.05
    assert adapter.adapt_layernorm is True
    assert adapter.adapt_last_n_blocks == 6
    assert adapter.head_lr_multiplier == 5.0


def test_cotta_custom_lr():
    adapter = CoTTAAdapter(lr=1e-3, weight_decay=0.01, max_grad_norm=0.5,
                           teacher_temperature=0.3)
    assert adapter.lr == 1e-3
    assert adapter.weight_decay == 0.01
    assert adapter.max_grad_norm == 0.5
    assert adapter.teacher_temperature == 0.3


def test_cotta_state_dict():
    adapter = CoTTAAdapter()
    state = adapter.state_dict()
    assert 'source_weights' in state
    assert 'teacher_weights' in state
    assert 'optimizer' in state


def test_palm_initialization():
    adapter = PALMAdapter()
    # DR-tuned official PALM defaults
    assert adapter.lr == 5e-4
    assert adapter.base_lr == 5e-4
    assert adapter.max_grad_norm == 5.0
    # Official PALM params
    assert adapter.temp == 50.0
    assert adapter.layer_selection_threshold is None
    assert adapter.selection_percentile == 0.30
    assert adapter.always_blocks == 1
    assert adapter.min_selected_ratio == 0.05
    assert adapter.sensitivity_alpha == 0.5
    assert adapter.consistency_lambda == 0.10
    assert adapter.entropy_margin == 0.4
    assert adapter.min_confident_fraction == 0.35
    assert adapter.per_class_cap == 4
    assert adapter.min_importance == 1.0
    assert adapter.max_importance == 1.0
    assert adapter.diversity_weight == 5.0
    assert adapter.head_lr_scale == 1.0
    # Collapse detection params
    assert adapter.entropy_collapse_threshold == 0.5
    assert adapter.confidence_spike_threshold == 0.9
    assert adapter.confidence_spike_factor == 1.5
    assert adapter.confidence_ema_alpha == 0.9
    assert adapter.collapse_window == 3


def test_palm_custom_params():
    adapter = PALMAdapter(
        sensitivity_alpha=0.9, lr=1e-3, per_class_cap=3,
        head_lr_scale=2.0, selection_percentile=0.3,
        min_importance=0.2, max_importance=8.0,
        consistency_lambda=1.0, diversity_weight=2.0,
    )
    assert adapter.sensitivity_alpha == 0.9
    assert adapter.lr == 1e-3
    assert adapter.per_class_cap == 3
    assert adapter.head_lr_scale == 2.0
    assert adapter.selection_percentile == 0.3
    assert adapter.min_importance == 0.2
    assert adapter.max_importance == 8.0
    assert adapter.consistency_lambda == 1.0
    assert adapter.diversity_weight == 2.0


def test_palm_state_dict():
    adapter = PALMAdapter()
    state = adapter.state_dict()
    assert 'source_weights' in state
    assert 'setup_done' in state
    assert 'fallback_active' in state
