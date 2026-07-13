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
    assert adapter.num_augmentations == 8
    assert adapter.confidence_threshold == 0.5
    assert adapter.lr == 1e-4
    assert adapter.weight_decay == 0.0


def test_cotta_custom_lr():
    adapter = CoTTAAdapter(lr=1e-3, weight_decay=0.01)
    assert adapter.lr == 1e-3
    assert adapter.weight_decay == 0.01


def test_cotta_state_dict():
    adapter = CoTTAAdapter()
    state = adapter.state_dict()
    assert 'source_weights' in state
    assert 'teacher_weights' in state
    assert 'optimizer' in state


def test_palm_initialization():
    adapter = PALMAdapter()
    assert adapter.lr == 1e-4
    assert adapter.temperature == 10.0
    assert adapter.sensitivity_alpha == 0.5
    assert adapter.consistency_lambda == 0.01
    assert adapter.selection_percentile == 0.3
    assert adapter.layer_selection_threshold is None
    assert adapter.entropy_threshold_factor == 0.8


def test_palm_custom_params():
    adapter = PALMAdapter(
        temperature=20.0, sensitivity_alpha=0.9,
        consistency_lambda=0.05, lr=5e-5,
    )
    assert adapter.temperature == 20.0
    assert adapter.sensitivity_alpha == 0.9
    assert adapter.consistency_lambda == 0.05
    assert adapter.lr == 5e-5


def test_palm_state_dict():
    adapter = PALMAdapter()
    state = adapter.state_dict()
    assert 'running_sensitivity' in state
    assert 'optimizer' in state
