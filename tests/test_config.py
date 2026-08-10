import pytest
from src.config import load_config, ExperimentConfig


def test_load_cotta_aptos_config():
    config = load_config("configs/method/cotta/aptos.yaml")
    assert isinstance(config, ExperimentConfig)
    assert config.model.name == "retfound"
    assert config.target_dataset.name == "aptos2019"
    assert config.ctta.method == "cotta"
    assert config.seed == 42


def test_config_defaults():
    config = ExperimentConfig()
    assert config.model.num_classes == 5
    assert config.model.image_size == 224
    assert config.model.weight_decay == 0.05
    assert config.ctta.ema_alpha == 0.999
    assert config.ctta.restore_prob == 0.01
    assert config.ctta.num_augmentations == 6
    assert config.ctta.confidence_threshold == 0.65
    assert config.ctta.adapt_layernorm is True
    assert config.ctta.adapt_last_n_blocks == 6
    assert config.ctta.head_lr_multiplier == 5.0
    assert config.prototype.temperature == 10.0
    assert config.use_amp is True
    assert config.gradient_checkpointing is True
    assert config.max_grad_norm == 1.0


def test_palm_config_defaults():
    config = load_config("configs/method/palm/aptos.yaml")
    assert config.ctta.method == "palm"
    # DR-tuned official PALM
    assert config.ctta.lr == 5e-4
    assert config.ctta.temperature == 50.0
    assert config.ctta.layer_selection_threshold is None
    assert config.ctta.selection_percentile == 0.30
    assert config.ctta.always_blocks == 1
    assert config.ctta.min_selected_ratio == 0.05
    assert config.ctta.consistency_lambda == 0.10
    assert config.ctta.entropy_margin == 0.4
    assert config.ctta.min_confident_fraction == 0.35
    assert config.ctta.per_class_cap == 4
    assert config.ctta.min_importance == 1.0
    assert config.ctta.max_importance == 1.0
    assert config.ctta.diversity_weight == 5.0
    assert config.ctta.head_lr_scale == 1.0
    assert config.ctta.adapt_every_batch is True
    assert config.prototype.temperature == 100.0 
    # Collapse detection params
    assert config.ctta.entropy_collapse_threshold == 0.5
    assert config.ctta.confidence_spike_threshold == 0.9
    assert config.ctta.confidence_spike_factor == 1.5
    assert config.ctta.confidence_ema_alpha == 0.9
    assert config.ctta.collapse_window == 3


def test_palm_config_idrid():
    config = load_config("configs/method/palm/idrid.yaml")
    assert config.ctta.method == "palm"
    assert config.ctta.lr == 5e-4
    assert config.ctta.selection_percentile == 0.30
    assert config.ctta.per_class_cap == 4
    assert config.ctta.min_importance == 1.0
    assert config.ctta.max_importance == 1.0
    assert config.ctta.diversity_weight == 5.0
    assert config.ctta.sensitivity_alpha == 0.5
    assert config.prototype.temperature == 100.0


def test_palm_config_aptos_visionfm():
    config = load_config("configs/method/palm/aptos_visionfm.yaml")
    assert config.ctta.method == "palm"
    assert config.ctta.lr == 1e-4
    assert config.ctta.selection_percentile == 0.30
    assert config.ctta.per_class_cap == 4
    assert config.ctta.min_importance == 1.0
    assert config.ctta.max_importance == 1.0
    assert config.ctta.diversity_weight == 5.0
    assert config.ctta.sensitivity_alpha == 0.5
    assert config.prototype.temperature == 100.0


def test_palm_config_idrid_visionfm():
    config = load_config("configs/method/palm/idrid_visionfm.yaml")
    assert config.ctta.method == "palm"
    assert config.ctta.lr == 1e-4
    assert config.ctta.selection_percentile == 0.30
    assert config.ctta.per_class_cap == 4
    assert config.ctta.min_importance == 1.0
    assert config.ctta.max_importance == 1.0
    assert config.ctta.diversity_weight == 5.0
    assert config.ctta.sensitivity_alpha == 0.5
    assert config.prototype.temperature == 100.0


def test_config_validation():
    with pytest.raises(Exception):
        ExperimentConfig(model={"num_classes": "invalid"})
