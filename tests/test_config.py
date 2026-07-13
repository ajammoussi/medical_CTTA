import pytest
from src.config import load_config, ExperimentConfig


def test_load_default_config():
    config = load_config("configs/method/cotta.yaml")
    assert isinstance(config, ExperimentConfig)
    assert config.model.name == "retfound"
    assert config.source_dataset.name == "idrid"
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
    assert config.ctta.num_augmentations == 8
    assert config.ctta.confidence_threshold == 0.3
    assert config.use_amp is True
    assert config.gradient_checkpointing is True
    assert config.gradient_accumulation_steps == 1
    assert config.max_grad_norm == 1.0
    assert config.lr_scheduler.name == "cosine"
    assert config.lr_scheduler.warmup_epochs == 5


def test_palm_config_defaults():
    config = load_config("configs/method/palm.yaml")
    assert config.ctta.method == "palm"
    assert config.ctta.temperature == 100.0
    assert config.ctta.layer_selection_threshold is None
    assert config.ctta.selection_percentile == 0.3
    assert config.ctta.sensitivity_alpha == 0.9
    assert config.ctta.consistency_lambda == 0.01
    assert config.ctta.entropy_threshold_factor == 0.4
    assert config.ctta.num_augmentations == 4
    assert config.ctta.lr == 1e-4


def test_config_validation():
    with pytest.raises(Exception):
        ExperimentConfig(model={"num_classes": "invalid"})
