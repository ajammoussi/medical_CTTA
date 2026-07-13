"""Configuration system for CTTA experiments.

Supports both single-target and sequential multi-domain CTTA protocols.
Config files support:
  - ``extends: <path>`` to inherit from a base config (deep-merged)
  - ``!include <path>`` YAML tag to inline another config file
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
import yaml
from pathlib import Path


# ---------------------------------------------------------------------------
# Custom YAML loader with !include tag
# ---------------------------------------------------------------------------

class _IncludeLoader(yaml.SafeLoader):
    """YAML loader that resolves ``!include <path>`` relative to the current file."""

    def __init__(self, stream):
        self._root = Path(stream.name).parent.resolve()
        super().__init__(stream)


def _include_constructor(loader: _IncludeLoader, node: yaml.Node) -> Any:
    """Resolve ``!include <path>`` — loads and returns the referenced YAML file."""
    path = loader.construct_scalar(node)
    full_path = (loader._root / path).resolve()
    with open(full_path, "r", encoding="utf-8") as f:
        return yaml.load(f, _IncludeLoader)


_IncludeLoader.add_constructor("!include", _include_constructor)


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge *override* into *base*, returning a new dict.

    Scalar values in *override* replace those in *base*.
    Dict values are merged recursively.
    List and other values are replaced entirely.
    """
    merged = base.copy()
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


# ---------------------------------------------------------------------------
# Pydantic config models
# ---------------------------------------------------------------------------


class ModelConfig(BaseModel):
    """Configuration for foundation model."""
    name: str = "retfound"
    checkpoint: str = "YukunZhou/RETFound_mae_natureCFP"
    num_classes: int = 5
    freeze_layers: int = 20
    lr: float = 1e-4
    weight_decay: float = 0.05
    layer_decay: float = 0.65
    drop_path: float = 0.2
    image_size: int = 224


class DatasetConfig(BaseModel):
    """Configuration for a single dataset."""
    name: str = "idrid"
    data_dir: str = "./data/IDRiD/"
    image_size: int = 224
    num_workers: int = 4


class CTTAConfig(BaseModel):
    """Configuration for CTTA adapter.

    CoTTA-specific fields: ema_alpha, restore_prob, confidence_threshold
    PALM-specific fields: temperature, selection_*, sensitivity_alpha,
                          consistency_lambda, entropy_threshold_factor
    """
    method: str = "cotta"
    ema_alpha: float = 0.999
    restore_prob: float = 0.01
    num_augmentations: int = 8
    confidence_threshold: float = 0.3
    entropy_threshold: float = 1.0
    lr: float = 1e-4
    weight_decay: float = 0.0
    batch_size: int = 16

    temperature: float = 10.0
    layer_selection_threshold: Optional[float] = None
    selection_percentile: float = 0.3
    sensitivity_alpha: float = 0.5
    consistency_lambda: float = 0.01
    entropy_threshold_factor: float = 0.8


class ShiftDetectionConfig(BaseModel):
    """Configuration for shift detection signals."""
    entropy_threshold: float = 0.3
    drift_threshold: float = 0.05
    distribution_threshold: float = 0.05
    ema_momentum: float = 0.9
    window_size: int = 50


class DomainTransitionConfig(BaseModel):
    """Configuration for a single domain in a sequential CTTA protocol."""
    dataset: DatasetConfig
    num_batches: Optional[int] = None
    description: str = ""


class SequentialCTTAConfig(BaseModel):
    """Configuration for sequential multi-domain CTTA.

    Protocol: source_domain -> target_domains[0] -> target_domains[1] -> ...
    No weight reset between domains -- model carries state forward.
    """
    enabled: bool = False
    source_domain: DomainTransitionConfig = Field(
        default_factory=lambda: DomainTransitionConfig(
            dataset=DatasetConfig(name="idrid", data_dir="./data/IDRiD/"),
            description="Source domain: single camera, single clinic (IDRiD)"
        )
    )
    target_domains: List[DomainTransitionConfig] = Field(
        default_factory=lambda: [
            DomainTransitionConfig(
                dataset=DatasetConfig(name="aptos2019", data_dir="./data/APTOS2019/"),
                description="Target domain 1: heterogeneous multi-site (APTOS 2019)"
            ),
            DomainTransitionConfig(
                dataset=DatasetConfig(name="idrid", data_dir="./data/IDRiD/"),
                description="Target domain 2: return to source domain (test forgetting)"
            ),
        ]
    )
    reset_weights_between_domains: bool = False


class LRSchedulerConfig(BaseModel):
    """Configuration for learning rate scheduler."""
    name: str = "cosine"
    warmup_epochs: int = 5
    min_lr: float = 1e-6


class ExperimentConfig(BaseModel):
    """Top-level experiment configuration.

    For sequential CTTA, use ``sequential_ctta.enabled=True``.
    For single-target CTTA, use ``source_dataset`` + ``target_dataset``.
    """
    model: ModelConfig = ModelConfig()
    source_dataset: DatasetConfig = DatasetConfig(name="idrid")
    target_dataset: DatasetConfig = DatasetConfig(name="aptos2019")
    ctta: CTTAConfig = CTTAConfig()
    shift_detection: ShiftDetectionConfig = ShiftDetectionConfig()
    sequential_ctta: SequentialCTTAConfig = SequentialCTTAConfig()
    seed: int = 42
    device: str = "cuda"
    output_dir: str = "./outputs/"
    shared_source_dir: str = "./outputs/source/"
    num_epochs_source: int = 50
    ctta_batch_size: int = 16
    gradient_accumulation_steps: int = 1
    use_amp: bool = True
    gradient_checkpointing: bool = True
    max_grad_norm: float = 1.0
    lr_scheduler: LRSchedulerConfig = LRSchedulerConfig()


# ---------------------------------------------------------------------------
# Public loader
# ---------------------------------------------------------------------------


def load_config(config_path: str) -> ExperimentConfig:
    """Load configuration from a YAML file.

    Supports:
    - ``extends: <path>`` — inherit from a base config (paths relative to the
      child config's directory).
    - ``!include <path>`` — inline another YAML file (paths relative to the
      current config's directory).

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        An ``ExperimentConfig`` instance.
    """
    path = Path(config_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(path, "r", encoding="utf-8") as f:
        config_dict = yaml.load(f, _IncludeLoader)

    # Handle extends: load base config and deep-merge
    if "extends" in config_dict:
        base_rel = config_dict.pop("extends")
        base_path = (path.parent / base_rel).resolve()
        if not base_path.exists():
            raise FileNotFoundError(
                f"Base config '{base_rel}' not found (resolved: {base_path})"
            )
        with open(base_path, "r", encoding="utf-8") as f:
            base_dict = yaml.load(f, _IncludeLoader)
        config_dict = _deep_merge(base_dict, config_dict)

    model_config = ModelConfig(**config_dict.get("model", {}))
    source_dataset_config = DatasetConfig(**config_dict.get("source_dataset", {}))
    target_dataset_config = DatasetConfig(**config_dict.get("target_dataset", {}))
    ctta_config = CTTAConfig(**config_dict.get("ctta", {}))
    shift_detection_config = ShiftDetectionConfig(**config_dict.get("shift_detection", {}))

    sequential_config_dict = config_dict.get("sequential_ctta", {})
    if sequential_config_dict.get("enabled", False):
        source_domain_dict = sequential_config_dict.get("source_domain", {})
        target_domains_list = sequential_config_dict.get("target_domains", [])

        source_domain = DomainTransitionConfig(
            dataset=DatasetConfig(**source_domain_dict.get("dataset", {})),
            num_batches=source_domain_dict.get("num_batches"),
            description=source_domain_dict.get("description", "")
        )

        target_domains = []
        for td in target_domains_list:
            target_domains.append(DomainTransitionConfig(
                dataset=DatasetConfig(**td.get("dataset", {})),
                num_batches=td.get("num_batches"),
                description=td.get("description", "")
            ))

        sequential_config = SequentialCTTAConfig(
            enabled=True,
            source_domain=source_domain,
            target_domains=target_domains,
            reset_weights_between_domains=sequential_config_dict.get(
                "reset_weights_between_domains", False
            )
        )
    else:
        sequential_config = SequentialCTTAConfig()

    top_level_keys = [
        "seed", "device", "output_dir", "shared_source_dir",
        "num_epochs_source", "ctta_batch_size",
        "gradient_accumulation_steps", "use_amp", "gradient_checkpointing",
        "max_grad_norm",
    ]
    top_level = {k: config_dict[k] for k in top_level_keys if k in config_dict}

    lr_scheduler_dict = config_dict.get("lr_scheduler", {})
    if lr_scheduler_dict:
        top_level["lr_scheduler"] = LRSchedulerConfig(**lr_scheduler_dict)

    return ExperimentConfig(
        model=model_config,
        source_dataset=source_dataset_config,
        target_dataset=target_dataset_config,
        ctta=ctta_config,
        shift_detection=shift_detection_config,
        sequential_ctta=sequential_config,
        **top_level
    )
