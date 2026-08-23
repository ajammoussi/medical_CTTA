"""Configuration system for CTTA experiments.

Load RETFound pretrained weights,
evaluate baseline, run CTTA adaptation, evaluate post-adaptation.

Config files support:
  - ``extends: <path>`` to inherit from a base config (deep-merged)
  - ``!include <path>`` YAML tag to inline another config file
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel
import yaml
from pathlib import Path


# ---------------------------------------------------------------------------
# Custom YAML loader with !include tag
# ---------------------------------------------------------------------------

class _IncludeLoader(yaml.SafeLoader):
    def __init__(self, stream):
        self._root = Path(stream.name).parent.resolve()
        super().__init__(stream)


def _include_constructor(loader: _IncludeLoader, node: yaml.Node) -> Any:
    path = loader.construct_scalar(node)
    full_path = (loader._root / path).resolve()
    with open(full_path, "r", encoding="utf-8") as f:
        return yaml.load(f, _IncludeLoader)


_IncludeLoader.add_constructor("!include", _include_constructor)


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
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
    name: str = "retfound"
    checkpoint: str = "YukunZhou/RETFound_dinov2_meh"
    num_classes: int = 5
    freeze_layers: int = 20
    lr: float = 1e-4
    weight_decay: float = 0.05
    layer_decay: float = 0.65
    drop_path: float = 0.2
    image_size: int = 224
    normalize_features: bool = True
    normalize_mean: List[float] = [0.485, 0.456, 0.406]
    normalize_std: List[float] = [0.229, 0.224, 0.225]


class DatasetConfig(BaseModel):
    name: str = "idrid"
    data_dir: str = "./data/IDRiD/"
    image_size: int = 224
    num_workers: int = 4


class CTTAConfig(BaseModel):
    method: str = "cotta"
    ema_alpha: float = 0.999
    restore_prob: float = 0.01
    num_augmentations: int = 6
    confidence_threshold: float = 0.65
    entropy_threshold: float = 1.0
    lr: float = 5e-4
    weight_decay: float = 0.0
    batch_size: int = 16
    entropy_weight: float = 0.1
    diversity_weight: float = 0.05
    adapt_layernorm: bool = True
    adapt_last_n_blocks: int = 6

    adapt_every_batch: bool = False
    teacher_temperature: float = 0.5
    temperature: float = 10.0
    layer_selection_threshold: Optional[float] = None
    selection_percentile: float = 0.4
    layer_selection_percentile: float = 0.3
    sensitivity_alpha: float = 0.5
    consistency_lambda: float = 0.05
    entropy_threshold_factor: float = 0.8  # legacy — kept for CoTTA compat
    entropy_margin: float = 0.4             # PALM H₀ factor (H0 = entropy_margin * log C)
    # DR-tuned PALM
    always_blocks: int = 1                  # trailing backbone blocks always selected
    min_selected_ratio: float = 0.05        # min fraction of trainable params selected
    min_confident_fraction: float = 0.25    # guaranteed confident-sample quota per batch
    min_importance: float = 0.1             # adaptive-LR lower clamp
    max_importance: float = 10.0            # adaptive-LR upper clamp
    head_lr_scale: float = 1.0              # classifier head LR multiplier
    head_lr_multiplier: float = 5.0     
    per_class_cap: int = 0                        # 0 = disabled; >0 = max confident samples per class per batch
    kl_label_smoothing: float = 0.0               # 0.0 = disabled; 0.1 = light smoothing on KL target
    class_weights: Optional[List[float]] = None   # per-class reliability weights for KL loss (e.g., baseline accuracy)
    confidence_gated_restore: bool = False         # skip restore when teacher has 0 confident samples
    teacher_ensemble_weight: float = 0.0           # blend pretrained weights into teacher for pseudo-labels (0=off)
    use_class_specific_thresholds: bool = False    # dynamic per-class thresholds from PLF paper (arxiv 2406.02609)
    class_prior_alignment_weight: float = 0.0      # CPA loss weight from PLF paper (0=off, 0.1=light, 1.0=strong)
    class_prior_alignment_temperature: float = 0.1 # temperature for CPA soft label smoothing
    class_forcing_threshold: int = 0               # N=force class after N batches with 0 pseudo-labels
    # Collapse detection params (PALM)
    entropy_collapse_threshold: float = 0.5        # entropy drop threshold to trigger collapse detection
    confidence_spike_threshold: float = 0.9        # absolute near-certainty floor for spike detection
    confidence_spike_factor: float = 1.5           # relative spike ratio (current/EMA) — scale-free
    confidence_ema_alpha: float = 0.9              # EMA speed for confidence baseline
    collapse_window: int = 3                        # number of batches to check for entropy drop
    # ViDA-specific params
    vida_rank1: int = 1                             # low-rank bottleneck dimension (domain-shared)
    vida_rank2: int = 128                           # high-rank bottleneck dimension (domain-specific)
    uncertainty_threshold: float = 0.2              # HKA uncertainty threshold (Theta)
    vida_num_augmentations: int = 10                # augmented passes for uncertainty estimation
    uncertainty_scale: float = 0.1                  # uncertainty scaling factor
    alpha_teacher: float = 0.99                     # EMA rate for original model params
    alpha_vida: float = 0.8                         # EMA rate for ViDA adapter params
    vida_lr: float = 5e-4                           # learning rate for ViDA adapter params (Adam)
    vida_model_lr: float = 5e-7                     # learning rate for original model params (via EMA)
    ce_loss_weight: float = 1.0                     # weight for pseudo-label CE loss
    pseudo_label_threshold: float = 0.5             # confidence threshold for pseudo-labels
    # EcoTTA-specific params (arXiv 2303.01904)
    ecotta_num_partitions: int = 4                  # K partitions of the frozen encoder
    ecotta_partition_sizes: Optional[List[int]] = None  # explicit per-partition block counts (auto if None)
    ecotta_meta_hidden_scale: float = 1.0           # MLP hidden dim = embed_dim * scale
    ecotta_reg_lambda: float = 0.25                 # self-distilled L1 regularization weight (paper 0.5 / README 0.25)
    ecotta_warmup_epochs: int = 5                   # source warmup epochs (CE)
    ecotta_warmup_lr: float = 5e-2                  # SGD lr during warmup (paper: 5e-2)
    ecotta_tta_lr: float = 5e-3                     # SGD lr during TTA (paper: 5e-3)
    ecotta_min_confident_fraction: float = 0.25     # DR safety net: guaranteed entropy-quota per batch
    ecotta_per_class_cap: Optional[int] = None      # max confident samples per predicted class (None = uncapped)
    # LCoTTA-specific params (NeurIPS 2025, subspace-projected entropy minimization)
    lcotta_subspace_dim: int = 10                   # r: principal subspace rank (paper 25 R50 / 50 ViT on ImageNet-C)
    lcotta_queue_length: int = 30                   # k: gradient queue length (paper 100; DR test sets are smaller)
    lcotta_sample_interval: int = 2                 # sample a gradient into the queue every N batches (paper 50/100)
    lcotta_entropy_margin: float = 0.4              # e_margin factor: H0 = margin * log(C) (official 0.4*ln(1000))
    lcotta_momentum: float = 0.9                    # SGD momentum (official BETA: 0.9)
    lcotta_cosine_filter: bool = True               # EATA-style redundancy filter vs running mean prediction
    lcotta_cosine_threshold: float = 0.05           # |cos| threshold for the redundancy filter (official 0.05)
    lcotta_prob_ema: float = 0.9                    # EMA rate of the running mean prediction (official 0.9)
    lcotta_min_confident_fraction: float = 0.25     # DR safety net: guaranteed entropy-quota per batch
    lcotta_per_class_cap: Optional[int] = None      # max confident samples per predicted class (None = uncapped)
    lcotta_adapt_head: bool = False                 # also adapt the prototype classifier head (DR-specific)
    lcotta_head_lr_multiplier: float = 10.0         # head param-group lr = lcotta lr * this
    lcotta_optimizer: str = "sgd"                   # "sgd" (official) | "adam" (DR: tiny entropy grads need adaptive steps)
    lcotta_prior_alignment_weight: float = 0.0      # SAR-style APU: keeps batch mean prediction high-entropy (stops class-2 drain)
    lcotta_head_anchor_weight: float = 0.0          # L2 trust region on head vs source prototypes (bounds per-class erosion)


class PrototypeConfig(BaseModel):
    temperature: float = 10.0


class DatasetPath(BaseModel):
    """A dataset with its name, data directory path, and optional per-domain CTTA params."""
    name: str
    data_dir: str
    image_size: int = 224
    num_workers: int = 4
    ctta: Optional[CTTAConfig] = None


class SequentialConfig(BaseModel):
    """Configuration for sequential multi-domain CTTA.

    Protocol:
      1. Evaluate baseline on source domain
      2. For each target domain in order:
         a. Adapt to target domain (adaptation mode) using that domain's CTTA params
         b. Evaluate on ALL previous domains (TEST mode, no adaptation)
      3. Final evaluation on all domains

    This measures catastrophic forgetting: after adapting to new domains,
    how much performance degrades on previously seen domains.

    Each target dataset can have its own CTTA params (ctta field).
    If ctta is None, the top-level ctta config is used as fallback.
    """
    enabled: bool = False
    source_dataset: DatasetPath = DatasetPath(name="idrid", data_dir="./data/IDRiD/")
    target_datasets: List[DatasetPath] = [
        DatasetPath(name="aptos2019", data_dir="./data/APTOS2019/")
    ]
    reset_weights_before_each_target: bool = False


class ExperimentConfig(BaseModel):
    model: ModelConfig = ModelConfig()
    target_dataset: DatasetConfig = DatasetConfig()
    ctta: CTTAConfig = CTTAConfig()
    prototype: PrototypeConfig = PrototypeConfig()
    sequential: SequentialConfig = SequentialConfig()
    seed: int = 42
    device: str = "cuda"
    output_dir: str = "./outputs/"
    ctta_batch_size: int = 16
    use_amp: bool = True
    gradient_checkpointing: bool = True
    max_grad_norm: float = 1.0


# ---------------------------------------------------------------------------
# Public loader
# ---------------------------------------------------------------------------


def load_config(config_path: str) -> ExperimentConfig:
    path = Path(config_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(path, "r", encoding="utf-8") as f:
        config_dict = yaml.load(f, _IncludeLoader)

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
    target_dataset_config = DatasetConfig(**config_dict.get("target_dataset", {}))
    ctta_config = CTTAConfig(**config_dict.get("ctta", {}))
    prototype_config = PrototypeConfig(**config_dict.get("prototype", {}))
    sequential_config = SequentialConfig(**config_dict.get("sequential", {}))

    top_level_keys = [
        "seed", "device", "output_dir",
        "ctta_batch_size", "use_amp", "gradient_checkpointing", "max_grad_norm",
    ]
    top_level = {k: config_dict[k] for k in top_level_keys if k in config_dict}

    return ExperimentConfig(
        model=model_config,
        target_dataset=target_dataset_config,
        ctta=ctta_config,
        prototype=prototype_config,
        sequential=sequential_config,
        **top_level
    )
