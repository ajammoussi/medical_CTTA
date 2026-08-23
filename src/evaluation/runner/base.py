"""Shared utilities for CTTA runners.

Contains common methods used by both single-domain and sequential runners:
- Model loading
- Classifier initialization from prototypes
- Adapter creation
- Model evaluation
- Result saving
"""

import torch
import torch.nn.functional as F
import logging
from pathlib import Path
from typing import Dict, Any
from torch.utils.data import DataLoader

from src.config import ExperimentConfig
from src.models.registry import ModelRegistry
from src.models.base import FoundationModel
from src.data.registry import DatasetRegistry
from src.adapters.registry import AdapterRegistry
from src.adapters.base import CTTAAdapter
from src.evaluation.metrics import quadratic_weighted_kappa, per_class_accuracy, overall_accuracy
from src.utils.checkpointing import save_results
from src.utils.logging import setup_logging


logger = logging.getLogger(__name__)


class BaseRunnerMixin:
    """Mixin class providing shared runner functionality.

    Provides common methods for model loading, evaluation, and adapter creation.
    Must be used with a class that has self.config and self.device attributes.
    """

    def _load_model(self) -> FoundationModel:
        """Load foundation model with pretrained weights."""
        model_class = ModelRegistry.get(self.config.model.name)
        model = model_class(
            num_classes=self.config.model.num_classes,
            freeze_layers=self.config.model.freeze_layers,
            checkpoint=self.config.model.checkpoint,
            normalize_features=getattr(self.config.model, 'normalize_features', True),
        )
        model.load_weights()
        model = model.to(self.device)

        if self.config.gradient_checkpointing:
            if hasattr(model.backbone, "set_grad_checkpointing"):
                model.backbone.set_grad_checkpointing(enable=True)
            elif hasattr(model.backbone, "grad_checkpointing"):
                model.backbone.grad_checkpointing = True

        logger.info("Loaded RETFound pretrained weights")
        return model

    def _initialize_classifier_from_prototypes(
        self,
        model: FoundationModel,
        dataset_config,
    ) -> None:
        """Initialize classifier weights with class prototypes from training-set features.

        Runs the training set through the model backbone (no classifier), extracts
        avg-pooled features, computes per-class mean vectors, and sets the
        classifier weight matrix to these prototypes (bias to zero).

        This replaces the random classifier head with meaningful class directions,
        enabling the entropy signal used by CoTTA / PALM.
        """
        dataset_class = DatasetRegistry.get(dataset_config.name)
        dataset = dataset_class(
            data_dir=dataset_config.data_dir,
            image_size=dataset_config.image_size,
            train=True,
            normalize_mean=self.config.model.normalize_mean,
            normalize_std=self.config.model.normalize_std,
        )

        loader = DataLoader(
            dataset,
            batch_size=self.config.ctta_batch_size,
            shuffle=False,
            num_workers=dataset_config.num_workers,
            pin_memory=True,
            persistent_workers=(dataset_config.num_workers > 0),
        )

        model.backbone.eval()
        all_features = []
        all_labels = []

        with torch.no_grad():
            for images, labels in loader:
                images = images.to(self.device)
                if hasattr(model, 'extract_features'):
                    features = model.extract_features(images)
                else:
                    features = model.backbone(images)
                all_features.append(features.cpu())
                all_labels.append(labels)

        features = torch.cat(all_features, dim=0)
        labels = torch.cat(all_labels, dim=0)
        num_classes = model.get_num_classes()

        features = F.normalize(features, dim=1)

        prototypes = []
        for c in range(num_classes):
            mask = labels == c
            if mask.sum() > 0:
                proto = features[mask].mean(dim=0)
            else:
                proto = torch.zeros(features.size(1))
            proto = F.normalize(proto.unsqueeze(0), dim=1).squeeze(0)
            prototypes.append(proto)
        prototypes = torch.stack(prototypes)

        model.classifier.weight.data.copy_(prototypes.to(self.device))
        model.classifier.bias.data.zero_()

        T = self.config.prototype.temperature
        if T != 1.0:
            model.classifier.weight.data.mul_(T)
            logger.info(f"Prototype weights scaled by T={T}")

        logger.info(
            f"Prototype classifier initialized from {len(features)} training samples "
            f"({num_classes} classes, {features.size(1)}-dim features, T={T})"
        )

    def _create_adapter(self) -> CTTAAdapter:
        """Create adapter instance based on config."""
        adapter_class = AdapterRegistry.get(self.config.ctta.method)
        base_kwargs = dict(
            lr=self.config.ctta.lr,
            weight_decay=self.config.ctta.weight_decay,
            image_size=self.config.model.image_size,
            max_grad_norm=self.config.max_grad_norm,
        )
        method_kwargs = self._get_method_kwargs()
        return adapter_class(**base_kwargs, **method_kwargs)

    def _get_method_kwargs(self) -> Dict[str, Any]:
        """Get method-specific kwargs for adapter creation."""
        cfg = self.config.ctta
        method = cfg.method
        if method == "cotta":
            return dict(
                ema_alpha=cfg.ema_alpha,
                restore_prob=cfg.restore_prob,
                num_augmentations=cfg.num_augmentations,
                confidence_threshold=cfg.confidence_threshold,
                teacher_temperature=cfg.teacher_temperature,
                entropy_weight=cfg.entropy_weight,
                diversity_weight=cfg.diversity_weight,
                adapt_layernorm=cfg.adapt_layernorm,
                adapt_last_n_blocks=cfg.adapt_last_n_blocks,
                head_lr_multiplier=cfg.head_lr_multiplier,
                per_class_cap=cfg.per_class_cap,
                kl_label_smoothing=cfg.kl_label_smoothing,
                class_weights=cfg.class_weights,
                confidence_gated_restore=cfg.confidence_gated_restore,
                teacher_ensemble_weight=cfg.teacher_ensemble_weight,
                use_class_specific_thresholds=cfg.use_class_specific_thresholds,
                class_prior_alignment_weight=cfg.class_prior_alignment_weight,
                class_prior_alignment_temperature=cfg.class_prior_alignment_temperature,
                class_forcing_threshold=cfg.class_forcing_threshold,
            )
        elif method == "palm":
            return dict(
                # Official PALM params
                temperature=cfg.temperature,
                layer_selection_threshold=cfg.layer_selection_threshold,
                sensitivity_alpha=cfg.sensitivity_alpha,
                consistency_lambda=cfg.consistency_lambda,
                entropy_margin=cfg.entropy_margin,
                # DR-tuned PALM
                selection_percentile=cfg.selection_percentile,
                always_blocks=cfg.always_blocks,
                min_selected_ratio=cfg.min_selected_ratio,
                min_confident_fraction=cfg.min_confident_fraction,
                per_class_cap=cfg.per_class_cap,
                min_importance=cfg.min_importance,
                max_importance=cfg.max_importance,
                head_lr_scale=cfg.head_lr_scale,
                # Collapse detection (safety)
                entropy_collapse_threshold=cfg.entropy_collapse_threshold,
                confidence_spike_threshold=cfg.confidence_spike_threshold,
                confidence_spike_factor=cfg.confidence_spike_factor,
                confidence_ema_alpha=cfg.confidence_ema_alpha,
                collapse_window=cfg.collapse_window,
                # (silently ignored by adapter)
                ema_alpha=cfg.ema_alpha,
                restore_prob=cfg.restore_prob,
                num_augmentations=cfg.num_augmentations,
                confidence_threshold=cfg.confidence_threshold,
                teacher_temperature=cfg.teacher_temperature,
                entropy_weight=cfg.entropy_weight,
                diversity_weight=cfg.diversity_weight,
                adapt_layernorm=cfg.adapt_layernorm,
                adapt_last_n_blocks=cfg.adapt_last_n_blocks,
                head_lr_multiplier=cfg.head_lr_multiplier,
                layer_selection_percentile=cfg.layer_selection_percentile,
            )
        elif method == "vida":
            return dict(
                vida_rank1=cfg.vida_rank1,
                vida_rank2=cfg.vida_rank2,
                uncertainty_threshold=cfg.uncertainty_threshold,
                num_augmentations=cfg.vida_num_augmentations,
                uncertainty_scale=cfg.uncertainty_scale,
                alpha_teacher=cfg.alpha_teacher,
                alpha_vida=cfg.alpha_vida,
                vida_lr=cfg.vida_lr,
                model_lr=cfg.vida_model_lr,
                restore_prob=cfg.restore_prob,
                ce_loss_weight=cfg.ce_loss_weight,
                pseudo_label_threshold=cfg.pseudo_label_threshold,
            )
        elif method == "ecotta":
            return dict(
                ecotta_num_partitions=cfg.ecotta_num_partitions,
                ecotta_partition_sizes=cfg.ecotta_partition_sizes,
                ecotta_meta_hidden_scale=cfg.ecotta_meta_hidden_scale,
                ecotta_reg_lambda=cfg.ecotta_reg_lambda,
                ecotta_entropy_margin=cfg.entropy_margin,
                ecotta_warmup_epochs=cfg.ecotta_warmup_epochs,
                ecotta_warmup_lr=cfg.ecotta_warmup_lr,
                ecotta_tta_lr=cfg.ecotta_tta_lr,
                ecotta_min_confident_fraction=cfg.ecotta_min_confident_fraction,
                ecotta_per_class_cap=cfg.ecotta_per_class_cap,
                ecotta_grad_checkpointing=self.config.gradient_checkpointing,
            )
        elif method == "lcotta":
            return dict(
                lcotta_subspace_dim=cfg.lcotta_subspace_dim,
                lcotta_queue_length=cfg.lcotta_queue_length,
                lcotta_sample_interval=cfg.lcotta_sample_interval,
                lcotta_entropy_margin=cfg.entropy_margin,
                lcotta_momentum=cfg.lcotta_momentum,
                lcotta_cosine_filter=cfg.lcotta_cosine_filter,
                lcotta_cosine_threshold=cfg.lcotta_cosine_threshold,
                lcotta_prob_ema=cfg.lcotta_prob_ema,
                lcotta_min_confident_fraction=cfg.lcotta_min_confident_fraction,
                lcotta_per_class_cap=cfg.lcotta_per_class_cap,
                lcotta_adapt_head=cfg.lcotta_adapt_head,
                lcotta_head_lr_multiplier=cfg.lcotta_head_lr_multiplier,
                lcotta_optimizer=cfg.lcotta_optimizer,
                lcotta_prior_alignment_weight=cfg.lcotta_prior_alignment_weight,
                lcotta_head_anchor_weight=cfg.lcotta_head_anchor_weight,
            )
        return {}

    def _build_train_loader(self, dataset_config):
        """Build a train-split DataLoader for the given dataset config.

        Used by methods that warm up on the source train split before adaptation
        (e.g. EcoTTA's source CE warmup).
        """
        dataset_class = DatasetRegistry.get(dataset_config.name)
        dataset = dataset_class(
            data_dir=dataset_config.data_dir,
            image_size=dataset_config.image_size,
            train=True,
            normalize_mean=self.config.model.normalize_mean,
            normalize_std=self.config.model.normalize_std,
        )
        return DataLoader(
            dataset,
            batch_size=self.config.ctta_batch_size,
            shuffle=True,
            num_workers=dataset_config.num_workers,
            pin_memory=True,
            persistent_workers=(dataset_config.num_workers > 0),
        )

    def _evaluate_model(
        self,
        model: FoundationModel,
        dataset_config,
    ) -> Dict[str, float]:
        """Evaluate model on a dataset (test mode, no adaptation)."""
        dataset_class = DatasetRegistry.get(dataset_config.name)
        dataset = dataset_class(
            data_dir=dataset_config.data_dir,
            image_size=dataset_config.image_size,
            train=False,
            normalize_mean=self.config.model.normalize_mean,
            normalize_std=self.config.model.normalize_std,
        )

        loader = DataLoader(
            dataset,
            batch_size=self.config.ctta_batch_size,
            shuffle=False,
            num_workers=dataset_config.num_workers,
            pin_memory=True,
            persistent_workers=(dataset_config.num_workers > 0),
        )

        model.backbone.eval()
        model.classifier.eval()

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels in loader:
                images = images.to(self.device)
                logits = model(images)
                preds = torch.argmax(logits, dim=-1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())

        qwk = quadratic_weighted_kappa(all_labels, all_preds)
        per_class = per_class_accuracy(all_labels, all_preds, model.get_num_classes())
        acc = overall_accuracy(all_labels, all_preds)

        return {
            "qwk": qwk,
            "overall_accuracy": acc,
            "per_class_accuracy": per_class,
            "num_samples": len(all_labels),
        }

    def _save_results(self, results: Dict[str, Any]) -> None:
        """Save results to JSON file."""
        save_results(results, str(self.output_dir / "ctta_results.json"))
        logger.info(f"Results saved to {self.output_dir / 'ctta_results.json'}")

    def _adapt_log_line(self, batch_idx: int, metrics: Dict[str, Any]) -> str:
        """Build the per-step adapt log line from adapter metrics.

        CoTTA reports KL / confident-sample / teacher confidence; PALM reports
        entropy signals and selected-parameter coverage instead. Unknown methods
        fall back to the loss only so no stale -1 values are printed.
        """
        method = getattr(self.config.ctta, "method", "").lower()
        loss = metrics.get("loss", float("nan"))
        base = f"  Adapt step {batch_idx}: loss={loss:.4f}"

        if method == "palm":
            ent = metrics.get("student_entropy", -1.0)
            n_sel = metrics.get("n_selected_params", -1)
            n_sel_mass = metrics.get("n_total_params", -1)
            n_cand = metrics.get("n_candidate_params", -1)
            coverage = (f"{100.0 * n_sel_mass / n_cand:.1f}%"
                        if n_cand and n_cand > 0 and n_sel_mass >= 0 else "n/a")
            return (
                f"{base}, entropy={ent:.4f}, "
                f"conf_samples={metrics.get('num_confident_samples', -1)}, "
                f"selected_mass={n_sel_mass:,}/{n_cand:,} ({coverage})"
            )

        if method == "vida":
            unc = metrics.get("uncertainty", -1.0)
            lam_l = metrics.get("lambda_low", -1.0)
            lam_h = metrics.get("lambda_high", -1.0)
            kl = metrics.get("kl_loss", -1.0)
            ce = metrics.get("ce_loss", -1.0)
            conf = metrics.get("confidence_frac", -1.0)
            return (
                f"{base}, kl={kl:.4f}, ce={ce:.4f}, "
                f"conf={conf:.2f}, unc={unc:.4f}, "
                f"lam_low={lam_l:.3f}, lam_high={lam_h:.3f}, "
                f"restored={metrics.get('restoration_count', 0)}"
            )

        if method == "lcotta":
            ent = metrics.get("student_entropy", -1.0)
            n_conf = metrics.get("num_confident_samples", -1)
            q_len = metrics.get("queue_len", -1)
            active = metrics.get("subspace_active", False)
            ratio = metrics.get("proj_norm_ratio", -1.0)
            return (
                f"{base}, entropy={ent:.4f}, conf_samples={n_conf}, "
                f"queue={q_len}, subspace={'on' if active else 'off'}, "
                f"proj_ratio={ratio:.3f}"
            )

        if method == "ecotta":
            ent = metrics.get("student_entropy", -1.0)
            reg = metrics.get("reg_loss", -1.0)
            ent_loss = metrics.get("entropy_loss", -1.0)
            return (
                f"{base}, entropy={ent:.4f}, ent_loss={ent_loss:.4f}, "
                f"reg={reg:.4f}, conf_samples={metrics.get('num_confident_samples', -1)}"
            )

        kl = metrics.get("kl_loss", -1.0)
        n_conf = metrics.get("num_confident_samples", -1)
        teacher_mmp = metrics.get("teacher_mean_max_prob", -1.0)
        return (
            f"{base}, kl={kl:.4f}, conf_samples={n_conf}, "
            f"teacher_max_prob={teacher_mmp:.4f}"
        )
