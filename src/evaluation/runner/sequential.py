"""Sequential Multi-Domain CTTA Runner.

Protocol:
  1. Load RETFound pretrained weights once
  2. Evaluate baseline on source domain
  3. For each target domain in order:
     a. Adapt to target domain (adaptation mode)
     b. Evaluate on ALL previous domains (TEST mode, no adaptation)
  4. Final evaluation on all domains

This measures catastrophic forgetting: after adapting to new domains,
how much performance degrades on previously seen domains.
"""

import torch
import logging
from pathlib import Path
from typing import Dict, Any
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import ExperimentConfig
from src.models.base import FoundationModel
from src.data.registry import DatasetRegistry
from src.adapters.base import CTTAAdapter
from src.utils.seed import set_seed
from src.utils.logging import setup_logging

from src.evaluation.runner.base import BaseRunnerMixin
from src.evaluation.runner.single import DomainResult


logger = logging.getLogger(__name__)


class SequentialCTTARunner(BaseRunnerMixin):
    """Runner for sequential multi-domain CTTA experiments.

    Protocol:
      1. Load RETFound pretrained weights once
      2. Evaluate baseline on source domain
      3. For each target domain in order:
         a. Adapt to target domain (adaptation mode)
         b. Evaluate on ALL previous domains (TEST mode, no adaptation)
      4. Final evaluation on all domains

    This measures catastrophic forgetting: after adapting to new domains,
    how much performance degrades on previously seen domains.

    The model weights carry forward through the sequence (no reset between domains
    unless reset_weights_before_each_target=True), showing cumulative adaptation
    and forgetting effects.
    """

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.adapt_dir = self.output_dir / "adapt"
        self.plots_dir = self.output_dir / "plots"
        self.adapt_dir.mkdir(exist_ok=True)
        self.plots_dir.mkdir(exist_ok=True)
        setup_logging(str(self.output_dir / "run.log"))
        if self.device.type == "cuda":
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True

    def run(self) -> Dict[str, Any]:
        set_seed(self.config.seed)
        if self.device.type == "cuda":
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True
        return self._run_sequential()

    def _run_sequential(self) -> Dict[str, Any]:
        seq_cfg = self.config.sequential
        logger.info(f"Starting sequential CTTA: {seq_cfg.source_dataset.name} -> "
                    f"{' -> '.join(t.name for t in seq_cfg.target_datasets)}")

        model = self._load_model()
        self._initialize_classifier_from_prototypes(model, seq_cfg.source_dataset)

        # Store pretrained weights BEFORE any adaptation (for teacher ensemble)
        pretrained_snapshot = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        # Read method from per-domain config (not top-level default)
        source_ctta = seq_cfg.source_dataset.ctta if seq_cfg.source_dataset.ctta is not None else self.config.ctta
        all_results = {
            "method": source_ctta.method,
            "protocol": "sequential",
            "source_domain": seq_cfg.source_dataset.name,
            "target_domains": [t.name for t in seq_cfg.target_datasets],
            "domain_results": {},
            "baseline_metrics": {},
        }

        logger.info(f"Evaluating baseline on source domain: {seq_cfg.source_dataset.name}")
        baseline_metrics = self._evaluate_model(model, seq_cfg.source_dataset)
        all_results["baseline_metrics"][seq_cfg.source_dataset.name] = baseline_metrics
        logger.info(f"Baseline QWK on {seq_cfg.source_dataset.name}: {baseline_metrics['qwk']:.4f}")

        seen_domains = [seq_cfg.source_dataset]

        logger.info(f"\n{'='*60}")
        logger.info(f"Phase 0: Adapting to source domain {seq_cfg.source_dataset.name}")
        logger.info(f"{'='*60}")

        source_adapter = self._create_adapter_for_domain(seq_cfg.source_dataset)
        source_result = self._adapt_to_domain(
            model=model,
            adapter=source_adapter,
            dataset_config=seq_cfg.source_dataset,
            baseline_metrics=baseline_metrics,
            domain_index=-1,
            pretrained_snapshot=pretrained_snapshot,
        )

        # Save teacher weights after source adaptation for cross-domain reset
        source_teacher_snapshot = {k: v.clone() for k, v in source_adapter.teacher_weights.items()}

        all_results["domain_results"][seq_cfg.source_dataset.name] = {
            "adaptation": source_result.to_dict(),
            "test_evaluations": {},
        }

        post_source_metrics = self._evaluate_model(model, seq_cfg.source_dataset)
        all_results["domain_results"][seq_cfg.source_dataset.name]["test_evaluations"][seq_cfg.source_dataset.name] = post_source_metrics
        logger.info(f"Post-adaptation QWK on {seq_cfg.source_dataset.name}: {post_source_metrics['qwk']:.4f}")

        for target_idx, target_ds in enumerate(seq_cfg.target_datasets):
            logger.info(f"\n{'='*60}")
            logger.info(f"Phase {target_idx + 1}: Adapting to {target_ds.name}")
            logger.info(f"{'='*60}")

            if seq_cfg.reset_weights_before_each_target and target_idx > 0:
                logger.info("Resetting model weights to pretrained state")
                model = self._load_model()
                self._initialize_classifier_from_prototypes(model, seq_cfg.source_dataset)

            adapter = self._create_adapter_for_domain(target_ds)
            # Reset teacher to post-source checkpoint (prevents APTOS-biased teacher)
            adapter.set_teacher_weights(source_teacher_snapshot)
            logger.info(f"Teacher reset to post-source snapshot for {target_ds.name}")
            domain_result = self._adapt_to_domain(
                model=model,
                adapter=adapter,
                dataset_config=target_ds,
                baseline_metrics=baseline_metrics,
                domain_index=target_idx,
                pretrained_snapshot=pretrained_snapshot,
            )

            all_results["domain_results"][target_ds.name] = {
                "adaptation": domain_result.to_dict(),
                "test_evaluations": {},
            }

            logger.info(f"\nEvaluating on all {len(seen_domains) + 1} seen domains after adapting to {target_ds.name}")
            for seen_ds in seen_domains:
                logger.info(f"  Test-only evaluation on {seen_ds.name} (no adaptation)")
                test_metrics = self._evaluate_model(model, seen_ds)
                all_results["domain_results"][target_ds.name]["test_evaluations"][seen_ds.name] = test_metrics
                logger.info(f"  QWK on {seen_ds.name}: {test_metrics['qwk']:.4f}")

            test_on_target = self._evaluate_model(model, target_ds)
            all_results["domain_results"][target_ds.name]["test_evaluations"][target_ds.name] = test_on_target
            logger.info(f"  QWK on {target_ds.name}: {test_on_target['qwk']:.4f}")

            seen_domains.append(target_ds)

        logger.info(f"\n{'='*60}")
        logger.info("Sequential CTTA Complete - Summary")
        logger.info(f"{'='*60}")
        for domain_name, metrics in all_results["baseline_metrics"].items():
            logger.info(f"Baseline QWK ({domain_name}): {metrics['qwk']:.4f}")
        for target_name, target_data in all_results["domain_results"].items():
            adapt_qwk = target_data["adaptation"]["post_adaptation_metrics"]["qwk"]
            logger.info(f"After adapting to {target_name}:")
            for test_name, test_metrics in target_data["test_evaluations"].items():
                logger.info(f"  QWK on {test_name}: {test_metrics['qwk']:.4f}")

        self._save_results(all_results)
        return all_results

    def _create_adapter_for_domain(self, target_ds) -> CTTAAdapter:
        """Create adapter using per-domain CTTA params if available, else fallback to top-level."""
        from src.adapters.registry import AdapterRegistry

        domain_ctta = target_ds.ctta if target_ds.ctta is not None else self.config.ctta
        adapter_class = AdapterRegistry.get(domain_ctta.method)

        base_kwargs = dict(
            lr=domain_ctta.lr,
            weight_decay=domain_ctta.weight_decay,
            image_size=self.config.model.image_size,
            max_grad_norm=self.config.max_grad_norm,
        )

        method_kwargs = {}
        if domain_ctta.method == "cotta":
            method_kwargs = dict(
                ema_alpha=domain_ctta.ema_alpha,
                restore_prob=domain_ctta.restore_prob,
                num_augmentations=domain_ctta.num_augmentations,
                confidence_threshold=domain_ctta.confidence_threshold,
                teacher_temperature=domain_ctta.teacher_temperature,
                entropy_weight=domain_ctta.entropy_weight,
                diversity_weight=domain_ctta.diversity_weight,
                adapt_layernorm=domain_ctta.adapt_layernorm,
                adapt_last_n_blocks=domain_ctta.adapt_last_n_blocks,
                head_lr_multiplier=domain_ctta.head_lr_multiplier,
                per_class_cap=domain_ctta.per_class_cap,
                kl_label_smoothing=domain_ctta.kl_label_smoothing,
                class_weights=domain_ctta.class_weights,
                confidence_gated_restore=getattr(domain_ctta, 'confidence_gated_restore', False),
                teacher_ensemble_weight=getattr(domain_ctta, 'teacher_ensemble_weight', 0.0),
                use_class_specific_thresholds=getattr(domain_ctta, 'use_class_specific_thresholds', False),
                class_prior_alignment_weight=getattr(domain_ctta, 'class_prior_alignment_weight', 0.0),
                class_prior_alignment_temperature=getattr(domain_ctta, 'class_prior_alignment_temperature', 0.1),
                class_forcing_threshold=getattr(domain_ctta, 'class_forcing_threshold', 0),
            )
        elif domain_ctta.method == "palm":
            method_kwargs = dict(
                # CoTTA stability (hybrid)
                ema_alpha=domain_ctta.ema_alpha,
                restore_prob=domain_ctta.restore_prob,
                num_augmentations=domain_ctta.num_augmentations,
                confidence_threshold=domain_ctta.confidence_threshold,
                teacher_temperature=domain_ctta.teacher_temperature,
                entropy_weight=domain_ctta.entropy_weight,
                diversity_weight=domain_ctta.diversity_weight,
                adapt_layernorm=domain_ctta.adapt_layernorm,
                adapt_last_n_blocks=domain_ctta.adapt_last_n_blocks,
                head_lr_multiplier=domain_ctta.head_lr_multiplier,
                per_class_cap=domain_ctta.per_class_cap,
                # PALM params
                sensitivity_alpha=domain_ctta.sensitivity_alpha,
                selection_percentile=domain_ctta.selection_percentile,
                layer_selection_percentile=domain_ctta.layer_selection_percentile,
                # Collapse detection params
                entropy_collapse_threshold=domain_ctta.entropy_collapse_threshold,
                confidence_spike_threshold=domain_ctta.confidence_spike_threshold,
                confidence_spike_factor=domain_ctta.confidence_spike_factor,
                confidence_ema_alpha=domain_ctta.confidence_ema_alpha,
                collapse_window=domain_ctta.collapse_window,
                # Legacy (ignored by adapter)
                temperature=domain_ctta.temperature,
                layer_selection_threshold=domain_ctta.layer_selection_threshold,
                consistency_lambda=domain_ctta.consistency_lambda,
                entropy_threshold_factor=domain_ctta.entropy_threshold_factor,
            )
        elif domain_ctta.method == "vida":
            method_kwargs = dict(
                vida_rank1=domain_ctta.vida_rank1,
                vida_rank2=domain_ctta.vida_rank2,
                uncertainty_threshold=domain_ctta.uncertainty_threshold,
                num_augmentations=domain_ctta.vida_num_augmentations,
                uncertainty_scale=domain_ctta.uncertainty_scale,
                alpha_teacher=domain_ctta.alpha_teacher,
                alpha_vida=domain_ctta.alpha_vida,
                vida_lr=domain_ctta.vida_lr,
                model_lr=domain_ctta.vida_model_lr,
                restore_prob=domain_ctta.restore_prob,
                ce_loss_weight=domain_ctta.ce_loss_weight,
                pseudo_label_threshold=domain_ctta.pseudo_label_threshold,
            )

        logger.info(f"Creating adapter for {target_ds.name}: method={domain_ctta.method}, "
                    f"lr={domain_ctta.lr}, conf_thresh={domain_ctta.confidence_threshold}")
        return adapter_class(**base_kwargs, **method_kwargs)

    def _adapt_to_domain(
        self,
        model: FoundationModel,
        adapter: CTTAAdapter,
        dataset_config,
        baseline_metrics: Dict[str, float],
        domain_index: int,
        pretrained_snapshot: Dict[str, Any] = None,
    ) -> DomainResult:
        """Run CTTA adaptation loop on a single domain (simplified, no shift detection)."""
        domain_ctta = dataset_config.ctta if dataset_config.ctta is not None else self.config.ctta

        setup_logging(str(self.adapt_dir / f"adapt_{dataset_config.name}.log"))
        result = DomainResult(
            domain_name=dataset_config.name,
            description=f"CTTA adaptation on {dataset_config.name}",
        )
        result.baseline_metrics = baseline_metrics

        snap = {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}
        adapter.setup(model, snap, pretrained_snapshot=pretrained_snapshot)

        dataset_class = DatasetRegistry.get(dataset_config.name)
        dataset = dataset_class(
            data_dir=dataset_config.data_dir,
            image_size=dataset_config.image_size,
            train=False,
            normalize_mean=self.config.model.normalize_mean,
            normalize_std=self.config.model.normalize_std,
        )

        is_palm = domain_ctta.method == "palm"
        always_adapt = is_palm or domain_ctta.adapt_every_batch

        batch_size = domain_ctta.batch_size if hasattr(domain_ctta, 'batch_size') else self.config.ctta_batch_size
        n_batches = len(dataset) // batch_size
        if len(dataset) % batch_size != 0:
            n_batches += 1
        max_batches = n_batches

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=dataset_config.num_workers,
            pin_memory=True,
            persistent_workers=(dataset_config.num_workers > 0),
        )

        model.backbone.eval()
        model.classifier.eval()
        torch.cuda.empty_cache()

        batch_accuracies = []
        num_adaptations = 0

        pbar = tqdm(enumerate(loader), total=min(len(loader), max_batches),
                    desc=f"Adapting to {dataset_config.name}")
        for batch_idx, (images, labels) in pbar:
            if batch_idx >= max_batches:
                break

            images = images.to(self.device)
            labels = labels.to(self.device)

            with torch.no_grad():
                logits = model(images)

            if always_adapt:
                num_adaptations += 1
                model.backbone.train()
                model.classifier.train()
                metrics = adapter.adapt_step(images, logits)
                model.backbone.eval()
                model.classifier.eval()

                forcing = metrics.get("class_forcing_active", False)
                starvation = metrics.get("starvation_counts", {})
                log_line = self._adapt_log_line(batch_idx, metrics)
                if forcing:
                    starving = {k: v for k, v in starvation.items() if v >= 3}
                    log_line += f" [FORCED] starvation={starving}"
                logger.info(log_line)
                result.adaptation_steps.append({
                    "batch_idx": batch_idx,
                    "signals": {},
                    "metrics": metrics,
                })

            with torch.no_grad():
                logits = model(images)
                preds = torch.argmax(logits, dim=-1)
                batch_acc = (preds == labels).float().mean().item()

            batch_accuracies.append(batch_acc)
            result.batch_metrics.append({
                "batch_idx": batch_idx,
                "accuracy": batch_acc,
            })
            pbar.set_postfix(acc=f"{batch_acc:.4f}", adaptations=num_adaptations)

        logger.info(f"Evaluating after adaptation on {dataset_config.name}")
        post_metrics = self._evaluate_model(model, dataset_config)
        result.post_adaptation_metrics = post_metrics
        result.num_adaptations = num_adaptations
        result.total_batches = max_batches

        # guaranteed fallback — if post-adaptation QWK or accuracy is
        # below baseline, revert to source weights and re-evaluate. Reported
        # post-adaptation metrics can never be worse than baseline by construction.
        if hasattr(adapter, "check_fallback"):
            reverted = adapter.check_fallback(baseline_metrics, post_metrics)
            if reverted:
                logger.warning(
                    f"Fallback: post QWK {post_metrics['qwk']:.4f} < baseline "
                    f"{baseline_metrics['qwk']:.4f}. Reverting to source weights."
                )
                model.backbone.eval()
                model.classifier.eval()
                post_metrics = self._evaluate_model(model, dataset_config)
                result.post_adaptation_metrics = post_metrics
                result.fallback_reverted = True
                logger.info(f"Post-fallback QWK: {post_metrics['qwk']:.4f}")

        logger.info(f"Post-adaptation QWK: {post_metrics['qwk']:.4f}")
        logger.info(f"Adaptations: {num_adaptations}/{max_batches}")

        return result
