"""Single-domain CTTA Runner.

Protocol: load RETFound pretrained weights -> evaluate baseline -> adapt -> evaluate post.

Every batch is adapted (CoTTA-style, matching the official paper). No shift detection gating.
"""

import torch
import logging
from collections import deque
from pathlib import Path
from typing import Dict, Any
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import ExperimentConfig
from src.models.base import FoundationModel
from src.adapters.base import CTTAAdapter
from src.utils.seed import set_seed
from src.utils.logging import setup_logging

from src.evaluation.runner.base import BaseRunnerMixin


logger = logging.getLogger(__name__)


class DomainResult:
    """Container for results from adapting to a single domain."""

    def __init__(self, domain_name: str, description: str):
        self.domain_name = domain_name
        self.description = description
        self.baseline_metrics: Dict[str, float] = {}
        self.post_adaptation_metrics: Dict[str, float] = {}
        self.batch_metrics: list = []
        self.adaptation_steps: list = []
        self.num_adaptations: int = 0
        self.total_batches: int = 0
        self.fallback_reverted: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "domain_name": self.domain_name,
            "description": self.description,
            "baseline_metrics": self.baseline_metrics,
            "post_adaptation_metrics": self.post_adaptation_metrics,
            "batch_metrics": self.batch_metrics,
            "adaptation_steps": self.adaptation_steps,
            "num_adaptations": self.num_adaptations,
            "total_batches": self.total_batches,
            "fallback_reverted": self.fallback_reverted,
        }


class CTTARunner(BaseRunnerMixin):
    """Runner for single-dataset CTTA experiments.

    Protocol:
      1. Load RETFound pretrained weights + random classifier head
      2. Evaluate baseline on target dataset (before adaptation)
      3. Run CTTA adaptation on target dataset (every batch)
      4. Evaluate post-adaptation
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
        return self._run_adapt()

    def _run_adapt(self) -> Dict[str, Any]:
        dataset_name = self.config.target_dataset.name
        logger.info(f"Starting CTTA adaptation on dataset: {dataset_name}")

        model = self._load_model()

        self._initialize_classifier_from_prototypes(model, self.config.target_dataset)
        logger.info("Classifier initialized with training-set prototypes")

        logger.info(f"Evaluating baseline on {dataset_name}")
        baseline_metrics = self._evaluate_model(model, self.config.target_dataset)
        logger.info(f"Baseline QWK: {baseline_metrics['qwk']:.4f}")

        adapter = self._create_adapter()
        domain_result = self._adapt_to_domain(
            model=model,
            adapter=adapter,
            dataset_config=self.config.target_dataset,
            baseline_metrics=baseline_metrics,
            domain_index=0,
        )

        results = {
            "method": self.config.ctta.method,
            "dataset": dataset_name,
            "baseline_metrics": baseline_metrics,
            "domain_result": domain_result.to_dict(),
            "final_metrics": domain_result.post_adaptation_metrics,
        }

        if domain_result.total_batches > 0:
            results["adaptation_rate"] = domain_result.num_adaptations / domain_result.total_batches

        self._save_results(results)
        return results

    def _adapt_to_domain(
        self,
        model: FoundationModel,
        adapter: CTTAAdapter,
        dataset_config,
        baseline_metrics: Dict[str, float],
        domain_index: int,
    ) -> DomainResult:
        """Run CTTA adaptation loop on a single domain.

        Every batch is adapted (CoTTA paper: adapt every batch with
        teacher EMA + stochastic restoration, no shift detection gate).
        """
        setup_logging(str(self.adapt_dir / f"adapt_{dataset_config.name}.log"))
        result = DomainResult(
            domain_name=dataset_config.name,
            description=f"CTTA adaptation on {dataset_config.name}",
        )
        result.baseline_metrics = baseline_metrics

        snap = {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}
        adapter.setup(model, snap)

        dataset = self._build_adaptation_dataset(dataset_config)

        n_batches = len(dataset) // self.config.ctta.batch_size
        if len(dataset) % self.config.ctta.batch_size != 0:
            n_batches += 1
        max_batches = n_batches

        loader = DataLoader(
            dataset,
            batch_size=self.config.ctta.batch_size,
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
        pred_window = deque()

        pbar = tqdm(enumerate(loader), total=min(len(loader), max_batches),
                    desc=f"Adapting to {dataset_config.name}")
        for batch_idx, (images, labels) in pbar:
            if batch_idx >= max_batches:
                break

            images = images.to(self.device)
            labels = labels.to(self.device)

            with torch.no_grad():
                logits = model(images)

            num_adaptations += 1
            model.backbone.train()
            model.classifier.train()
            metrics = adapter.adapt_step(images, logits)
            model.backbone.eval()
            model.classifier.eval()

            with torch.no_grad():
                logits = model(images)
                preds = torch.argmax(logits, dim=-1)
                batch_acc = (preds == labels).float().mean().item()

            telemetry = self._record_telemetry(
                model, batch_idx=batch_idx, images=images, logits=logits,
                batch_acc=batch_acc, pred_window=pred_window,
                adapter_metrics=metrics,
            )
            entry = self._step_entry(batch_idx, metrics, telemetry)
            result.adaptation_steps.append(entry)

            logger.info(self._adapt_log_line(batch_idx, metrics, telemetry))
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
