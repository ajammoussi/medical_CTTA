"""CTTA Runner for sequential multi-domain adaptation.

Protocol: source -> target1 -> target2 -> ... (no weight reset between domains).
This measures catastrophic forgetting and domain robustness.
"""

import torch
import json
import logging
import copy
from pathlib import Path
from typing import Dict, Any, List, Optional
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import ExperimentConfig, SequentialCTTAConfig, DomainTransitionConfig, LRSchedulerConfig, CTTAConfig
from src.models.registry import ModelRegistry
from src.models.base import FoundationModel
from src.data.registry import DatasetRegistry
from src.adapters.registry import AdapterRegistry
from src.adapters.base import CTTAAdapter
from src.evaluation.metrics import quadratic_weighted_kappa, per_class_accuracy, forgetting_metric
from src.shift_detection.combined import CombinedShiftDetector
from src.utils.seed import set_seed
from src.utils.checkpointing import (
    save_checkpoint, load_checkpoint, save_adaptation_checkpoint,
    load_adaptation_checkpoint, find_latest_epoch_checkpoint,
    find_latest_adaptation_checkpoint, save_results,
    save_train_metrics,
)
from src.utils.logging import setup_logging


logger = logging.getLogger(__name__)


class DomainResult:

    def __init__(self, domain_name: str, description: str):
        self.domain_name = domain_name
        self.description = description
        self.baseline_metrics: Dict[str, float] = {}
        self.post_adaptation_metrics: Dict[str, float] = {}
        self.batch_metrics: List[Dict[str, Any]] = []
        self.adaptation_steps: List[Dict[str, Any]] = []
        self.num_adaptations: int = 0
        self.total_batches: int = 0

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
        }


class CTTARunner:
    """Runner for Continual Test-Time Adaptation experiments.

    Supports two modes:
    1. Single-target: source -> one target domain
    2. Sequential: source -> target1 -> target2 -> ... (no weight reset)
    """

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # Organized subdirectories
        self.source_dir = self.output_dir / "source"
        self.adapt_dir = self.output_dir / "adapt"
        self.plots_dir = self.output_dir / "plots"
        self.source_dir.mkdir(exist_ok=True)
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
        if self.config.sequential_ctta.enabled:
            return self._run_sequential()
        else:
            return self._run_single_target()

    def _run_sequential(self) -> Dict[str, Any]:
        seq_config = self.config.sequential_ctta
        logger.info("Starting sequential CTTA experiment")
        logger.info(f"Source: {seq_config.source_domain.description}")
        for i, td in enumerate(seq_config.target_domains):
            logger.info(f"Target {i+1}: {td.description}")

        model = self._load_model()

        # 2. Train on source domain (with resume) or load best model
        source_model_path = self.source_dir / "source_model.pth"
        source_classifier_path = self.source_dir / "source_classifier.pth"
        resume_ckpt = find_latest_epoch_checkpoint(str(self.source_dir))

        if resume_ckpt:
            logger.info(f"Resuming training from checkpoint")
            ckpt = load_checkpoint(resume_ckpt, model=model.backbone, map_location=self.device)
            if "classifier_state_dict" in ckpt:
                model.classifier.load_state_dict(ckpt["classifier_state_dict"])
            logger.info(f"Resumed from epoch {ckpt.get('epoch', '?')}, best QWK: {ckpt.get('best_metric', '?'):.4f}")
        elif source_model_path.exists():
            logger.info(f"Loading best source model from {source_model_path.name}")
            model.backbone.load_state_dict(torch.load(str(source_model_path), map_location=self.device, weights_only=True))
            if source_classifier_path.exists():
                model.classifier.load_state_dict(torch.load(str(source_classifier_path), map_location=self.device, weights_only=True))
        else:
            shared_dir = Path(self.config.shared_source_dir)
            shared_model = shared_dir / "source_model.pth"
            shared_classifier = shared_dir / "source_classifier.pth"
            if shared_model.exists():
                logger.info(f"Loading shared source model from {shared_dir}")
                model.backbone.load_state_dict(torch.load(str(shared_model), map_location=self.device, weights_only=True))
                if shared_classifier.exists():
                    model.classifier.load_state_dict(torch.load(str(shared_classifier), map_location=self.device, weights_only=True))

            else:
                logger.info(f"Training on source domain: {seq_config.source_domain.dataset.name}")
                self._train_source_domain(model, seq_config.source_domain)

        # 3. Evaluate baseline on source test set
        logger.info("Evaluating source baseline")
        source_baseline = self._evaluate_model(
            model, seq_config.source_domain.dataset, train=False
        )

        # 4. Sequential adaptation on target domains (with resume)
        domain_results: List[DomainResult] = []
        start_domain = 0
        is_palm = self.config.ctta.method == "palm"
        palm_adapter = None

        adapt_ckpt_path = find_latest_adaptation_checkpoint(str(self.adapt_dir))
        if adapt_ckpt_path:
            logger.info(f"Found adaptation checkpoint: {adapt_ckpt_path}")
            adapt_ckpt = load_adaptation_checkpoint(adapt_ckpt_path, model=model.backbone)
            start_domain = adapt_ckpt.get("domain_index", 0)
            logger.info(f"Resuming from domain index {start_domain}")

        for i in range(start_domain, len(seq_config.target_domains)):
            target_domain = seq_config.target_domains[i]
            logger.info(f"\n{'='*60}")
            logger.info(f"Adapting to target domain {i+1}: {target_domain.description}")
            logger.info(f"{'='*60}")

            # PALM adapter persists across domains (running_sensitivity EMA, optimizer state)
            if is_palm:
                if palm_adapter is None:
                    palm_adapter = self._create_adapter()
                adapter = palm_adapter
            else:
                adapter = self._create_adapter()

            domain_result = self._adapt_to_domain(
                model=model,
                adapter=adapter,
                domain_config=target_domain,
                source_baseline=source_baseline,
                domain_index=i,
            )
            domain_results.append(domain_result)

            model_state = {k: v.cpu() for k, v in model.backbone.state_dict().items()}
            torch.cuda.empty_cache()
            save_adaptation_checkpoint(
                path=str(self.adapt_dir / "adapt_checkpoint.pth"),
                model_state_dict=model_state,
                adapter_state_dict=adapter.state_dict(),
                domain_index=i,
                batch_index=-1,
                extra={"domain_name": target_domain.dataset.name},
            )

        # Delete rolling adaptation checkpoint — results saved separately
        (self.adapt_dir / "adapt_checkpoint.pth").unlink(missing_ok=True)

        results = self._compile_sequential_results(
            source_baseline=source_baseline,
            domain_results=domain_results,
        )
        self._save_results(results)
        return results

    def _run_single_target(self) -> Dict[str, Any]:
        logger.info("Starting single-target CTTA experiment")

        model = self._load_model()

        logger.info(f"Training on source: {self.config.source_dataset.name}")
        self._train_source_domain(
            model,
            DomainTransitionConfig(dataset=self.config.source_dataset)
        )

        source_baseline = self._evaluate_model(
            model, self.config.source_dataset, train=False
        )

        adapter = self._create_adapter()
        target_config = DomainTransitionConfig(dataset=self.config.target_dataset)
        domain_result = self._adapt_to_domain(
            model=model,
            adapter=adapter,
            domain_config=target_config,
            source_baseline=source_baseline,
            domain_index=0,
        )

        results = {
            "mode": "single_target",
            "source_baseline": source_baseline,
            "target_result": domain_result.to_dict(),
            "final_metrics": domain_result.post_adaptation_metrics,
        }

        self._save_results(results)
        return results

    def _load_model(self) -> FoundationModel:
        model_class = ModelRegistry.get(self.config.model.name)
        model = model_class(
            num_classes=self.config.model.num_classes,
            freeze_layers=self.config.model.freeze_layers,
            checkpoint=self.config.model.checkpoint,
        )
        model.load_weights()
        model = model.to(self.device)

        if self.config.gradient_checkpointing:
            if hasattr(model.backbone, "set_grad_checkpointing"):
                model.backbone.set_grad_checkpointing(enable=True)
            elif hasattr(model.backbone, "grad_checkpointing"):
                model.backbone.grad_checkpointing = True

        source_ckpt = self.source_dir / "source_model.pth"
        if source_ckpt.exists():
            logger.info("Loading source checkpoint")
            state = torch.load(source_ckpt, map_location=self.device)
            if "model_state_dict" in state:
                model.backbone.load_state_dict(state["model_state_dict"])
            else:
                model.backbone.load_state_dict(state)

        classifier_ckpt = self.source_dir / "source_classifier.pth"
        if classifier_ckpt.exists():
            model.classifier.load_state_dict(
                torch.load(classifier_ckpt, map_location=self.device)
            )
            logger.info("Loaded source classifier checkpoint")

        return model

    def _create_adapter(self) -> CTTAAdapter:
        adapter_class = AdapterRegistry.get(self.config.ctta.method)
        base_kwargs = dict(
            lr=self.config.ctta.lr,
            weight_decay=self.config.ctta.weight_decay,
            image_size=self.config.model.image_size,
        )
        method_kwargs = self._get_method_kwargs()
        return adapter_class(**base_kwargs, **method_kwargs)

    def _get_method_kwargs(self) -> Dict[str, Any]:
        cfg = self.config.ctta
        method = cfg.method
        if method == "cotta":
            return dict(
                ema_alpha=cfg.ema_alpha,
                restore_prob=cfg.restore_prob,
                num_augmentations=cfg.num_augmentations,
                confidence_threshold=cfg.confidence_threshold,
            )
        elif method == "palm":
            return dict(
                temperature=cfg.temperature,
                layer_selection_threshold=cfg.layer_selection_threshold,
                selection_percentile=cfg.selection_percentile,
                sensitivity_alpha=cfg.sensitivity_alpha,
                consistency_lambda=cfg.consistency_lambda,
                entropy_threshold_factor=cfg.entropy_threshold_factor,
                num_augmentations=cfg.num_augmentations,
            )
        return {}

    def _train_source_domain(
        self,
        model: FoundationModel,
        domain_config: DomainTransitionConfig,
    ) -> None:
        setup_logging(str(self.source_dir / "train.log"))
        dataset_class = DatasetRegistry.get(domain_config.dataset.name)
        train_dataset = dataset_class(
            data_dir=domain_config.dataset.data_dir,
            image_size=domain_config.dataset.image_size,
            train=True,
        )

        # ---- Class-weighted loss ----
        class_dist = train_dataset.get_class_distribution()
        class_counts = [class_dist.get(i, 0) for i in range(self.config.model.num_classes)]
        weights = torch.tensor(
            [1.0 / max(c, 1) for c in class_counts], dtype=torch.float32, device=self.device
        )
        weights = weights / weights.sum() * self.config.model.num_classes
        criterion = torch.nn.CrossEntropyLoss(weight=weights)

        loader = DataLoader(
            train_dataset,
            batch_size=self.config.ctta.batch_size,
            shuffle=True,
            num_workers=domain_config.dataset.num_workers,
            pin_memory=True,
            persistent_workers=(domain_config.dataset.num_workers > 0),
            drop_last=False,
        )

        model.backbone.train()
        model.classifier.train()

        # ---- Layer-wise learning rate decay (LLRD) for ViT-L ----
        base_lr = self.config.model.lr
        num_blocks = len(model.backbone.blocks)
        param_groups = []
        for i, block in enumerate(model.backbone.blocks):
            lr_scale = self.config.model.layer_decay ** (num_blocks - 1 - i)
            param_groups.append({"params": block.parameters(), "lr": base_lr * lr_scale})
        param_groups.append({"params": model.backbone.fc_norm.parameters(), "lr": base_lr})
        param_groups.append({"params": model.classifier.parameters(), "lr": base_lr})
        optimizer = torch.optim.AdamW(
            param_groups,
            lr=base_lr,
            weight_decay=self.config.model.weight_decay,
        )

        # ---- Single cosine decay (no restarts) + warmup ----
        lr_scheduler_config = self.config.lr_scheduler
        total_epochs = self.config.num_epochs_source
        warmup_epochs = lr_scheduler_config.warmup_epochs

        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return float(epoch + 1) / float(max(1, warmup_epochs))
            progress = float(epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
            return max(
                lr_scheduler_config.min_lr / self.config.model.lr,
                0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159)))
            )

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        start_epoch = 0
        best_qwk = 0.0
        patience_counter = 0
        patience = 15
        resume_ckpt = find_latest_epoch_checkpoint(str(self.source_dir))
        if resume_ckpt:
            logger.info(f"Resuming training from {resume_ckpt}")
            ckpt = load_checkpoint(
                resume_ckpt, model=model.backbone, optimizer=optimizer, scheduler=scheduler,
                map_location=self.device,
            )
            start_epoch = ckpt.get("epoch", 0) + 1
            best_qwk = ckpt.get("best_metric", 0.0)
            model.backbone.train()
            model.classifier.train()

        scaler = torch.amp.GradScaler(device="cuda", enabled=self.config.use_amp)
        all_params = list(model.backbone.parameters()) + list(model.classifier.parameters())

        pbar_epoch = tqdm(range(start_epoch, total_epochs),
                          desc="Training source", initial=start_epoch, total=total_epochs)
        for epoch in pbar_epoch:
            epoch_loss = 0.0
            optimizer.zero_grad()
            scheduler.step()

            pbar_batch = tqdm(enumerate(loader), total=len(loader),
                              desc=f"Epoch {epoch+1}/{total_epochs}", leave=False)
            for batch_idx, (images, labels) in pbar_batch:
                images = images.to(self.device)
                labels = labels.to(self.device)

                with torch.amp.autocast(device_type="cuda", enabled=self.config.use_amp):
                    logits = model(images)
                    loss = criterion(logits, labels) / self.config.gradient_accumulation_steps

                scaler.scale(loss).backward()

                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(all_params, self.config.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

                epoch_loss += loss.item() * self.config.gradient_accumulation_steps
                pbar_batch.set_postfix(loss=f"{epoch_loss / (batch_idx + 1):.4f}")

            avg_loss = epoch_loss / len(loader)
            current_lr = optimizer.param_groups[-1]["lr"]

            model.backbone.eval()
            model.classifier.eval()
            all_preds = []
            all_labels = []
            with torch.no_grad():
                val_dataset = dataset_class(
                    data_dir=domain_config.dataset.data_dir,
                    image_size=domain_config.dataset.image_size,
                    train=False,
                )
                val_loader = DataLoader(
                    val_dataset,
                    batch_size=self.config.ctta.batch_size,
                    shuffle=False,
                    num_workers=domain_config.dataset.num_workers,
                    pin_memory=True,
                    persistent_workers=(domain_config.dataset.num_workers > 0),
                )
                for val_images, val_labels in val_loader:
                    val_images = val_images.to(self.device)
                    logits = model(val_images)
                    preds = torch.argmax(logits, dim=-1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(val_labels.numpy())

            qwk = quadratic_weighted_kappa(all_labels, all_preds)
            model.backbone.train()
            model.classifier.train()

            pbar_epoch.set_postfix(loss=f"{avg_loss:.4f}", qwk=f"{qwk:.4f}")
            logger.info(
                f"Epoch {epoch+1}/{total_epochs}, Loss: {avg_loss:.4f}, "
                f"LR: {current_lr:.2e}, Val QWK: {qwk:.4f}, Best QWK: {max(best_qwk, qwk):.4f}"
            )

            # Rolling checkpoint
            save_checkpoint(
                path=str(self.source_dir / "checkpoint.pth"),
                model_state_dict=model.backbone.state_dict(),
                optimizer_state_dict=optimizer.state_dict(),
                scheduler_state_dict=scheduler.state_dict(),
                epoch=epoch,
                best_metric=max(best_qwk, qwk),
                extra={"classifier_state_dict": model.classifier.state_dict()},
            )

            save_train_metrics(
                path=str(self.source_dir / "train_metrics.jsonl"),
                epoch=epoch, loss=avg_loss, lr=current_lr,
                val_qwk=qwk, best_qwk=max(best_qwk, qwk),
            )

            if qwk > best_qwk:
                best_qwk = qwk
                patience_counter = 0
                torch.save(model.backbone.state_dict(), self.source_dir / "source_model.pth")
                torch.save(model.classifier.state_dict(), self.source_dir / "source_classifier.pth")
                logger.info(f"  New best model with QWK: {qwk:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch+1} (no QWK improvement for {patience} epochs)")
                    break

        (self.source_dir / "checkpoint.pth").unlink(missing_ok=True)

        logger.info(f"Source training complete. Best QWK: {best_qwk:.4f}")

    def _evaluate_model(
        self,
        model: FoundationModel,
        dataset_config,
        train: bool = False,
    ) -> Dict[str, float]:
        if isinstance(dataset_config, DomainTransitionConfig):
            dataset_config = dataset_config.dataset

        dataset_class = DatasetRegistry.get(dataset_config.name)
        dataset = dataset_class(
            data_dir=dataset_config.data_dir,
            image_size=dataset_config.image_size,
            train=train,
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

        return {
            "qwk": qwk,
            "per_class_accuracy": per_class,
            "num_samples": len(all_labels),
        }

    def _adapt_to_domain(
        self,
        model: FoundationModel,
        adapter: CTTAAdapter,
        domain_config: DomainTransitionConfig,
        source_baseline: Dict[str, float],
        domain_index: int,
    ) -> DomainResult:
        setup_logging(str(self.adapt_dir / f"adapt_{domain_config.dataset.name}.log"))
        result = DomainResult(
            domain_name=domain_config.dataset.name,
            description=domain_config.description,
        )

        logger.info(f"Evaluating baseline on {domain_config.dataset.name}")
        baseline_metrics = self._evaluate_model(
            model, domain_config.dataset, train=False
        )
        result.baseline_metrics = baseline_metrics
        logger.info(f"Baseline QWK: {baseline_metrics['qwk']:.4f}")

        # Capture current model state as source for stochastic restore
        snap = {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}
        adapter.setup(model, snap)

        dataset_class = DatasetRegistry.get(domain_config.dataset.name)
        dataset = dataset_class(
            data_dir=domain_config.dataset.data_dir,
            image_size=domain_config.dataset.image_size,
            train=False,
        )

        is_palm = self.config.ctta.method == "palm"

        if is_palm:
            # PALM paper protocol: adapt every batch, include final partial batch
            n_batches = len(dataset) // self.config.ctta_batch_size
            if len(dataset) % self.config.ctta_batch_size != 0:
                n_batches += 1
            max_batches = domain_config.num_batches or n_batches
        else:
            max_batches = domain_config.num_batches or (len(dataset) // self.config.ctta_batch_size)

        loader = DataLoader(
            dataset,
            batch_size=self.config.ctta_batch_size,
            shuffle=False,
            num_workers=domain_config.dataset.num_workers,
            pin_memory=True,
            persistent_workers=(domain_config.dataset.num_workers > 0),
        )

        shift_detector = CombinedShiftDetector(
            entropy_threshold=self.config.shift_detection.entropy_threshold,
            drift_threshold=self.config.shift_detection.drift_threshold,
            distribution_threshold=self.config.shift_detection.distribution_threshold,
            ema_momentum=self.config.shift_detection.ema_momentum,
            window_size=self.config.shift_detection.window_size,
        )

        cal_domain = (self.config.sequential_ctta.source_domain
                      if self.config.sequential_ctta.enabled
                      else DomainTransitionConfig(dataset=self.config.source_dataset))
        self._calibrate_shift_detector(
            shift_detector, model, cal_domain,
        )

        # PALM keeps the model in eval() mode (drop_path=0.2 adds stochasticity in train mode)
        model.backbone.eval()
        model.classifier.eval()
        torch.cuda.empty_cache()

        batch_accuracies = []
        num_adaptations = 0

        pbar = tqdm(enumerate(loader), total=min(len(loader), max_batches),
                    desc=f"Adapting to {domain_config.dataset.name}")
        for batch_idx, (images, labels) in pbar:
            if batch_idx >= max_batches:
                break

            images = images.to(self.device)
            labels = labels.to(self.device)

            with torch.no_grad():
                logits = model(images)
                embeddings = model.extract_embedding(images)

            shift_detector.update(images, logits, embeddings)

            if is_palm or shift_detector.detect():
                if not is_palm:
                    logger.info(f"  Shift detected at batch {batch_idx}")
                num_adaptations += 1
                if is_palm:
                    metrics = adapter.adapt_step(images, logits)
                else:
                    model.backbone.train()
                    model.classifier.train()
                    metrics = adapter.adapt_step(images, logits)
                    model.backbone.eval()
                    model.classifier.eval()
                mean_max_prob = metrics.get("mean_max_prob", -1)
                n_confident = metrics.get("confident_samples", metrics.get("num_confident_samples", -1))
                logger.info(f"  Adapt step batch {batch_idx}: loss={metrics['loss']:.4f}, "
                             f"conf_samples={n_confident}, mean_max_prob={mean_max_prob:.4f}")
                result.adaptation_steps.append({
                    "batch_idx": batch_idx,
                    "signals": shift_detector.get_signals(),
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

        logger.info(f"Evaluating after adaptation on {domain_config.dataset.name}")
        post_metrics = self._evaluate_model(
            model, domain_config.dataset, train=False
        )
        result.post_adaptation_metrics = post_metrics
        result.num_adaptations = num_adaptations
        result.total_batches = max_batches

        logger.info(f"Post-adaptation QWK: {post_metrics['qwk']:.4f}")
        logger.info(f"Adaptations: {num_adaptations}/{max_batches}")

        return result

    def _calibrate_shift_detector(
        self,
        shift_detector: CombinedShiftDetector,
        model: FoundationModel,
        domain_config: DomainTransitionConfig,
    ) -> None:
        dataset_class = DatasetRegistry.get(domain_config.dataset.name)
        dataset = dataset_class(
            data_dir=domain_config.dataset.data_dir,
            image_size=domain_config.dataset.image_size,
            train=True,
        )
        cal_loader = DataLoader(
            dataset,
            batch_size=self.config.ctta_batch_size,
            shuffle=True,
            num_workers=domain_config.dataset.num_workers,
            pin_memory=True,
        )

        entropy_values = []
        embeddings = []
        predictions = []

        model.backbone.eval()
        model.classifier.eval()

        with torch.no_grad():
            for images, labels in cal_loader:
                images = images.to(self.device)
                logits = model(images)
                emb = model.extract_embedding(images)
                probs = torch.softmax(logits, dim=-1)
                log_probs = torch.log(probs + 1e-10)
                entropy = -(probs * log_probs).sum(dim=-1)
                preds = torch.argmax(logits, dim=-1)

                entropy_values.extend(entropy.cpu().tolist())
                embeddings.append(emb.cpu())
                predictions.extend(preds.cpu().tolist())

                if len(entropy_values) >= 500:
                    break

        if embeddings:
            embeddings_cat = torch.cat(embeddings, dim=0)
            shift_detector.calibrate(entropy_values, embeddings_cat, predictions)
            logger.info(f"Shift detector calibrated on SOURCE TRAINING SET ({len(entropy_values)} samples)")

    def _compile_sequential_results(
        self,
        source_baseline: Dict[str, float],
        domain_results: List[DomainResult],
    ) -> Dict[str, Any]:
        results = {
            "mode": "sequential",
            "method": self.config.ctta.method,
            "source_baseline": source_baseline,
            "domain_results": [dr.to_dict() for dr in domain_results],
        }

        if len(domain_results) >= 2:
            final_domain = domain_results[-1]
            if final_domain.domain_name == self.config.sequential_ctta.source_domain.dataset.name:
                forgetting = forgetting_metric(
                    source_baseline["qwk"],
                    final_domain.post_adaptation_metrics["qwk"],
                )
                results["forgetting"] = forgetting
                logger.info(f"Forgetting (source -> ... -> source): {forgetting:.4f}")

            for dr in domain_results:
                logger.info(
                    f"Domain {dr.domain_name}: "
                    f"baseline QWK={dr.baseline_metrics['qwk']:.4f}, "
                    f"post QWK={dr.post_adaptation_metrics['qwk']:.4f}"
                )

        if domain_results:
            total_adaptations = sum(dr.num_adaptations for dr in domain_results)
            total_batches = sum(dr.total_batches for dr in domain_results)
            results["num_adaptations"] = total_adaptations
            results["total_batches"] = total_batches
            if total_batches > 0:
                results["adaptation_rate"] = total_adaptations / total_batches
            results["final_metrics"] = domain_results[-1].post_adaptation_metrics

        return results

    def _save_results(self, results: Dict[str, Any]) -> None:
        save_results(results, str(self.output_dir / "ctta_results.json"))
        logger.info(f"Results saved to {self.output_dir / 'ctta_results.json'}")
