import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import logging
from src.config import ExperimentConfig, load_config
from src.models.registry import ModelRegistry
from src.data.registry import DatasetRegistry
from src.utils.seed import set_seed
from src.utils.checkpointing import (
    save_checkpoint, load_checkpoint,
    find_latest_epoch_checkpoint,
)
from src.evaluation.metrics import quadratic_weighted_kappa


def train_source(config: ExperimentConfig) -> None:
    """Train source model on source dataset with AMP, scheduler, and resume."""
    logger = logging.getLogger(__name__)
    logger.info("Starting source model training")

    set_seed(config.seed)
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")

    model_class = ModelRegistry.get(config.model.name)
    model = model_class(
        num_classes=config.model.num_classes,
        freeze_layers=config.model.freeze_layers,
        checkpoint=config.model.checkpoint,
    )
    model.load_weights()
    model = model.to(device)

    if config.gradient_checkpointing:
        if hasattr(model.backbone, "set_grad_checkpointing"):
            model.backbone.set_grad_checkpointing(enable=True)
        elif hasattr(model.backbone, "grad_checkpointing"):
            model.backbone.grad_checkpointing = True

    dataset_class = DatasetRegistry.get(config.source_dataset.name)
    train_dataset = dataset_class(
        data_dir=config.source_dataset.data_dir,
        image_size=config.source_dataset.image_size,
        train=True,
    )
    test_dataset = dataset_class(
        data_dir=config.source_dataset.data_dir,
        image_size=config.source_dataset.image_size,
        train=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.ctta.batch_size,
        shuffle=True,
        num_workers=config.source_dataset.num_workers,
        pin_memory=True,
        persistent_workers=(config.source_dataset.num_workers > 0),
        drop_last=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.ctta.batch_size,
        shuffle=False,
        num_workers=config.source_dataset.num_workers,
        pin_memory=True,
        persistent_workers=(config.source_dataset.num_workers > 0),
        drop_last=False,
    )

    # ---- Class-weighted loss ----
    class_dist = train_dataset.get_class_distribution()
    class_counts = [class_dist.get(i, 0) for i in range(config.model.num_classes)]
    weights = torch.tensor(
        [1.0 / max(c, 1) for c in class_counts], dtype=torch.float32, device=device
    )
    weights = weights / weights.sum() * config.model.num_classes
    criterion = nn.CrossEntropyLoss(weight=weights)

    # ---- Layer-wise learning rate decay (LLRD) for ViT-L ----
    base_lr = config.model.lr
    num_blocks = len(model.backbone.blocks)
    param_groups = []
    for i, block in enumerate(model.backbone.blocks):
        lr_scale = config.model.layer_decay ** (num_blocks - 1 - i)
        param_groups.append({"params": block.parameters(), "lr": base_lr * lr_scale})
    param_groups.append({"params": model.backbone.fc_norm.parameters(), "lr": base_lr})
    param_groups.append({"params": model.classifier.parameters(), "lr": base_lr})
    optimizer = optim.AdamW(
        param_groups,
        lr=base_lr,
        weight_decay=config.model.weight_decay,
    )

    lr_scheduler_config = config.lr_scheduler
    total_epochs = config.num_epochs_source
    warmup_epochs = lr_scheduler_config.warmup_epochs

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(max(1, warmup_epochs))
        progress = float(epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
        return max(
            lr_scheduler_config.min_lr / config.model.lr,
            0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159)))
        )

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    start_epoch = 0
    best_qwk = 0.0
    patience_counter = 0
    patience = 10
    all_params = list(model.backbone.parameters()) + list(model.classifier.parameters())
    resume_ckpt = find_latest_epoch_checkpoint(str(output_dir))
    if resume_ckpt:
        logger.info(f"Resuming training from {resume_ckpt}")
        ckpt = load_checkpoint(
            resume_ckpt, model=model.backbone, optimizer=optimizer, scheduler=scheduler,
            map_location=device,
        )
        start_epoch = ckpt.get("epoch", 0) + 1
        best_qwk = ckpt.get("best_metric", 0.0)
        if "classifier_state_dict" in ckpt:
            model.classifier.load_state_dict(ckpt["classifier_state_dict"])
        model.backbone.train()
        model.classifier.train()

    scaler = torch.amp.GradScaler(device="cuda", enabled=config.use_amp)

    for epoch in range(start_epoch, total_epochs):
        scheduler.step()
        model.backbone.train()
        model.classifier.train()

        train_loss = 0.0
        optimizer.zero_grad()

        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            with torch.amp.autocast(device_type="cuda", enabled=config.use_amp):
                logits = model(images)
                loss = criterion(logits, labels) / config.gradient_accumulation_steps

            scaler.scale(loss).backward()

            if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(all_params, config.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            train_loss += loss.item() * config.gradient_accumulation_steps

            if batch_idx % 10 == 0:
                logger.info(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")

        model.backbone.eval()
        model.classifier.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                logits = model(images)
                preds = torch.argmax(logits, dim=-1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())

        qwk = quadratic_weighted_kappa(all_labels, all_preds)
        current_lr = optimizer.param_groups[-1]["lr"]

        logger.info(
            f"Epoch {epoch}: Train Loss: {train_loss/len(train_loader):.4f}, "
            f"LR: {current_lr:.2e}, QWK: {qwk:.4f}"
        )

        save_checkpoint(
            path=str(output_dir / f"epoch_{epoch}.pth"),
            model_state_dict=model.backbone.state_dict(),
            optimizer_state_dict=optimizer.state_dict(),
            scheduler_state_dict=scheduler.state_dict(),
            epoch=epoch,
            best_metric=qwk,
            extra={"classifier_state_dict": model.classifier.state_dict()},
        )

        if qwk > best_qwk:
            best_qwk = qwk
            patience_counter = 0
            torch.save(model.backbone.state_dict(), output_dir / "source_model.pth")
            torch.save(model.classifier.state_dict(), output_dir / "source_classifier.pth")
            logger.info(f"Saved best model with QWK: {qwk:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch} (no QWK improvement for {patience} epochs)")
                break

    for p in output_dir.glob("epoch_*.pth"):
        p.unlink(missing_ok=True)

    logger.info(f"Training completed. Best QWK: {best_qwk:.4f}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        config = load_config(sys.argv[1])
    else:
        config = load_config("configs/default.yaml")
    train_source(config)
