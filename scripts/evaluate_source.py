import torch
from torch.utils.data import DataLoader
from pathlib import Path
import logging
from src.config import ExperimentConfig, load_config
from src.models.registry import ModelRegistry
from src.data.registry import DatasetRegistry
from src.evaluation.metrics import quadratic_weighted_kappa, per_class_accuracy


def evaluate_baseline(config: ExperimentConfig) -> dict:
    logger = logging.getLogger(__name__)
    logger.info("Starting baseline evaluation (RETFound pretrained, no adaptation)")

    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    device = torch.device(config.device if torch.cuda.is_available() else "cpu")

    model_class = ModelRegistry.get(config.model.name)
    model = model_class(
        num_classes=config.model.num_classes,
        freeze_layers=config.model.freeze_layers,
        checkpoint=config.model.checkpoint
    )
    model.load_weights()
    model = model.to(device)

    dataset_class = DatasetRegistry.get(config.target_dataset.name)
    test_dataset = dataset_class(
        data_dir=config.target_dataset.data_dir,
        image_size=config.target_dataset.image_size,
        train=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.ctta_batch_size,
        shuffle=False,
        num_workers=config.target_dataset.num_workers
    )

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
    per_class = per_class_accuracy(all_labels, all_preds, model.get_num_classes())

    results = {
        "qwk": qwk,
        "per_class_accuracy": per_class,
        "num_samples": len(all_labels)
    }

    logger.info(f"Baseline QWK on {config.target_dataset.name}: {qwk:.4f}")
    logger.info(f"Per-class accuracy: {per_class}")

    return results


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        config = load_config(sys.argv[1])
    else:
        config = load_config("configs/method/cotta/idrid.yaml")
    evaluate_baseline(config)
