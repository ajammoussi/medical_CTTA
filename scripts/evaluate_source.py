import torch
from torch.utils.data import DataLoader
from pathlib import Path
import logging
from src.config import ExperimentConfig, load_config
from src.models.registry import ModelRegistry
from src.data.registry import DatasetRegistry
from src.evaluation.metrics import quadratic_weighted_kappa, per_class_accuracy


def evaluate_source(config: ExperimentConfig) -> dict:
    """Evaluate source model on source test set."""
    logger = logging.getLogger(__name__)
    logger.info("Starting source model evaluation")

    # Set seeds
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    # Device
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")

    # Load model
    model_class = ModelRegistry.get(config.model.name)
    model = model_class(
        num_classes=config.model.num_classes,
        freeze_layers=config.model.freeze_layers,
        checkpoint=config.model.checkpoint
    )
    model.load_weights()

    # Load source checkpoint
    output_dir = Path(config.output_dir)
    source_model_path = output_dir / "source_model.pth"
    source_classifier_path = output_dir / "source_classifier.pth"

    if source_model_path.exists():
        model.backbone.load_state_dict(torch.load(source_model_path, map_location=device))
        model.classifier.load_state_dict(torch.load(source_classifier_path, map_location=device))
        logger.info("Loaded source checkpoint")
    else:
        logger.warning("No source checkpoint found, using pretrained weights")

    model = model.to(device)

    # Load dataset
    dataset_class = DatasetRegistry.get(config.source_dataset.name)
    test_dataset = dataset_class(
        data_dir=config.source_dataset.data_dir,
        image_size=config.source_dataset.image_size,
        train=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.ctta.batch_size,
        shuffle=False,
        num_workers=config.source_dataset.num_workers
    )

    # Evaluate
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

    # Compute metrics
    qwk = quadratic_weighted_kappa(all_labels, all_preds)
    per_class = per_class_accuracy(all_labels, all_preds, model.get_num_classes())

    results = {
        "qwk": qwk,
        "per_class_accuracy": per_class,
        "num_samples": len(all_labels)
    }

    logger.info(f"Source evaluation QWK: {qwk:.4f}")
    logger.info(f"Per-class accuracy: {per_class}")

    return results


if __name__ == "__main__":
    import sys
    from pathlib import Path
    if len(sys.argv) > 1:
        config = load_config(sys.argv[1])
    else:
        config = load_config("configs/default.yaml")
    evaluate_source(config)