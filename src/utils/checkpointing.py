import torch
import json
import random
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional


def save_checkpoint(
    path: str,
    model_state_dict: Dict[str, Any],
    optimizer_state_dict: Optional[Dict[str, Any]] = None,
    scheduler_state_dict: Optional[Dict[str, Any]] = None,
    epoch: int = 0,
    best_metric: float = 0.0,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Save a comprehensive checkpoint with full state for resume."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": model_state_dict,
        "epoch": epoch,
        "best_metric": best_metric,
        "rng_state": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }
    if optimizer_state_dict is not None:
        checkpoint["optimizer_state_dict"] = optimizer_state_dict
    if scheduler_state_dict is not None:
        checkpoint["scheduler_state_dict"] = scheduler_state_dict
    if extra is not None:
        checkpoint.update(extra)
    torch.save(checkpoint, path)


def load_checkpoint(
    path: str,
    model: Optional[torch.nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    map_location: str = "cpu",
) -> Dict[str, Any]:
    """Load checkpoint and optionally restore model/optimizer/scheduler state."""
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    if model is not None and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    # Restore RNG state if available
    if "rng_state" in checkpoint:
        rng = checkpoint["rng_state"]
        if "python" in rng:
            random.setstate(rng["python"])
        if "numpy" in rng:
            np.random.set_state(rng["numpy"])
        if "torch" in rng:
            torch.set_rng_state(rng["torch"])
        if "torch_cuda" in rng and rng["torch_cuda"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rng["torch_cuda"])
    return checkpoint


def save_adaptation_checkpoint(
    path: str,
    model_state_dict: Dict[str, Any],
    adapter_state_dict: Optional[Dict[str, Any]] = None,
    domain_index: int = 0,
    batch_index: int = 0,
    shift_detector_state: Optional[Dict[str, Any]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Save an adaptation checkpoint for CTTA resume."""
    checkpoint = {
        "model_state_dict": model_state_dict,
        "domain_index": domain_index,
        "batch_index": batch_index,
        "adapter_state_dict": adapter_state_dict,
        "shift_detector_state": shift_detector_state,
        "rng_state": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }
    if extra is not None:
        checkpoint.update(extra)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)


def load_adaptation_checkpoint(
    path: str,
    model: Optional[torch.nn.Module] = None,
    adapter: Optional[Any] = None,
    map_location: str = "cpu",
) -> Dict[str, Any]:
    """Load adaptation checkpoint and restore model/adapter state."""
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    if model is not None and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    if adapter is not None and "adapter_state_dict" in checkpoint:
        adapter_state = checkpoint["adapter_state_dict"]
        if adapter_state is not None:
            adapter.load_state_dict(adapter_state)
    if "rng_state" in checkpoint:
        rng = checkpoint["rng_state"]
        if "python" in rng:
            random.setstate(rng["python"])
        if "numpy" in rng:
            np.random.set_state(rng["numpy"])
        if "torch" in rng:
            torch.set_rng_state(rng["torch"])
        if "torch_cuda" in rng and rng["torch_cuda"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rng["torch_cuda"])
    return checkpoint


def find_latest_checkpoint(checkpoint_dir: str, pattern: str = "*.pth") -> Optional[str]:
    """Find the latest checkpoint file matching pattern."""
    ckpt_dir = Path(checkpoint_dir)
    if not ckpt_dir.exists():
        return None
    checkpoints = sorted(ckpt_dir.glob(pattern))
    if not checkpoints:
        return None
    # Return the most recent by mtime
    latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
    return str(latest)


def find_latest_epoch_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """Find the rolling checkpoint for source training (not per-epoch)."""
    ckpt = Path(checkpoint_dir) / "checkpoint.pth"
    return str(ckpt) if ckpt.exists() else None


def find_latest_adaptation_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """Find the rolling adaptation checkpoint."""
    ckpt = Path(checkpoint_dir) / "adapt_checkpoint.pth"
    return str(ckpt) if ckpt.exists() else None


def save_results(results: Dict[str, Any], path: str) -> None:
    """Save results to JSON file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)


def save_train_metrics(path: str, epoch: int, loss: float, lr: float, val_qwk: float, best_qwk: float) -> None:
    """Append a single epoch's metrics as a JSON line (JSONL format for easy plotting)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    entry = {"epoch": epoch, "loss": round(float(loss), 4), "lr": float(lr),
             "val_qwk": round(float(val_qwk), 4), "best_qwk": round(float(best_qwk), 4)}
    with open(path, "a") as f:
        f.write(json.dumps(entry) + "\n")
