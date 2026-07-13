import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def plot_shift_detection_signals(aptos_steps: list, plots_dir: Path) -> None:
    if not aptos_steps:
        return
    batch_indices = [s["batch_idx"] for s in aptos_steps]
    entropy_diffs = [s["signals"]["entropy_diff"] for s in aptos_steps]
    drifts = [s["signals"]["drift"] for s in aptos_steps]
    jsds = [s["signals"]["jsd"] for s in aptos_steps]
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    axes[0].plot(batch_indices, entropy_diffs, "b-o", markersize=5, linewidth=1.5)
    axes[0].axhline(y=0.3, color="red", linestyle="--", alpha=0.7, label="Threshold = 0.3")
    axes[0].fill_between(batch_indices, 0.3, entropy_diffs, alpha=0.15, color="blue")
    axes[0].set_ylabel("Entropy Diff")
    axes[0].set_title("EMA Entropy Shift Signal")
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(batch_indices, drifts, "s-", color="purple", markersize=5, linewidth=1.5)
    axes[1].axhline(y=0.05, color="red", linestyle="--", alpha=0.7, label="Threshold = 0.05")
    axes[1].set_ylabel("Drift")
    axes[1].set_title("Embedding Drift Signal")
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)
    axes[2].plot(batch_indices, jsds, "D-", color="orange", markersize=5, linewidth=1.5)
    axes[2].axhline(y=0.05, color="red", linestyle="--", alpha=0.7, label="Threshold = 0.05")
    triggered = [(b, j) for b, j in zip(batch_indices, jsds) if j > 0.05]
    if triggered:
        axes[2].scatter([t[0] for t in triggered], [t[1] for t in triggered], color="red", s=80, zorder=5, label="Triggered")
    axes[2].set_ylabel("JSD")
    axes[2].set_xlabel("Batch Index")
    axes[2].set_title("Distribution Drift Signal (JSD)")
    axes[2].legend(fontsize=9)
    axes[2].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / "shift_detection_signals.png", dpi=150, bbox_inches="tight")
    plt.show()
