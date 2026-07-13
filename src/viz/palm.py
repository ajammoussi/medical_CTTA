import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def plot_parameter_selection(aptos_steps: list, plots_dir: Path) -> None:
    if not aptos_steps:
        return
    batch_indices = [s["batch_idx"] for s in aptos_steps]
    selected_fracs = [s["metrics"].get("selected_fraction", 0) for s in aptos_steps]
    n_params = [s["metrics"].get("n_selected_params", 0) for s in aptos_steps]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    ax.plot(batch_indices, selected_fracs, "b-o", markersize=5, linewidth=1.5)
    ax.axhline(y=np.mean(selected_fracs), color="red", linestyle="--", alpha=0.7, label=f"Mean: {np.mean(selected_fracs):.3f}")
    ax.set_xlabel("Batch Index")
    ax.set_ylabel("Selected Fraction")
    ax.set_title("PALM: Fraction of Parameters Selected per Batch")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax = axes[1]
    ax.bar(batch_indices, n_params, color="purple", edgecolor="gray", linewidth=0.5, width=0.7)
    ax.set_xlabel("Batch Index")
    ax.set_ylabel("Number of Selected Parameters")
    ax.set_title("PALM: Number of Parameters Selected")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / "palm_parameter_selection.png", dpi=150, bbox_inches="tight")
    plt.show()


def plot_kl_loss_and_entropy(aptos_steps: list, plots_dir: Path) -> None:
    if not aptos_steps:
        return
    batch_indices = [s["batch_idx"] for s in aptos_steps]
    kl_losses = [s["metrics"].get("kl_loss", 0) for s in aptos_steps]
    post_entropies = [s["metrics"].get("post_adapt_entropy", 0) for s in aptos_steps]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    ax.plot(batch_indices, kl_losses, "s-", color="darkorange", markersize=5, linewidth=1.5)
    ax.axhline(y=np.mean(kl_losses), color="red", linestyle="--", alpha=0.7, label=f"Mean: {np.mean(kl_losses):.4f}")
    ax.set_xlabel("Batch Index")
    ax.set_ylabel("KL Loss")
    ax.set_title("PALM: KL Divergence Loss (Consistency)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax = axes[1]
    ax.plot(batch_indices, post_entropies, "D-", color="teal", markersize=5, linewidth=1.5)
    ax.axhline(y=np.mean(post_entropies), color="red", linestyle="--", alpha=0.7, label=f"Mean: {np.mean(post_entropies):.4f}")
    ax.set_xlabel("Batch Index")
    ax.set_ylabel("Post-Adapt Entropy")
    ax.set_title("PALM: Model Entropy After Adaptation")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / "palm_losses_and_entropy.png", dpi=150, bbox_inches="tight")
    plt.show()
