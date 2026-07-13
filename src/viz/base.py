import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, Optional, Tuple


def get_domain_data(results: Dict[str, Any]) -> Tuple[Optional[Dict], Optional[Dict], Optional[list], Optional[list]]:
    aptos_dr = None
    idrid_dr = None
    for dr in results.get("domain_results", []):
        if dr["domain_name"] == "aptos2019":
            aptos_dr = dr
        if dr["domain_name"] == "idrid":
            idrid_dr = dr
    aptos_steps = (aptos_dr or {}).get("adaptation_steps", [])
    aptos_batches = (aptos_dr or {}).get("batch_metrics", [])
    return aptos_dr, idrid_dr, aptos_steps, aptos_batches


def plot_training_metrics(config, plots_dir: Path) -> None:
    train_metrics_path = Path(config.output_dir) / "source" / "train_metrics.jsonl"
    if not train_metrics_path.exists():
        print("No training metrics found.")
        return
    df = pd.read_json(train_metrics_path, lines=True)
    epochs = df["epoch"] + 1
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    axes[0].plot(epochs, df["loss"], "b-o", markersize=4, linewidth=1.5)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-Entropy Loss")
    axes[0].set_title("Training Loss")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(epochs, df["val_qwk"], "r-o", markersize=4, linewidth=1.5, label="Val QWK")
    axes[1].plot(epochs, df["best_qwk"], "g--", linewidth=1.5, label="Best QWK")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("QWK")
    axes[1].set_title("Validation QWK")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)
    axes[2].plot(epochs, df["lr"] * 1e5, "m-o", markersize=4, linewidth=1.5)
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("LR x 10^5")
    axes[2].set_title("Learning Rate Schedule")
    axes[2].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / "training_metrics_detailed.png", dpi=150, bbox_inches="tight")
    plt.show()


def plot_qwk_bar(results: Dict[str, Any], plots_dir: Path, method_name: str = "") -> None:
    if results.get("mode") != "sequential":
        return
    label_suffix = f" ({method_name})" if method_name else ""
    labels = ["Source\n(IDRiD)"]
    values = [results["source_baseline"]["qwk"]]
    colors = ["steelblue"]
    for dr in results["domain_results"]:
        labels.append(f"{dr['domain_name']}\n(before)")
        values.append(dr["baseline_metrics"]["qwk"])
        colors.append("coral")
        labels.append(f"{dr['domain_name']}\n(after)")
        values.append(dr["post_adaptation_metrics"]["qwk"])
        colors.append("darkred" if values[-1] != values[-2] else "lightcoral")
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(labels, values, color=colors, width=0.6, edgecolor="gray", linewidth=0.5)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f"{val:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.axhline(y=results["source_baseline"]["qwk"], color="steelblue", linestyle="--", linewidth=1, alpha=0.6, label=f"Source QWK = {results['source_baseline']['qwk']:.4f}")
    ax.set_ylabel("QWK Score")
    ax.set_title(f"QWK Across Domains{label_suffix}")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    ar = results.get("adaptation_rate", 0)
    ax.text(0.98, 0.02, f"Adaptation rate: {ar:.1%}", transform=ax.transAxes, ha="right", va="bottom", fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.7))
    plt.tight_layout()
    plt.savefig(plots_dir / "qwk_across_domains.png", dpi=150, bbox_inches="tight")
    plt.show()


def plot_forgetting(results: Dict[str, Any], plots_dir: Path, method_name: str = "") -> None:
    if "forgetting" not in results:
        return
    label_suffix = f" ({method_name})" if method_name else ""
    plt.figure(figsize=(6, 4))
    color_f = "red" if results["forgetting"] > 0 else "green"
    plt.bar(["Forgetting"], [results["forgetting"]], color=color_f)
    plt.ylabel("QWK Drop")
    plt.title(f"Catastrophic Forgetting{label_suffix}: {results['forgetting']:.4f}")
    plt.axhline(y=0, color="gray", linestyle="--")
    plt.tight_layout()
    plt.savefig(plots_dir / "forgetting.png", dpi=150, bbox_inches="tight")
    plt.show()


def plot_per_class_accuracy(results: Dict[str, Any], aptos_dr: Optional[Dict], plots_dir: Path, method_name: str = "") -> None:
    if not aptos_dr:
        return
    label_suffix = f" ({method_name})" if method_name else ""
    classes = ["0 (No DR)", "1 (Mild)", "2 (Moderate)", "3 (Severe)", "4 (Prolif.)"]
    x = np.arange(len(classes))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 5))
    sb = results["source_baseline"]["per_class_accuracy"]
    src_acc = [sb[str(i)] * 100 for i in range(5)]
    ax.bar(x - width, src_acc, width, label="Source: IDRiD", color="steelblue", edgecolor="gray")
    baseline_acc = [aptos_dr["baseline_metrics"]["per_class_accuracy"][str(i)] * 100 for i in range(5)]
    ax.bar(x, baseline_acc, width, label="Target: APTOS (baseline)", color="coral", edgecolor="gray")
    post_acc = [aptos_dr["post_adaptation_metrics"]["per_class_accuracy"][str(i)] * 100 for i in range(5)]
    ax.bar(x + width, post_acc, width, label="Target: APTOS (post-adapt)", color="lightcoral", edgecolor="gray")
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"Per-Class Accuracy: Source vs Target{label_suffix}")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    for i in range(5):
        for vals, offset, color in [(src_acc, -width, "steelblue"), (baseline_acc, 0, "coral"), (post_acc, width, "lightcoral")]:
            ax.text(x[i] + offset, vals[i] + 1, f"{vals[i]:.1f}%", ha="center", va="bottom", fontsize=7, color=color)
    plt.tight_layout()
    plt.savefig(plots_dir / "per_class_accuracy.png", dpi=150, bbox_inches="tight")
    plt.show()


def plot_per_batch_accuracy(aptos_dr: Optional[Dict], idrid_dr: Optional[Dict], plots_dir: Path, method_name: str = "") -> None:
    label_suffix = f" ({method_name})" if method_name else ""
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    for ax_idx, dr in enumerate([aptos_dr, idrid_dr]):
        ax = axes[ax_idx]
        if dr is None:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            continue
        batch_ids = [bm["batch_idx"] for bm in dr["batch_metrics"]]
        accs = [bm["accuracy"] * 100 for bm in dr["batch_metrics"]]
        adapt_batches = {s["batch_idx"] for s in dr.get("adaptation_steps", [])}
        bar_colors = ["coral" if bi in adapt_batches else "lightgray" for bi in batch_ids]
        ax.bar(batch_ids, accs, color=bar_colors, edgecolor="gray", linewidth=0.5, width=0.7)
        ax.axhline(y=20, color="red", linestyle="--", linewidth=1, alpha=0.6, label="Random (20%)")
        ax.axhline(y=np.mean(accs), color="green", linestyle=":", linewidth=1, alpha=0.7, label=f"Mean: {np.mean(accs):.1f}%")
        ax.set_xlabel("Batch Index")
        ax.set_ylabel("Accuracy (%)")
        ax.set_title(f"{dr['domain_name']}: Per-Batch Accuracy{label_suffix}\n(adaptations: {dr['num_adaptations']}/{dr['total_batches']})")
        ax.legend(fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)
        ax.set_xticks(batch_ids)
    plt.tight_layout()
    plt.savefig(plots_dir / "per_batch_accuracy.png", dpi=150, bbox_inches="tight")
    plt.show()


def plot_adaptation_metrics(aptos_steps: list, plots_dir: Path) -> None:
    if not aptos_steps:
        return
    batch_indices = [s["batch_idx"] for s in aptos_steps]
    mean_probs = [s["metrics"]["mean_max_prob"] for s in aptos_steps]
    conf_samples = [s["metrics"].get("num_confident_samples", s["metrics"].get("confident_samples", 0)) for s in aptos_steps]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    ax.bar(batch_indices, mean_probs, color="teal", edgecolor="gray", linewidth=0.5, width=0.7)
    ax.axhline(y=0.5, color="red", linestyle="--", linewidth=1.5, alpha=0.7, label="Old threshold = 0.5")
    ax.axhline(y=0.3, color="green", linestyle="--", linewidth=1.5, alpha=0.7, label="New threshold = 0.3")
    ax.axhline(y=0.2, color="gray", linestyle=":", linewidth=1, alpha=0.5, label="Random chance = 0.2")
    ax.fill_between(batch_indices, 0, mean_probs, alpha=0.1, color="teal")
    ax.set_xlabel("Batch Index")
    ax.set_ylabel("Mean Max Softmax Probability")
    ax.set_title(f"Confidence Level per Batch (range: {min(mean_probs):.4f} - {max(mean_probs):.4f})")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_xticks(batch_indices)
    ax = axes[1]
    colors_bar = ["coral" if c > 0 else "lightgray" for c in conf_samples]
    ax.bar(batch_indices, conf_samples, color=colors_bar, edgecolor="gray", linewidth=0.5, width=0.7)
    ax.set_xlabel("Batch Index")
    ax.set_ylabel("Number of Confident Samples")
    total = sum(conf_samples)
    ax.set_title(f"Confident Samples per Batch (total = {total} / {16 * len(batch_indices)})")
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_xticks(batch_indices)
    plt.tight_layout()
    plt.savefig(plots_dir / "adaptation_metrics.png", dpi=150, bbox_inches="tight")
    plt.show()


def plot_confidence_distribution(aptos_steps: list, plots_dir: Path, method_name: str = "") -> None:
    if not aptos_steps:
        return
    label_suffix = f" ({method_name})" if method_name else ""
    mean_probs = [s["metrics"]["mean_max_prob"] for s in aptos_steps]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(mean_probs, bins=10, range=(0.25, 0.50), color="teal", edgecolor="black", alpha=0.7, rwidth=0.9)
    ax.axvline(x=np.mean(mean_probs), color="red", linestyle="--", linewidth=1.5, label=f"Mean = {np.mean(mean_probs):.4f}")
    ax.axvline(x=0.5, color="darkred", linestyle=":", linewidth=2, alpha=0.7, label="Old threshold = 0.5")
    ax.axvline(x=0.3, color="green", linestyle=":", linewidth=2, label="New threshold = 0.3")
    ax.set_xlabel("Mean Max Probability")
    ax.set_ylabel("Frequency (batches)")
    ax.set_title(f"Distribution of Model Confidence Across Batches{label_suffix}")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / "confidence_distribution.png", dpi=150, bbox_inches="tight")
    plt.show()


def plot_entropy_vs_accuracy(aptos_steps: list, aptos_batches: list, plots_dir: Path, method_name: str = "") -> None:
    if not aptos_steps or not aptos_batches:
        return
    label_suffix = f" ({method_name})" if method_name else ""
    batch_entropies = []
    batch_accs = []
    batch_indices_plot = []
    for bm in aptos_batches:
        bi = bm["batch_idx"]
        for s in aptos_steps:
            if s["batch_idx"] == bi:
                batch_entropies.append(s["signals"]["ema_entropy"])
                batch_accs.append(bm["accuracy"] * 100)
                batch_indices_plot.append(bi)
                break
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(batch_entropies, batch_accs, c=batch_indices_plot, cmap="viridis", s=80, edgecolor="black", linewidth=0.5, zorder=3)
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Batch Index")
    ax.set_xlabel("EMA Entropy")
    ax.set_ylabel("Batch Accuracy (%)")
    ax.set_title(f"APTOS: Entropy vs Accuracy (colored by batch){label_suffix}")
    ax.grid(True, alpha=0.3)
    for ent, acc, bi in zip(batch_entropies, batch_accs, batch_indices_plot):
        if acc < 15 or acc > 45 or bi in [0, 9, 10, 20]:
            ax.annotate(str(bi), (ent, acc), textcoords="offset points", xytext=(5, 5), fontsize=8, alpha=0.8)
    plt.tight_layout()
    plt.savefig(plots_dir / "entropy_vs_accuracy.png", dpi=150, bbox_inches="tight")
    plt.show()


def print_summary_table(results: Dict[str, Any], method_name: str = "") -> None:
    if results.get("mode") != "sequential":
        return
    header = f" Sequential CTTA Results{ ' (' + method_name + ')' if method_name else ''} ".center(65, "=")
    print(header)
    print(f"Source baseline QWK: {results['source_baseline']['qwk']:.4f}")
    print()
    for dr in results["domain_results"]:
        print(f"--- {dr['domain_name']} ---")
        print(f"  Baseline QWK:       {dr['baseline_metrics']['qwk']:.4f}")
        print(f"  Post-adapt QWK:     {dr['post_adaptation_metrics']['qwk']:.4f}")
        print(f"  Adaptations:        {dr['num_adaptations']}/{dr['total_batches']}")
        print(f"  Mean batch acc:     {np.mean([bm['accuracy'] for bm in dr['batch_metrics']]) * 100:.1f}%")
        if dr.get("adaptation_steps"):
            steps_dr = dr["adaptation_steps"]
            total_conf = sum(s["metrics"].get("num_confident_samples", s["metrics"].get("confident_samples", 0)) for s in steps_dr)
            total_samples = 16 * len(steps_dr)
            print(f"  Conf samples:       {total_conf} / {total_samples}")
            print(f"  Total loss:         {sum(s['metrics']['loss'] for s in steps_dr):.4f}")
        print()
    print(f"Forgetting:           {results.get('forgetting', 'N/A')}")
    print(f"Adaptation rate:      {results.get('adaptation_rate', 0):.2%}")
    print("=" * 65)
