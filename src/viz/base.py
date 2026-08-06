import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, Optional, Tuple


def get_domain_data(results: Dict[str, Any]) -> Tuple[Optional[Dict], Optional[list], Optional[list]]:
    dr = results.get("domain_result")
    if dr is None:
        return None, [], []
    return dr, dr.get("adaptation_steps", []), dr.get("batch_metrics", [])


def plot_qwk_bar(results: Dict[str, Any], plots_dir: Path, method_name: str = "") -> None:
    label_suffix = f" ({method_name})" if method_name else ""
    labels = ["Baseline\n(RETFound)", "Post-adapt"]
    baseline = results["baseline_metrics"]["qwk"]
    final = results["final_metrics"]["qwk"]
    values = [baseline, final]
    colors = ["steelblue", "darkred" if final != baseline else "lightcoral"]
    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(labels, values, color=colors, width=0.5, edgecolor="gray", linewidth=0.5)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f"{val:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_ylabel("QWK Score")
    ax.set_title(f"QWK: Baseline vs Post-Adaptation{label_suffix}")
    ax.set_ylim(0, 1)
    ax.grid(True, axis="y", alpha=0.3)
    gap = final - baseline
    ax.text(0.98, 0.02, f"Change: {gap:+.4f}", transform=ax.transAxes, ha="right", va="bottom", fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.7))
    plt.tight_layout()
    plt.savefig(plots_dir / "qwk_comparison.png", dpi=150, bbox_inches="tight")
    plt.show()


def plot_per_class_accuracy(results: Dict[str, Any], dr: Optional[Dict], plots_dir: Path, method_name: str = "") -> None:
    if dr is None:
        return
    label_suffix = f" ({method_name})" if method_name else ""
    classes = ["0 (No DR)", "1 (Mild)", "2 (Moderate)", "3 (Severe)", "4 (Prolif.)"]
    x = np.arange(len(classes))
    width = 0.3
    fig, ax = plt.subplots(figsize=(9, 5))
    baseline_acc = [results["baseline_metrics"]["per_class_accuracy"][str(i)] * 100 for i in range(5)]
    post_acc = [results["final_metrics"]["per_class_accuracy"][str(i)] * 100 for i in range(5)]
    ax.bar(x - width / 2, baseline_acc, width, label="Baseline", color="coral", edgecolor="gray")
    ax.bar(x + width / 2, post_acc, width, label="Post-adapt", color="lightcoral", edgecolor="gray")
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"Per-Class Accuracy{label_suffix}")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    for i in range(5):
        for vals, offset in [(baseline_acc, -width / 2), (post_acc, width / 2)]:
            ax.text(x[i] + offset, vals[i] + 1, f"{vals[i]:.1f}%", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig(plots_dir / "per_class_accuracy.png", dpi=150, bbox_inches="tight")
    plt.show()


def plot_per_batch_accuracy(dr: Optional[Dict], _unused: Optional[Dict], plots_dir: Path, method_name: str = "") -> None:
    if dr is None:
        return
    label_suffix = f" ({method_name})" if method_name else ""
    fig, ax = plt.subplots(figsize=(12, 4.5))
    batch_ids = [bm["batch_idx"] for bm in dr["batch_metrics"]]
    accs = [bm["accuracy"] * 100 for bm in dr["batch_metrics"]]
    adapt_batches = {s["batch_idx"] for s in dr.get("adaptation_steps", [])}
    bar_colors = ["coral" if bi in adapt_batches else "lightgray" for bi in batch_ids]
    ax.bar(batch_ids, accs, color=bar_colors, edgecolor="gray", linewidth=0.5, width=0.7)
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


def plot_adaptation_metrics(adaptation_steps: list, plots_dir: Path) -> None:
    if not adaptation_steps:
        return
    batch_indices = [s["batch_idx"] for s in adaptation_steps]
    mean_probs = [s["metrics"]["mean_max_prob"] for s in adaptation_steps]
    conf_samples = [s["metrics"].get("num_confident_samples", s["metrics"].get("confident_samples", 0)) for s in adaptation_steps]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    ax.bar(batch_indices, mean_probs, color="teal", edgecolor="gray", linewidth=0.5, width=0.7)
    ax.set_xlabel("Batch Index")
    ax.set_ylabel("Mean Max Softmax Probability")
    ax.set_title(f"Confidence Level per Batch (range: {min(mean_probs):.4f} - {max(mean_probs):.4f})")
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


def plot_confidence_distribution(adaptation_steps: list, plots_dir: Path, method_name: str = "") -> None:
    if not adaptation_steps:
        return
    label_suffix = f" ({method_name})" if method_name else ""
    mean_probs = [s["metrics"]["mean_max_prob"] for s in adaptation_steps]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(mean_probs, bins=10, color="teal", edgecolor="black", alpha=0.7, rwidth=0.9)
    ax.axvline(x=np.mean(mean_probs), color="red", linestyle="--", linewidth=1.5, label=f"Mean = {np.mean(mean_probs):.4f}")
    ax.set_xlabel("Mean Max Probability")
    ax.set_ylabel("Frequency (batches)")
    ax.set_title(f"Distribution of Model Confidence Across Batches{label_suffix}")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / "confidence_distribution.png", dpi=150, bbox_inches="tight")
    plt.show()


def plot_entropy_vs_accuracy(adaptation_steps: list, batch_metrics: list, plots_dir: Path, method_name: str = "") -> None:
    if not adaptation_steps or not batch_metrics:
        return
    label_suffix = f" ({method_name})" if method_name else ""
    batch_entropies = []
    batch_accs = []
    batch_indices_plot = []
    for bm in batch_metrics:
        bi = bm["batch_idx"]
        for s in adaptation_steps:
            if s["batch_idx"] == bi:
                entropy = s.get("signals", {}).get("ema_entropy")
                if entropy is None:
                    entropy = s.get("metrics", {}).get("mean_max_prob", 0)
                batch_entropies.append(entropy)
                batch_accs.append(bm["accuracy"] * 100)
                batch_indices_plot.append(bi)
                break
    if not batch_entropies:
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(batch_entropies, batch_accs, c=batch_indices_plot, cmap="viridis", s=80, edgecolor="black", linewidth=0.5, zorder=3)
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Batch Index")
    ax.set_xlabel("EMA Entropy / Confidence")
    ax.set_ylabel("Batch Accuracy (%)")
    ax.set_title(f"Entropy vs Accuracy (colored by batch){label_suffix}")
    ax.grid(True, alpha=0.3)
    for ent, acc, bi in zip(batch_entropies, batch_accs, batch_indices_plot):
        if acc < 15 or acc > 45 or bi in [0, 9, 10, 20]:
            ax.annotate(str(bi), (ent, acc), textcoords="offset points", xytext=(5, 5), fontsize=8, alpha=0.8)
    plt.tight_layout()
    plt.savefig(plots_dir / "entropy_vs_accuracy.png", dpi=150, bbox_inches="tight")
    plt.show()


def print_summary_table(results: Dict[str, Any], method_name: str = "") -> None:
    label = f" CTTA Results{ ' (' + method_name + ')' if method_name else ''} ".center(65, "=")
    print(label)
    print(f"Baseline QWK:       {results['baseline_metrics']['qwk']:.4f}")
    print(f"Post-adapt QWK:     {results['final_metrics']['qwk']:.4f}")
    dr = results.get("domain_result", {})
    print(f"Adaptations:        {dr.get('num_adaptations', 0)}/{dr.get('total_batches', 0)}")
    print(f"Adaptation rate:    {results.get('adaptation_rate', 0):.2%}")
    gap = results['final_metrics']['qwk'] - results['baseline_metrics']['qwk']
    print(f"QWK change:         {gap:+.4f}")
    if dr.get("batch_metrics"):
        mean_acc = np.mean([bm["accuracy"] for bm in dr["batch_metrics"]]) * 100
        print(f"Mean batch acc:     {mean_acc:.1f}%")
    print("=" * 65)
