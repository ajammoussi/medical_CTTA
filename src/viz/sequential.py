"""Visualization functions for sequential multi-domain CTTA results."""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, Optional


def load_standalone_results(json_path: str) -> Optional[Dict[str, Any]]:
    """Load standalone (single-domain) CTTA results for reference comparison."""
    path = Path(json_path)
    if not path.exists():
        return None
    with open(path) as f:
        data = json.load(f)
    fm = data.get("final_metrics", data.get("domain_result", {}).get("post_adaptation_metrics", {}))
    bm = data.get("baseline_metrics", {})
    return {
        "qwk": fm.get("qwk", 0),
        "overall_accuracy": fm.get("overall_accuracy", 0),
        "per_class_accuracy": fm.get("per_class_accuracy", {}),
        "baseline_qwk": bm.get("qwk", 0),
        "baseline_accuracy": bm.get("overall_accuracy", 0),
    }


def plot_sequential_qwk_comparison(
    results: Dict[str, Any],
    plots_dir: Path,
    method_name: str = "",
    standalone_references: Optional[Dict[str, Dict[str, Any]]] = None,
) -> None:
    """Plot QWK comparison across all domains at each stage of the sequential protocol.

    Shows baseline, post-adaptation, and test-only evaluations for each domain.
    If standalone_references is provided (e.g. {"retfound_dinov2": {...}}), horizontal
    reference lines are drawn on the target domain subplot.
    """
    label_suffix = f" ({method_name})" if method_name else ""

    source_domain = results["source_domain"]
    baseline_qwk = results["baseline_metrics"][source_domain]["qwk"]
    baseline_acc = results["baseline_metrics"][source_domain]["overall_accuracy"]

    target_domains = results["target_domains"]
    has_source_adapt = source_domain in results["domain_results"]

    n_cols = 1 + (1 if has_source_adapt else 0) + len(target_domains)

    fig, axes_flat = plt.subplots(1, n_cols, figsize=(6 * n_cols, 5), sharey=True)
    if n_cols == 1:
        axes_flat = [axes_flat]
    col = 0

    ax = axes_flat[col]
    ax.bar(["Baseline"], [baseline_qwk], color="steelblue", width=0.5, edgecolor="gray")
    ax.set_ylabel("QWK Score")
    ax.set_title(f"{source_domain}\n(Baseline)\nAcc: {baseline_acc*100:.1f}%")
    ax.set_ylim(0, 1)
    ax.grid(True, axis="y", alpha=0.3)
    ax.text(0, baseline_qwk + 0.02, f"{baseline_qwk:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    col += 1

    if has_source_adapt:
        ax = axes_flat[col]
        sd = results["domain_results"][source_domain]
        post_source_qwk = sd["test_evaluations"][source_domain]["qwk"]
        post_source_acc = sd["test_evaluations"][source_domain]["overall_accuracy"]
        ax.bar([source_domain], [post_source_qwk], color="darkred", width=0.5, edgecolor="gray")
        ax.set_title(f"{source_domain}\n(After source adaptation)\nAcc: {post_source_acc*100:.1f}%")
        ax.set_ylim(0, 1)
        ax.grid(True, axis="y", alpha=0.3)
        ax.text(0, post_source_qwk + 0.02, f"{post_source_qwk:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

        diff_qwk = post_source_qwk - baseline_qwk  # positive = improvement
        ax.text(0.98, 0.02, f"Δ: {diff_qwk:+.4f}", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.7))
        col += 1

    for target_name in target_domains:
        ax = axes_flat[col]
        target_data = results["domain_results"][target_name]

        test_evals = target_data["test_evaluations"]

        domains_to_plot = list(test_evals.keys())
        qwk_values = [test_evals[d]["qwk"] for d in domains_to_plot]
        acc_values = [test_evals[d]["overall_accuracy"] * 100 for d in domains_to_plot]

        colors = []
        for d in domains_to_plot:
            if d == target_name:
                colors.append("darkred")
            elif d == source_domain:
                colors.append("coral")
            else:
                colors.append("lightcoral")

        bars = ax.bar(domains_to_plot, qwk_values, color=colors, width=0.5, edgecolor="gray")
        for bar, val, acc in zip(bars, qwk_values, acc_values):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02, f"{val:.4f}",
                    ha="center", va="bottom", fontsize=9, fontweight="bold")
            ax.text(bar.get_x() + bar.get_width() / 2, 0.03, f"Acc: {acc:.1f}%",
                    ha="center", va="bottom", fontsize=7, color="dimgray")

        # Add standalone reference lines
        if standalone_references and target_name in results.get("target_domains", []):
            for ref_name, ref_data in standalone_references.items():
                if ref_data:
                    ref_qwk = ref_data.get("qwk", 0)
                    ax.axhline(y=ref_qwk, linestyle=":", alpha=0.6, linewidth=1.2,
                               label=f"{ref_name}: {ref_qwk:.4f}")
            if standalone_references:
                ax.legend(fontsize=7, loc="lower right")

        title_acc = f"Acc: {test_evals.get(target_name, {}).get('overall_accuracy', 0)*100:.1f}%"
        ax.set_title(f"After adapting to {target_name}\n{title_acc}")
        ax.set_ylim(0, 1)
        ax.grid(True, axis="y", alpha=0.3)

        baseline_for_domain = results["baseline_metrics"].get(source_domain, {}).get("qwk", 0)
        if source_domain in test_evals:
            forgetting = test_evals[source_domain]["qwk"] - baseline_for_domain  # positive = improvement
            ax.text(0.98, 0.02, f"Δ on source: {forgetting:+.4f}", transform=ax.transAxes,
                    ha="right", va="bottom", fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.7))
        col += 1

    plt.suptitle(f"Sequential CTTA: QWK Across Domains{label_suffix}", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(plots_dir / "sequential_qwk_comparison.png", dpi=150, bbox_inches="tight")
    plt.show()


def plot_sequential_overall_accuracy(
    results: Dict[str, Any],
    plots_dir: Path,
    method_name: str = "",
    standalone_references: Optional[Dict[str, Dict[str, Any]]] = None,
) -> None:
    """Plot overall accuracy comparison across all domains at each stage."""
    label_suffix = f" ({method_name})" if method_name else ""

    source_domain = results["source_domain"]
    baseline_acc = results["baseline_metrics"][source_domain]["overall_accuracy"]

    target_domains = results["target_domains"]
    has_source_adapt = source_domain in results["domain_results"]

    n_cols = 1 + (1 if has_source_adapt else 0) + len(target_domains)
    fig, axes_flat = plt.subplots(1, n_cols, figsize=(6 * n_cols, 5), sharey=True)
    if n_cols == 1:
        axes_flat = [axes_flat]
    col = 0

    ax = axes_flat[col]
    ax.bar(["Baseline"], [baseline_acc * 100], color="steelblue", width=0.5, edgecolor="gray")
    ax.set_ylabel("Overall Accuracy (%)")
    ax.set_title(f"{source_domain}\n(Baseline)")
    ax.set_ylim(0, 100)
    ax.grid(True, axis="y", alpha=0.3)
    ax.text(0, baseline_acc * 100 + 2, f"{baseline_acc*100:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
    col += 1

    if has_source_adapt:
        ax = axes_flat[col]
        sd = results["domain_results"][source_domain]
        post_acc = sd["test_evaluations"][source_domain]["overall_accuracy"]
        ax.bar([source_domain], [post_acc * 100], color="darkred", width=0.5, edgecolor="gray")
        ax.set_title(f"{source_domain}\n(After source adaptation)")
        ax.set_ylim(0, 100)
        ax.grid(True, axis="y", alpha=0.3)
        ax.text(0, post_acc * 100 + 2, f"{post_acc*100:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
        delta = baseline_acc - post_acc
        ax.text(0.98, 0.02, f"Delta: {delta*100:+.2f}pp", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.7))
        col += 1

    for target_name in target_domains:
        ax = axes_flat[col]
        target_data = results["domain_results"][target_name]
        test_evals = target_data["test_evaluations"]
        domains_to_plot = list(test_evals.keys())
        acc_values = [test_evals[d]["overall_accuracy"] * 100 for d in domains_to_plot]

        colors = ["darkred" if d == target_name else "coral" if d == source_domain else "lightcoral" for d in domains_to_plot]
        bars = ax.bar(domains_to_plot, acc_values, color=colors, width=0.5, edgecolor="gray")
        for bar, val in zip(bars, acc_values):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 2, f"{val:.1f}%",
                    ha="center", va="bottom", fontsize=9, fontweight="bold")

        if standalone_references and target_name in target_domains:
            for ref_name, ref_data in standalone_references.items():
                if ref_data:
                    ref_acc = ref_data.get("overall_accuracy", 0) * 100
                    ax.axhline(y=ref_acc, linestyle=":", alpha=0.6, linewidth=1.2,
                               label=f"{ref_name}: {ref_acc:.1f}%")
            if standalone_references:
                ax.legend(fontsize=7, loc="lower right")

        ax.set_title(f"After adapting to {target_name}")
        ax.set_ylim(0, 100)
        ax.grid(True, axis="y", alpha=0.3)

        if source_domain in test_evals:
            delta = baseline_acc - test_evals[source_domain]["overall_accuracy"]
            ax.text(0.98, 0.02, f"Delta on source: {delta*100:+.2f}pp", transform=ax.transAxes,
                    ha="right", va="bottom", fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.7))
        col += 1

    plt.suptitle(f"Sequential CTTA: Overall Accuracy{label_suffix}", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(plots_dir / "sequential_overall_accuracy.png", dpi=150, bbox_inches="tight")
    plt.show()


def plot_sequential_forgetting(results: Dict[str, Any], plots_dir: Path, method_name: str = "") -> None:
    """Plot forgetting metrics for each domain after each adaptation step."""
    label_suffix = f" ({method_name})" if method_name else ""

    source_domain = results["source_domain"]
    baseline_qwk = results["baseline_metrics"][source_domain]["qwk"]

    target_domains = results["target_domains"]

    fig, ax = plt.subplots(figsize=(10, 6))

    stages = [f"Baseline\n({source_domain})"]
    qwk_on_source = [baseline_qwk]

    source_data = results["domain_results"].get(source_domain)
    if source_data and "adaptation" in source_data:
        post_source_qwk = source_data["test_evaluations"][source_domain]["qwk"]
        qwk_on_source.append(post_source_qwk)
        stages.append(f"After {source_domain}\nadaptation")

    for target_name in target_domains:
        target_data = results["domain_results"][target_name]
        if source_domain in target_data["test_evaluations"]:
            qwk_on_source.append(target_data["test_evaluations"][source_domain]["qwk"])
            stages.append(f"After {target_name}\nadaptation")

    ax.plot(range(len(stages)), qwk_on_source, "o-", color="steelblue", linewidth=2, markersize=10, label=f"QWK on {source_domain}")
    ax.axhline(y=baseline_qwk, color="gray", linestyle="--", alpha=0.5, label="Baseline")

    for i, (stage, qwk) in enumerate(zip(stages, qwk_on_source)):
        ax.annotate(f"{qwk:.4f}", (i, qwk), textcoords="offset points", xytext=(0, 12),
                    ha="center", fontsize=10, fontweight="bold")

    ax.set_xticks(range(len(stages)))
    ax.set_xticklabels(stages)
    ax.set_ylabel("QWK Score")
    ax.set_title(f"Forgetting Curve: QWK on {source_domain}{label_suffix}")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(plots_dir / "sequential_forgetting.png", dpi=150, bbox_inches="tight")
    plt.show()


def plot_sequential_adaptation_comparison(results: Dict[str, Any], plots_dir: Path, method_name: str = "") -> None:
    """Plot comparison of adaptation metrics across all target domains."""
    label_suffix = f" ({method_name})" if method_name else ""

    source_domain = results["source_domain"]
    target_domains = results["target_domains"]
    all_domains = [source_domain] + target_domains
    n_domains = len(all_domains)

    fig, axes = plt.subplots(2, n_domains, figsize=(7 * n_domains, 10), squeeze=False)

    for i, domain_name in enumerate(all_domains):
        domain_data = results["domain_results"][domain_name]
        adapt_data = domain_data["adaptation"]

        batch_metrics = adapt_data.get("batch_metrics", [])
        adaptation_steps = adapt_data.get("adaptation_steps", [])

        ax_acc = axes[0, i]
        if batch_metrics:
            batch_ids = [bm["batch_idx"] for bm in batch_metrics]
            accs = [bm["accuracy"] * 100 for bm in batch_metrics]
            adapt_batches = {s["batch_idx"] for s in adaptation_steps}
            bar_colors = ["coral" if bi in adapt_batches else "lightgray" for bi in batch_ids]
            ax_acc.bar(batch_ids, accs, color=bar_colors, edgecolor="gray", linewidth=0.5, width=0.7)
            ax_acc.axhline(y=np.mean(accs), color="green", linestyle=":", linewidth=1, alpha=0.7,
                          label=f"Mean: {np.mean(accs):.1f}%")
            ax_acc.set_xlabel("Batch Index")
            ax_acc.set_ylabel("Accuracy (%)")
            ax_acc.set_title(f"{domain_name}: Per-Batch Accuracy")
            ax_acc.legend(fontsize=8)
            ax_acc.grid(True, axis="y", alpha=0.3)
            ax_acc.set_xticks(batch_ids)

        ax_conf = axes[1, i]
        if adaptation_steps:
            mean_probs = [s["metrics"]["mean_max_prob"] for s in adaptation_steps]
            conf_samples = [s["metrics"].get("num_confident_samples", 0) for s in adaptation_steps]

            ax_conf.bar(range(len(mean_probs)), mean_probs, color="teal", edgecolor="gray", linewidth=0.5)
            ax_conf.axvline(x=np.mean(mean_probs), color="red", linestyle="--", linewidth=1.5,
                           label=f"Mean = {np.mean(mean_probs):.4f}")
            ax_conf.set_xlabel("Batch Index")
            ax_conf.set_ylabel("Mean Max Probability")
            ax_conf.set_title(f"{domain_name}: Confidence per Batch")
            ax_conf.legend(fontsize=8)
            ax_conf.grid(True, axis="y", alpha=0.3)

    plt.suptitle(f"Sequential CTTA: Adaptation Metrics{label_suffix}", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(plots_dir / "sequential_adaptation_comparison.png", dpi=150, bbox_inches="tight")
    plt.show()


def plot_sequential_per_class_accuracy(
    results: Dict[str, Any],
    plots_dir: Path,
    method_name: str = "",
    standalone_references: Optional[Dict[str, Dict[str, Any]]] = None,
) -> None:
    """Plot per-class accuracy comparison across domains and stages.

    For the source domain, shows three bars: baseline, post-source-adaptation,
    and test-on-source after target adaptation.  Includes overall accuracy in titles.
    """
    label_suffix = f" ({method_name})" if method_name else ""

    classes = ["0 (No DR)", "1 (Mild)", "2 (Moderate)", "3 (Severe)", "4 (Prolif.)"]
    source_domain = results["source_domain"]

    all_domains = [source_domain] + results["target_domains"]
    n_domains = len(all_domains)

    def _per_class_arr(pca_dict):
        keys = list(pca_dict.keys())
        use_str = len(keys) > 0 and isinstance(keys[0], str)
        return [pca_dict[str(c) if use_str else c] * 100 for c in range(5)]

    baseline_acc = _per_class_arr(results["baseline_metrics"][source_domain]["per_class_accuracy"])
    baseline_total_acc = results["baseline_metrics"][source_domain]["overall_accuracy"] * 100

    source_data = results["domain_results"].get(source_domain)
    if source_data and "adaptation" in source_data:
        post_source_acc = _per_class_arr(source_data["test_evaluations"][source_domain]["per_class_accuracy"])
        post_source_total_acc = source_data["test_evaluations"][source_domain]["overall_accuracy"] * 100
    else:
        post_source_acc = baseline_acc
        post_source_total_acc = baseline_total_acc

    # Check if any target has test_evaluations on the source domain
    post_target_test_on_source = None
    post_target_test_on_source_total_acc = None
    last_target_name = results["target_domains"][-1] if results["target_domains"] else None
    if last_target_name:
        td = results["domain_results"].get(last_target_name, {})
        te = td.get("test_evaluations", {})
        if source_domain in te:
            post_target_test_on_source = _per_class_arr(te[source_domain]["per_class_accuracy"])
            post_target_test_on_source_total_acc = te[source_domain]["overall_accuracy"] * 100

    show_third_bar = post_target_test_on_source is not None
    bar_width = 0.25 if show_third_bar else 0.35

    fig_width = 5 * n_domains + (2 if show_third_bar else 0)
    fig, axes = plt.subplots(1, n_domains, figsize=(fig_width, 5), sharey=True, squeeze=False)

    for i, domain_name in enumerate(all_domains):
        ax = axes[0, i]

        if domain_name == source_domain:
            post_acc = post_source_acc
        else:
            target_data = results["domain_results"][domain_name]
            post_acc = _per_class_arr(target_data["test_evaluations"][domain_name]["per_class_accuracy"])

        x = np.arange(len(classes))

        if domain_name == source_domain and show_third_bar:
            ax.bar(x - bar_width, baseline_acc, bar_width, label="Baseline", color="coral", edgecolor="gray")
            ax.bar(x, post_acc, bar_width, label="Post-source-adapt", color="lightcoral", edgecolor="gray")
            ax.bar(x + bar_width, post_target_test_on_source, bar_width,
                   label=f"After {last_target_name}\n(test on source)", color="mediumseagreen", edgecolor="gray")
            title = f"{source_domain}\nBase: {baseline_total_acc:.1f}% → Adapt: {post_source_total_acc:.1f}% → Test: {post_target_test_on_source_total_acc:.1f}%"
        else:
            width = 0.35
            ax.bar(x - width / 2, baseline_acc, width, label="Baseline", color="coral", edgecolor="gray")
            ax.bar(x + width / 2, post_acc, width, label="Post-adapt", color="lightcoral", edgecolor="gray")
            total_acc = results["domain_results"][domain_name]["test_evaluations"][domain_name]["overall_accuracy"] * 100
            title = f"{domain_name}\nBase: {baseline_total_acc:.1f}% → Adapt: {total_acc:.1f}%"

        ax.set_xticks(x)
        ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Accuracy (%)" if i == 0 else "")
        ax.set_title(title, fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(True, axis="y", alpha=0.3)
        ax.set_ylim(0, 105)

    plt.suptitle(f"Per-Class Accuracy Comparison{label_suffix}", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(plots_dir / "sequential_per_class_accuracy.png", dpi=150, bbox_inches="tight")
    plt.show()


def print_sequential_summary(
    results: Dict[str, Any],
    method_name: str = "",
    standalone_references: Optional[Dict[str, Dict[str, Any]]] = None,
) -> None:
    """Print a summary table of sequential CTTA results.

    If standalone_references is provided, prints side-by-side comparison
    with standalone (single-domain) runs.
    """
    label = f" Sequential CTTA Results{ ' (' + method_name + ')' if method_name else ''} ".center(70, "=")
    print(label)

    source_domain = results["source_domain"]
    baseline_qwk = results["baseline_metrics"][source_domain]["qwk"]
    baseline_acc = results["baseline_metrics"][source_domain]["overall_accuracy"]
    print(f"\nSource domain: {source_domain}")
    print(f"  Baseline QWK:           {baseline_qwk:.4f}")
    print(f"  Baseline Overall Acc:   {baseline_acc*100:.2f}%")

    print(f"\nTarget domains: {', '.join(results['target_domains'])}")

    source_data = results["domain_results"].get(source_domain)
    if source_data and "adaptation" in source_data:
        adapt_qwk = source_data["adaptation"]["post_adaptation_metrics"]["qwk"]
        adapt_acc = source_data["adaptation"]["post_adaptation_metrics"]["overall_accuracy"]
        n_adapt = source_data["adaptation"]["num_adaptations"]
        total_batches = source_data["adaptation"]["total_batches"]
        print(f"\n{'-' * 70}")
        print(f"After adapting to source {source_domain} ({n_adapt}/{total_batches} batches):")
        print(f"  QWK:                  {adapt_qwk:.4f}  (delta: {adapt_qwk - baseline_qwk:+.4f})")
        print(f"  Overall Acc:          {adapt_acc*100:.2f}% (delta: {(adapt_acc - baseline_acc)*100:+.2f}pp)")

    for target_name in results["target_domains"]:
        target_data = results["domain_results"][target_name]
        adapt_qwk = target_data["adaptation"]["post_adaptation_metrics"]["qwk"]
        adapt_acc = target_data["adaptation"]["post_adaptation_metrics"]["overall_accuracy"]
        n_adapt = target_data["adaptation"]["num_adaptations"]
        total_batches = target_data["adaptation"]["total_batches"]

        print(f"\n{'-' * 70}")
        print(f"After adapting to {target_name} ({n_adapt}/{total_batches} batches):")
        print(f"  Post-adaptation QWK on {target_name}: {adapt_qwk:.4f}")
        print(f"  Post-adaptation Acc on {target_name}: {adapt_acc*100:.2f}%")

        for test_name, test_metrics in target_data["test_evaluations"].items():
            if test_name != target_name:
                baseline_for_test = results["baseline_metrics"].get(test_name, {}).get("qwk", 0)
                baseline_acc_test = results["baseline_metrics"].get(test_name, {}).get("overall_accuracy", 0)
                forgetting_qwk = test_metrics["qwk"] - baseline_for_test  # positive = improvement
                forgetting_acc = test_metrics.get("overall_accuracy", 0) - baseline_acc_test
                print(f"  Test on {test_name}:")
                print(f"    QWK: {test_metrics['qwk']:.4f}  (delta vs baseline: {forgetting_qwk:+.4f})")
                print(f"    Acc: {test_metrics['overall_accuracy']*100:.2f}% (delta vs baseline: {forgetting_acc*100:+.2f}pp)")

            # Standalone reference comparison for the target domain itself
            if test_name == target_name and standalone_references:
                print(f"\n  -- Standalone comparison for {target_name} --")
                for ref_name, ref_data in standalone_references.items():
                    if ref_data is None:
                        continue
                    ref_qwk = ref_data.get("qwk", 0)
                    ref_acc = ref_data.get("overall_accuracy", 0)
                    ref_baseline_qwk = ref_data.get("baseline_qwk", 0)
                    ref_baseline_acc = ref_data.get("baseline_accuracy", 0)
                    print(f"    {ref_name}:")
                    print(f"      Baseline QWK:     {ref_baseline_qwk:.4f}  ->  Post-adapt QWK: {ref_qwk:.4f}")
                    print(f"      Baseline Acc:     {ref_baseline_acc*100:.2f}%  ->  Post-adapt Acc: {ref_acc*100:.2f}%")

    print("\n" + "=" * 70)
