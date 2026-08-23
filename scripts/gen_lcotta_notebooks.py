"""Generate LCoTTA standalone notebooks mirroring the ecotta notebook structure."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "notebooks" / "lcotta"
OUT.mkdir(exist_ok=True)

# (config_name, dataset_label, title_line, desc_line)
EXPERIMENTS = [
    ("idrid", "IDRiD", "LCoTTA Adaptation on IDRiD (RETFound-DINOv2)",
     "LCoTTA: entropy minimization on norm layers with updates projected onto an online-tracked low-dimensional gradient subspace."),
    ("idrid_visionfm", "IDRiD", "LCoTTA Adaptation on IDRiD (VisionFM)",
     "LCoTTA: entropy minimization on norm layers with updates projected onto an online-tracked low-dimensional gradient subspace."),
    ("aptos", "APTOS2019", "LCoTTA Adaptation on APTOS2019 (RETFound-DINOv2)",
     "LCoTTA: entropy minimization on norm layers with updates projected onto an online-tracked low-dimensional gradient subspace."),
    ("aptos_visionfm", "APTOS2019", "LCoTTA Adaptation on APTOS2019 (VisionFM)",
     "LCoTTA: entropy minimization on norm layers with updates projected onto an online-tracked low-dimensional gradient subspace."),
    ("messidor2", "Messidor-2", "LCoTTA Adaptation on Messidor-2 (RETFound-DINOv2)",
     "LCoTTA: entropy minimization on norm layers with updates projected onto an online-tracked low-dimensional gradient subspace."),
    ("messidor2_visionfm", "Messidor-2", "LCoTTA Adaptation on Messidor-2 (VisionFM)",
     "LCoTTA: entropy minimization on norm layers with updates projected onto an online-tracked low-dimensional gradient subspace."),
]


def md(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source}


def code(source):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source}


def build(cfg_name, dataset_label, title, desc):
    config_path = f"configs/method/lcotta/{cfg_name}.yaml"
    cells = []

    cells.append(md([
        f"# {title}\n",
        f"{desc}"
    ]))

    cells.append(code([
        "import torch\n",
        "torch.cuda.empty_cache()"
    ]))

    cells.append(code([
        "import sys, os; sys.path.insert(0, os.getcwd())\n",
        "import os, sys\n",
        "_d = os.getcwd()\n",
        "PROJECT_ROOT = None\n",
        "while _d != os.path.dirname(_d):\n",
        "    if os.path.exists(os.path.join(_d, 'setup.py')):\n",
        "        PROJECT_ROOT = _d\n",
        "        break\n",
        "    _d = os.path.dirname(_d)\n",
        "if PROJECT_ROOT is None:\n",
        "    for _p in ['/kaggle/working/medical_CTTA', '/content/medical_CTTA']:\n",
        "        if os.path.isdir(_p):\n",
        "            PROJECT_ROOT = _p\n",
        "            break\n",
        "if PROJECT_ROOT and PROJECT_ROOT not in sys.path:\n",
        "    sys.path.insert(0, PROJECT_ROOT)\n",
        "os.chdir(PROJECT_ROOT)\n",
        "\n",
        "from src.env import init\n",
        "init()\n",
        "\n",
        "from src.config import load_config"
    ]))

    cells.append(code([
        f"config = load_config('{config_path}')\n",
        "\n",
        "print('Dataset:', config.target_dataset.name)\n",
        "print('Method:', config.ctta.method)\n",
        "print('Mode: every-batch (no shift detection gate)')\n",
        "print('Output:', config.output_dir)\n",
        "\n",
        "# RETFound weights are gated behind HuggingFace; require a token.\n",
        "if config.model.name == 'retfound' and not os.environ.get('HF_TOKEN'):\n",
        "    import getpass\n",
        "    token = getpass.getpass('Enter your HuggingFace token (hf_...): ')\n",
        "    os.environ['HF_TOKEN'] = token\n",
        "    print('HF_TOKEN set')\n",
        "elif config.model.name != 'retfound':\n",
        "    print('No HF token needed (VisionFM / Google Drive)')"
    ]))

    cells.append(code([
        "from src.evaluation.runner import CTTARunner\n",
        "\n",
        "runner = CTTARunner(config)\n",
        "results = runner.run()"
    ]))

    cells.append(code([
        "print('=== Results ===')\n",
        "print(f'Baseline QWK:        {results[\"baseline_metrics\"][\"qwk\"]:.4f}')\n",
        "print(f'Baseline Accuracy:   {results[\"baseline_metrics\"][\"overall_accuracy\"]*100:.2f}%')\n",
        "print(f'Post-adapt QWK:      {results[\"final_metrics\"][\"qwk\"]:.4f}')\n",
        "print(f'Post-adapt Accuracy: {results[\"final_metrics\"][\"overall_accuracy\"]*100:.2f}%')\n",
        "dr = results['domain_result']\n",
        "print(f'Adaptations:          {dr[\"num_adaptations\"]}/{dr[\"total_batches\"]}')\n",
        "print(f'Adaptation rate:      {results.get(\"adaptation_rate\", 0)*100:.2f}%')\n",
        "\n",
        "gap = results['final_metrics']['qwk'] - results['baseline_metrics']['qwk']\n",
        "print(f'QWK change:           {gap:+.4f}')\n",
        "acc_gap = results['final_metrics']['overall_accuracy'] - results['baseline_metrics']['overall_accuracy']\n",
        "print(f'Accuracy change:      {acc_gap*100:+.2f}%')"
    ]))

    cells.append(code([
        "from src.viz import (\n",
        "    plot_qwk_bar, plot_per_batch_accuracy,\n",
        "    plot_adaptation_metrics, plot_confidence_distribution,\n",
        "    plot_entropy_vs_accuracy, print_summary_table, plot_per_class_accuracy as plot_pca,\n",
        ")\n",
        "\n",
        "import json\n",
        "from pathlib import Path\n",
        "\n",
        "plots_dir = Path(config.output_dir) / 'plots'\n",
        "plots_dir.mkdir(exist_ok=True)\n",
        "\n",
        "with open(Path(config.output_dir) / 'ctta_results.json') as f:\n",
        "    results = json.load(f)\n",
        "\n",
        "dr = results['domain_result']\n",
        "method = results.get(\"method\", \"\")\n",
        "\n",
        "plot_qwk_bar(results, plots_dir, method_name=method)\n",
        "plot_pca(results, dr, plots_dir, method_name=method)\n",
        "plot_per_batch_accuracy(dr, None, plots_dir, method_name=method)\n",
        "plot_adaptation_metrics(dr['adaptation_steps'], plots_dir)\n",
        "plot_confidence_distribution(dr['adaptation_steps'], plots_dir, method_name=method)\n",
        "plot_entropy_vs_accuracy(dr['adaptation_steps'], dr['batch_metrics'], plots_dir, method_name=method)\n",
        "print_summary_table(results, method_name=method)"
    ]))

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (ipykernel)",
                "language": "python",
                "name": "python3"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }
    (OUT / f"{cfg_name}.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print(f"wrote notebooks/lcotta/{cfg_name}.ipynb")


for cfg_name, ds, title, desc in EXPERIMENTS:
    build(cfg_name, ds, title, desc)
print("done")
