# Medical CTTA Framework

A Python research framework for Continual Test-Time Adaptation (CTTA) of medical foundation models on retinal fundus image analysis.

## Overview

This framework implements CTTA methods for adapting pretrained foundation models (RETFound) to new medical imaging domains without access to source data or labeled targets. It supports **sequential multi-domain adaptation** to measure catastrophic forgetting.

### Key Features

- **RETFound** foundation model wrapper (ViT-Large pretrained on retinal images)
- **CoTTA** adapter (teacher-student framework with stochastic parameter restoration)
- **Sequential CTTA protocol**: IDRiD → APTOS → IDRiD (no weight reset between domains)
- **Shift detection** module (entropy, embedding drift, distribution drift signals)
- **Evaluation metrics** (QWK, per-class accuracy, forgetting metric)
- **Kaggle integration** via VS Code Remote Tunnels for GPU training
- **Kaggle API** integration for dataset downloads

## Project Structure

```
medical_CTTA/
├── configs/              # YAML configuration files
├── src/                  # Python source modules
│   ├── config.py         # Pydantic config with sequential CTTA support
│   ├── data/             # Dataset loaders + download utilities
│   ├── models/           # Foundation model wrappers
│   ├── adapters/         # CTTA adapters (CoTTA, ViDA, etc.)
│   ├── shift_detection/  # Shift detection signals
│   ├── evaluation/       # Metrics and runner
│   └── utils/            # Logging, seeding, checkpointing
├── notebooks/            # Jupyter notebooks for experiments
├── scripts/              # Training and evaluation scripts
│   ├── train_source.py   # Train source model
│   ├── run_cotta.py      # Run CoTTA adaptation
│   ├── evaluate_source.py# Evaluate source model
│   └── kaggle_sync.py    # Bundle code as Kaggle Dataset for code sync
├── tests/                # Unit tests
├── KAGGLE_SETUP.md       # Kaggle VS Code Remote Tunnels guide
├── kernel-metadata.json  # Kaggle kernel metadata
├── kaggle.yml            # Kaggle kernel config
├── setup.py              # Package installation
└── requirements.txt      # Pinned dependencies
```

## Sequential CTTA Protocol

The framework implements a sequential multi-domain CTTA protocol:

```
Source (IDRiD) → Target 1 (APTOS) → Target 2 (IDRiD)
     ↓                ↓                    ↓
  Train model    Adapt online          Measure forgetting
  on source      (no weight reset)     (return to source)
```

This protocol measures:
- **Domain adaptation**: How well the model adapts to APTOS
- **Catastrophic forgetting**: Performance drop when returning to IDRiD
- **Robustness**: Model stability across domain transitions

## Installation

### Local Development

1. Activate conda environment:
   ```bash
   conda activate deeplearning
   ```

2. Install package in editable mode:
   ```bash
   cd medical_CTTA
   pip install -e .
   ```

### Kaggle

See `KAGGLE_SETUP.md` for detailed instructions on:
- Connecting local VS Code to Kaggle runtime via VS Code Remote Tunnels
- No SSH keys or third-party tunneling services needed
- Syncing code via git clone or VS Code's direct file editing

## Usage

### Download Datasets

```python
from src.data.download import DatasetDownloader

downloader = DatasetDownloader(data_dir="./data")
downloader.download_all()
```

### Training Source Model

```python
from src.config import load_config
from src.evaluation.runner import CTTARunner

config = load_config("configs/default.yaml")
runner = CTTARunner(config)
model = runner._load_model()
runner._train_source_domain(model, config.sequential_ctta.source_domain)
```

### Running Sequential CTTA

```python
from src.config import load_config
from src.evaluation.runner import CTTARunner

config = load_config("configs/method/cotta.yaml")
runner = CTTARunner(config)
results = runner.run()

# Results include:
# - Source baseline QWK
# - Per-domain adaptation results
# - Forgetting metric
```

### Using Notebooks

- `00_setup_kaggle.ipynb` - Kaggle environment setup (VS Code Remote Tunnels)
- `01_explore_datasets.ipynb` - Dataset visualization
- `02_train_source.ipynb` - Train source model
- `03_baseline_eval.ipynb` - Baseline evaluation
- `04_1_run_cotta.ipynb` - Run CoTTA sequential CTTA
- `04_2_run_palm.ipynb` - Run PALM sequential CTTA
- `05_compare_adaptations.ipynb` - Compare and visualize results

## Configuration

All experiments are configured via YAML files in `configs/`. Key configuration options:

```yaml
# Sequential CTTA config
sequential_ctta:
  enabled: true
  source_domain:
    dataset:
      name: idrid
      data_dir: ./data/IDRiD/
    description: "Source domain: single camera, single clinic"
  target_domains:
    - dataset:
        name: aptos2019
        data_dir: ./data/APTOS2019/
      description: "Target domain 1: heterogeneous multi-site"
    - dataset:
        name: idrid
        data_dir: ./data/IDRiD/
      description: "Target domain 2: return to source (test forgetting)"
  reset_weights_between_domains: false  # Must be false for CTTA
```

## Datasets

- **IDRiD**: Indian Diabetic Retinopathy Image Dataset (source domain)
  - Kaggle: https://www.kaggle.com/datasets/aaryapatel98/indian-diabetic-retinopathy-image-dataset
  - 516 images, single camera, single clinic
  
- **APTOS 2019**: APTOS Blindness Detection (target domain)
  - Kaggle: https://www.kaggle.com/datasets/mariaherrerot/aptos2019
  - ~5,590 images, multiple sites, heterogeneous

## Methods

- **CoTTA**: Continual Test-Time Adaptation with teacher-student framework
- **PALM**: Probabilistic adaptation with per-tensor adaptive LR

## RETFound Version

The framework uses RETFound MAE (Nature 2023):
- Checkpoint: `YukunZhou/RETFound_mae_natureCFP`
- Pretrained on 1.6M retinal images from Moorfields Eye Hospital
- **Not pretrained on IDRiD or APTOS** (verified)

## License

This project is for research purposes only.