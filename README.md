# Medical CTTA Framework

A Python research framework for Continual Test-Time Adaptation (CTTA) of medical foundation models on retinal fundus image analysis (5-class diabetic retinopathy grading).

## Overview

This framework adapts pretrained foundation models (RETFound, VisionFM) to new medical imaging domains without access to source data or labeled targets. It supports **single-domain adaptation** (CoTTA, PALM, ViDA) and **sequential multi-domain adaptation** to measure catastrophic forgetting.

### Key Features

- **RETFound** foundation model wrapper (ViT-Large, DINOv2 checkpoint) and **VisionFM** (ViT-Base, fundus checkpoint)
- **CoTTA**, **PALM**, and **ViDA** adapters (every-batch adaptation; shift-detection gating was removed)
- **Sequential CTTA protocol**: e.g. IDRiD → APTOS → IDRiD (no weight reset between domains, teacher reset per target)
- **Evaluation metrics** (QWK, per-class accuracy, forgetting metric)
- **Kaggle integration** via VS Code Remote Tunnels for GPU training
- **kagglehub** integration for dataset downloads

## Project Structure

```
medical_CTTA/
├── configs/              # YAML configuration files
│   ├── default.yaml      # Global experiment defaults
│   ├── data/             # Dataset configs (idrid, aptos2019, messidor2)
│   ├── model/            # Model configs (retfound, visionfm)
│   ├── {cotta,palm,vida}/  # Flat method-param fragments
│   └── method/           # Composed experiment configs (extends + !include)
├── src/                  # Python source modules
│   ├── config.py         # Pydantic config with extends/!include support
│   ├── env.py            # Kaggle vs local auto-detect (notebook bootstrap)
│   ├── data/             # Dataset loaders + download utilities
│   ├── models/           # Foundation model wrappers
│   ├── adapters/         # CTTA adapters (CoTTA, PALM, ViDA)
│   ├── evaluation/       # Metrics and runner package (single + sequential)
│   ├── utils/            # Logging, seeding, checkpointing
│   └── viz/              # Plotting helpers
├── notebooks/            # Jupyter notebooks for experiments
│   ├── setup/ baselines/ cotta/ palm/ vida/ sequential/
├── scripts/              # run_cotta.py, run_vida.py, kaggle_sync.py
├── tests/                # Unit tests
├── KAGGLE_SETUP.md       # Kaggle VS Code Remote Tunnels guide
├── kaggle.yml            # Kaggle kernel config
├── setup.py              # Package installation
└── requirements.txt      # Pinned dependencies
```

## Protocols

### Single-domain CTTA

Each experiment runs on one dataset:
1. Load pretrained weights (RETFound from HuggingFace, VisionFM from Google Drive via gdown)
2. Replace the classifier head with class prototypes from the train split (scaled by `prototype.temperature`)
3. Evaluate baseline (no adaptation)
4. Run the CTTA method (CoTTA, PALM, or ViDA) on the full test stream, adapting every batch
5. Evaluate post-adaptation

### Sequential multi-domain CTTA

For measuring catastrophic forgetting across domains:

```
Source → Target 1 (adapt) → All previous (test) → Target 2 (adapt) → All previous (test) → ...
```

1. Load pretrained weights once
2. Adapt to the source domain itself, snapshot the teacher
3. For each target domain in order:
   a. Adapt to the target (no weight reset between domains)
   b. Evaluate on ALL previously seen domains in TEST mode
4. Results show cumulative adaptation and forgetting effects

Example configs: `configs/method/sequential/cotta/cotta_idrid_aptos.yaml`, `configs/method/sequential/vida/idrid_aptos.yaml`, plus 3-domain Messidor-2 sequences in both methods.

## Installation

### Local Development

1. Activate conda environment:
   ```bash
   conda activate deeplearning
   ```

2. Install package in editable mode:
   ```bash
   pip install -e .
   ```

### Kaggle

See `KAGGLE_SETUP.md` for connecting local VS Code to a Kaggle GPU runtime via VS Code Remote Tunnels (no SSH keys needed). Notebooks auto-detect the platform via `src.env.init()`.

## Usage

### Download Datasets

```python
from src.data.download import DatasetDownloader

downloader = DatasetDownloader(data_dir="./data")
downloader.download_all()
```

### Running a Single-Domain Experiment

```python
from src.config import load_config
from src.evaluation.runner import CTTARunner

config = load_config("configs/method/cotta/idrid.yaml")
runner = CTTARunner(config)
results = runner.run()
```

### Running Sequential CTTA

```python
from src.config import load_config
from src.evaluation.runner import SequentialCTTARunner

config = load_config("configs/method/sequential/cotta/cotta_idrid_aptos.yaml")
runner = SequentialCTTARunner(config)
results = runner.run()
```

Alternatively, run experiments from notebooks (`notebooks/cotta/`, `notebooks/palm/`, `notebooks/vida/`, `notebooks/sequential/`) — each notebook loads fresh weights and writes `ctta_results.json` into its own output directory. Results include baseline QWK, per-domain adaptation results, and forgetting metrics.

## Configuration

All experiments are configured via YAML files. Config loading is non-standard:
- `extends: <path>` deep-merges a base config recursively
- `!include <path>` inlines another YAML file (resolved relative to the including file)
- `load_config(path)` returns a pydantic `ExperimentConfig`

Fragments (`configs/{cotta,palm,vida}/*.yaml`) hold method-specific parameters and are composed into experiment configs under `configs/method/{method}/` (cross-domain variants use `*_target_from_*.yaml` naming).

## Datasets

- **IDRiD**: Indian Diabetic Retinopathy Image Dataset
  - 516 images, single camera, single clinic; typical source domain
- **APTOS 2019**: APTOS Blindness Detection
  - ~5,590 images, multiple sites, heterogeneous; typical target domain
- **Messidor-2**: 1,748 images, French population, Topcon camera, non-mydriatic
  - No predefined split: the loader filters ungradable images (`adjudicated_gradable=0`) and splits via `train_ratio`

## Methods

- **CoTTA**: teacher-student framework with stochastic restoration and confidence-gated augmentation
- **PALM**: layer selection by gradient magnitude with per-parameter adaptive learning rates (percentile mode by default; fixed-threshold mode per original paper)
- **ViDA**: dual low-rank/high-rank adapter injection with uncertainty-based gating (HKA)

All methods adapt on every batch; no adaptation is gated on shift detection.

## Models

- **RETFound**: ViT-Large from HuggingFace `YukunZhou/RETFound_dinov2_meh`
  - 1024-dim, DINOv2 checkpoint; weights are **gated** — set the `HF_TOKEN` environment variable
  - Loaded with `pretrained=False` + `hf_hub_download`
- **VisionFM**: ViT-Base/16 (768-dim) fine-tuned on 3.4M fundus images
  - Weights download from Google Drive via gdown to `./cache/visionfm/`

## License

This project is for research purposes only.