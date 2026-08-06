# AGENTS.md — Medical CTTA Framework

Research framework for Continual Test-Time Adaptation (CTTA) of medical foundation models (RETFound, VisionFM) on retinal fundus images (5-class DR grading). Two protocols: single-domain (CoTTA/PALM/ViDA) and sequential multi-domain (forgetting measurement, no weight reset).

## Commands

```bash
pip install -e .        # required: enables `from src.xxx` (no lint/typecheck configured)
pytest tests/ -v        # all tests
pytest tests/test_X.py::test_Y -v   # single test
```

## Registration pattern — easy to miss

New model/dataset/adapter needs **3 steps**:
1. Implement class with `@register_X("name")` decorator (`src/models/registry.py`, `src/data/registry.py`, `src/adapters/registry.py`)
2. Add trigger import in `src/<submodule>/__init__.py`
3. New submodule → create its `__init__.py`

Step 2 forgotten is the most common bug: the decorator never runs because nothing imports the file.

## Config system — non-standard

- `extends: <path>` deep-merges recursively; `!include <path>` YAML tag inlines another file (resolved **relative to the including YAML**)
- `load_config(path)` returns a pydantic `ExperimentConfig`, not a dict (`src/config.py`)
- Two-layer composition: `configs/{cotta,palm,vida}/*.yaml` = flat method-param fragments; `configs/method/{method}/*.yaml` = composed experiments that `extends` default.yaml and `!include` fragments
- Changing pydantic defaults in `src/config.py` can be silently overridden by YAML (e.g. `max_grad_norm`: pydantic default 1.0 vs YAML 5.0)

## Models & weights

- **RETFound**: checkpoint `YukunZhou/RETFound_dinov2_meh` (NOT the older `RETFound_mae_natureCFP` still quoted in README; `retfound.py` auto-detects arch from repo name — `dinov2` → timm `vit_large_patch14_dinov2`). Weights are **gated: requires `HF_TOKEN` env var**. Loaded via `hf_hub_download` + `pretrained=False`, cached in CPU RAM.
- **VisionFM**: ViT-Base, weights via `gdown` to `./cache/visionfm/`; classifier uses CLS tokens of last 4 blocks (3072-dim input).
- Classifier head is **replaced at runtime**: `_initialize_classifier_from_prototypes()` runs the dataset train split through the backbone, sets head weights = L2-normalized per-class mean features × `prototype.temperature` (`src/evaluation/runner/base.py`). The random head is never used.
- **No adapted weights are ever saved**; every run starts fresh from pretrained weights.

## Protocols

- **Single-domain** (`src/evaluation/runner/single.py`): load model → prototype head → baseline eval → adapt every batch → post-adapt eval → save `output_dir/ctta_results.json`.
- **Sequential** (`src/evaluation/runner/sequential.py`): weights carry forward across the whole sequence (`reset_weights_before_each_target: false`); Phase 0 adapts to the **source** domain, then the post-source teacher snapshot is pushed into every target adapter via `set_teacher_weights()` (prevents target-biased teacher); after adapting each target, **all previously seen domains** are re-evaluated in TEST mode.

## Method quirks

- **CoTTA/PALM/ViDA all adapt every batch** (`adapt_every_batch: true`). Shift detection was deliberately removed (signals were AI-generated, unreliable on overconfident medical models, and had an AND→OR bug — see `changes/2026-07-31_remove_shift_detection.md`). Don't re-add gating.
- **PALM layer selection** (`src/adapters/palm.py`): percentile mode selects **top** `selection_percentile` (default 0.30) layers by **mean-abs grad**, descending, plus mandatory head + last `always_blocks` blocks and a `min_selected_ratio` mass floor. Fixed-threshold mode (`layer_selection_threshold`) also selects **high-grad** layers. (Do not "fix" this to the original paper's low-grad selection — DR versions explicitly switched.)
- PALM has a `check_fallback` that reverts to source weights if QWK/acc drops >0.02 below baseline.

## Sequential config names — gotchas

- CoTTA 2-domain files are prefixed: `cotta_idrid_aptos.yaml` (not `idrid_aptos.yaml`). Vida 2-domain files are unprefixed.
- CoTTA VisionFM idrid→aptos sequential is the misleadingly named `aptos_visionfm.yaml`.
- 3-domain (messidor) configs exist for both methods in `configs/method/sequential/{cotta,vida}/`.
- **Stale notebook config refs**: `notebooks/sequential/cotta/2_adaptations/cotta_idrid_aptos*.ipynb` load non-existent config paths (`idrid_aptos.yaml`, `idrid_aptos_visionfm.yaml`) and will raise `FileNotFoundError` if run unmodified.
- `configs/method/vida.yaml` is a legacy stub (TODO comment) — harmless, not loaded.
- No PALM messidor2 config/notebook; no `scripts/run_palm.py`; no script runs the sequential runner (notebooks are the interface).

## Notebooks

- Bootstrap: `from src.env import init; init()` auto-detects Kaggle vs local (`src/env.py`). Kaggle: writes `.pth` into site-packages, symlinks `data/` → `/kaggle/working/data/`.
- Notebooks are isolated: fresh weights, own runner, own output dir; `torch.cuda.empty_cache()` in first cell avoids CUDA bleed in shared kernels.
- Each loads config → builds `CTTARunner`/`SequentialCTTARunner` → `runner.run()`.

## Datasets

- IDRiD (source), APTOS2019 (target), Messidor-2 (extra). Loaders do multi-pass filename discovery with a permissive CSV/images fallback (`src/data/`).
- Messidor-2 has **no predefined split**: loader filters `adjudicated_gradable==0` and splits via `train_ratio` (stratified).
- Datasets download via `kagglehub` (incl. `/kaggle/input/...` fallback) — run on Kaggle GPU via `notebooks/setup/setup_kaggle.ipynb` (VS Code Remote Tunnel); see `KAGGLE_SETUP.md`.