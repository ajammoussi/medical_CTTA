# AGENTS.md — Medical CTTA Framework

Research framework for Continual Test-Time Adaptation (CTTA) of medical foundation models (RETFound, VisionFM) on retinal fundus images (5-class DR grading). Two protocols: single-domain (CoTTA/PALM/ViDA/E-CoTTA/LCoTTA) and sequential multi-domain (forgetting measurement, no weight reset).

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
- Two-layer composition: `configs/{cotta,palm,vida,ecotta,lcotta}/*.yaml` = flat method-param fragments; `configs/method/{method}/*.yaml` = composed experiments that `extends` default.yaml and `!include` fragments
- Changing pydantic defaults in `src/config.py` can be silently overridden by YAML (e.g. `max_grad_norm`: pydantic default 1.0 vs YAML 5.0)
- All methods are **tuned per dataset × model** — fragments differ across idrid/aptos/messidor2 × retfound/visionfm. Don't copy one config to another target blindly.

## Models & weights

- **RETFound**: checkpoint `YukunZhou/RETFound_dinov2_meh` (NOT the older `RETFound_mae_natureCFP` still quoted in README; `retfound.py` auto-detects arch from repo name — `dinov2` → timm `vit_large_patch14_dinov2`). Weights are **gated: requires `HF_TOKEN` env var**. Loaded via `hf_hub_download` + `pretrained=False`, cached in CPU RAM.
- **VisionFM**: ViT-Base, weights via `gdown` to `./cache/visionfm/`; classifier uses CLS tokens of last 4 blocks (3072-dim input).
- Classifier head is **replaced at runtime**: `_initialize_classifier_from_prototypes()` runs the dataset train split through the backbone, sets head weights = L2-normalized per-class mean features × `prototype.temperature` (`src/evaluation/runner/base.py`). The random head is never used.
- **No adapted weights are ever saved**; every run starts fresh from pretrained weights.

## Protocols

- **Single-domain** (`src/evaluation/runner/single.py`): load model → prototype head → baseline eval → adapt every batch → post-adapt eval → save `output_dir/ctta_results.json`.
- **Sequential** (`src/evaluation/runner/sequential.py`): weights carry forward across the whole sequence (`reset_weights_before_each_target: false`); Phase 0 adapts to the **source** domain, then the post-source teacher snapshot is pushed into every target adapter via `set_teacher_weights()` (prevents target-biased teacher); after adapting each target, **all previously seen domains** are re-evaluated in TEST mode.

## Method quirks

- **CoTTA/PALM/ViDA/E-CoTTA/LCoTTA all adapt every batch** (`adapt_every_batch: true`). Shift detection was deliberately removed (signals were AI-generated, unreliable on overconfident medical models, and had an AND→OR bug — see `changes/2026-07-31_remove_shift_detection.md`). Don't re-add gating.
- **PALM layer selection** (`src/adapters/palm.py`): percentile mode selects **top** `selection_percentile` (default 0.30) layers by **mean-abs grad**, descending, plus mandatory head + last `always_blocks` blocks and a `min_selected_ratio` mass floor. Fixed-threshold mode (`layer_selection_threshold`) also selects **high-grad** layers. (Do not "fix" this to the original paper's low-grad selection — DR versions explicitly switched.)
- PALM has a `check_fallback` that reverts to source weights if QWK/acc drops >0.02 below baseline.
- **E-CoTTA** (`src/adapters/ecotta.py`): frozen encoder partitioned into K groups; per partition `out = out1 + mlp(x)` (mlp zero-init so it starts at **exact** identity); loss = confident-set entropy + `ecotta_reg_lambda`·Σ `|mlp(x.detach())|.mean()` (self-distilled reg = official `|f_M(x)-f_orig(x)|`, gradients reach only that partition's meta). Only the meta MLPs are trainable — collect `part.mlp.parameters()`, **never** `part.parameters()` (that un-freezes the frozen encoder: it was a real bug causing warmup to fine-tune the whole ViT → Kaggle OOM + warmup overfit). Frozen blocks run under **per-partition gradient checkpointing** wired from the global `gradient_checkpointing` flag; the backbone-level flag is forced off in setup to avoid timm's outer `checkpoint_seq` double-wrapping. Confident samples = entropy < `entropy_margin`·ln(C), per-class capped by `ecotta_per_class_cap`, with top-k quota fallback via `ecotta_min_confident_fraction`. Single forward per batch. Optional `adapter.warmup(train_loader)` (CE) runs after `adapter.setup` post-baseline-eval. VisionFM constraint: classifier taps last-4-blocks CLS via `get_intermediate_layers(x, n=4)` → partition scheme must keep those 4 blocks as separate single-block partitions (`ecotta_num_partitions: 7`, sizes `[2,2,4,1,1,1,1]` for VisionFM; RETFound uses K=4 → auto `[4,4,8,8]`).
- **LCoTTA** (`src/adapters/lcotta.py`): entropy minimization on norm-layer affine params (+ optional classifier head), updates projected onto an online-tracked principal gradient subspace (FIFO grad queue → SVD-PCA rank-r → `g̃ = Pᵀ(Pg)`). Kept from official (`ThunderDavid/LCoTTA`): exp(H0−H) weighting, projection math, queue skips batch 0, plain steps until the queue fills. Safety nets: grad clipping, per-class cap, `check_fallback` (>0.02 QWK/acc drop → revert to source). Sequential LCoTTA is **not** implemented.
  Hard-won DR tuning facts (all discovered empirically — do not regress):
  - DR test sets are **7–23 batches total**; the paper's ImageNet-C queue params (interval 50–100, k=100) never fill the queue there. Use interval=1 and size k/r to the dataset (idrid k=6/r=4; aptos/messidor k=14–16/r=8).
  - SGD on LN affine at small lr with ONE step per batch is a no-op → use `lcotta_optimizer: adam`; entropy gradients are tiny and Adam normalizes step scales. Corollary: an L2 penalty inside the loss is also ~a no-op under Adam — constraints must be applied post-step (see head anchor below).
  - Overconfident models sit entirely below the entropy margin (~zero entropy → vanishing gradient). The quota (`lcotta_min_confident_fraction`) is therefore enforced EVERY batch (margin set ∪ class-balanced top-up), not just as empty-mask fallback. Cosine-redundancy filter stays OFF on DR (`lcotta_cosine_filter: false`).
  - **Class-2 erosion**: entropy minimization drains borderline moderate-DR predictions into adjacent grades — QWK rises while grade-2 recall collapses. Prior-alignment (APU) can't fix it (marginals are already near-max entropy ⇒ zero gradient). What works: `lcotta_head_anchor_weight` — post-step pull of head weights toward their source prototype values (`p ← lerp(p, p0, λ)`), bounding boundary sharpening so gains come from LN feature adaptation. Head multiplier ×10 gives peak QWK but maximum erosion; ×2–5 + anchor trades some QWK for per-class safety. There is an inherent tension: QWK forgives adjacent-grade errors, so max-QWK and all-per-class-positive cannot both be maximized under entropy minimization.
  - Rare-grade baselines sit extreme (e.g. cls4 = 100% on few samples): 1–2 flipped images exceed the >0.02 `check_fallback` tolerance → gentler lr/head AND richer quota for such datasets.

## Sequential config names — gotchas

- CoTTA 2-domain files are prefixed: `cotta_idrid_aptos.yaml` (not `idrid_aptos.yaml`). Vida 2-domain files are unprefixed.
- CoTTA VisionFM idrid→aptos sequential is the misleadingly named `aptos_visionfm.yaml`.
- 3-domain (messidor) configs exist for both methods in `configs/method/sequential/{cotta,vida}/`.
- **Stale notebook config refs**: `notebooks/sequential/cotta/2_adaptations/cotta_idrid_aptos*.ipynb` load non-existent config paths (`idrid_aptos.yaml`, `idrid_aptos_visionfm.yaml`) and will raise `FileNotFoundError` if run unmodified.
- `configs/method/vida.yaml` is a legacy stub (TODO comment) — harmless, not loaded.
- No PALM messidor2 config/notebook; no `scripts/run_palm.py`; no script runs the sequential runner (notebooks are the interface).
- E-CoTTA and LCoTTA each have 6 single-domain notebooks in `notebooks/ecotta/` and `notebooks/lcotta/` loading `configs/method/{ecotta,lcotta}/*.yaml`; sequential E-CoTTA/LCoTTA are **not** implemented. Regenerate via `scripts/gen_ecotta_notebooks.py` / `scripts/gen_lcotta_notebooks.py` (overwrites notebook outputs).

## Notebooks

- Bootstrap: `from src.env import init; init()` auto-detects Kaggle vs local (`src/env.py`). Kaggle: writes `.pth` into site-packages, symlinks `data/` → `/kaggle/working/data/`.
- Notebooks are isolated: fresh weights, own runner, own output dir; `torch.cuda.empty_cache()` in first cell avoids CUDA bleed in shared kernels.
- Each loads config → builds `CTTARunner`/`SequentialCTTARunner` → `runner.run()`.
- Real experiment runs happen on Kaggle GPU (weights/datasets live there); local runs need `HF_TOKEN` + datasets under `data/`.

## Datasets

- IDRiD (source), APTOS2019 (target), Messidor-2 (extra). Loaders do multi-pass filename discovery with a permissive CSV/images fallback (`src/data/`).
- Messidor-2 has **no predefined split**: loader filters `adjudicated_gradable==0` and splits via `train_ratio` (stratified).
- Datasets download via `kagglehub` (incl. `/kaggle/input/...` fallback) — run on Kaggle GPU via `notebooks/setup/setup_kaggle.ipynb` (VS Code Remote Tunnel); see `KAGGLE_SETUP.md`.
- Test sets are tiny (IDRiD ≈100 images; rare grades ≈10 samples): metric deltas under ±2% are within a few image flips — treat them as noise, and remember `check_fallback` tolerance (0.02) is at this noise level.
