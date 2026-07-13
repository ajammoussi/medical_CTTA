# AGENTS.md — Medical CTTA Framework

## Commands

```bash
pip install -e .        # editable install (required, enables `from src.xxx`)
pytest tests/ -v        # all tests (no lint/typecheck configured)
pytest tests/test_X.py::test_Y -v   # single test
```

## Registration Pattern — Easy to Miss

Adding a new model/dataset/adapter requires **3 steps**:
1. Implement class with `@register_X("name")` decorator
2. Add import in `src/<submodule>/__init__.py` (e.g., `from src.models import retfound`)
3. If it's a new submodule, add a `__init__.py` with the trigger import

Forgetting step 2 is the most common bug — the implementation file exists and decorates, but the decorator never runs because nothing imports the file.

## Config: `extends` + `!include`

Config loading is non-standard:
- `configs/method/cotta.yaml` has `extends: ../default.yaml` — deep-merged recursively
- `!include <path>` YAML tag inlines another file (resolved relative to current YAML)
- `load_config("configs/method/palm.yaml")` returns a pydantic `ExperimentConfig`, not a dict
- If you change pydantic defaults in `src/config.py`, the YAML may override them

## Runner Source Model Loading

`_run_sequential()` loads the source model in this priority order:
1. Resume checkpoint (epoch checkpoint in `{output_dir}/source/`)
2. Local `{output_dir}/source/source_model.pth`
3. `shared_source_dir` config path (defaults to `./outputs/source/`)
4. Train from scratch

No adapted weights are ever saved. Each method always loads the original source.

## CTTA Protocol

Protocol: **IDRiD → APTOS → IDRiD** with `reset_weights_between_domains: false`.
The model carries state across domains. This is intentional (measures forgetting).
Do NOT change `reset_weights_between_domains` to true.

## PALM Runner Changes Are Gated

All runner changes for PALM are guarded by `method == "palm"`:
- Persists adapter across domains (single PALM instance, EMA carries over)
- Adapts on every batch (no shift detection gating), includes final partial batch
- CoTTA is **unaffected** — still uses shift detection, fresh adapter per domain

## Dataset Loader Fallback

Both `idrid.py` and `aptos2019.py` use a 3-step discovery:
1. Check known filename/directory lists directly in `data_dir`
2. `os.walk` subdirectories for those same names
3. **Fallback**: any `.csv` (labels) or any dir with `.jpg`/`.png` (images)

This handles Kaggle's unpredictable directory nesting.

## Notebook Bootstrap

Every experiment notebook starts with:
```python
from src.env import init
init()
```
Auto-detects Kaggle vs local. No platform-specific code elsewhere.

## Source Model

RETFound ViT-Large from HuggingFace: `YukunZhou/RETFound_mae_natureCFP`
- 1024-dim, no BatchNorm, not pretrained on IDRiD/APTOS
- Config: `configs/model/retfound.yaml`
