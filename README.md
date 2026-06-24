# RF-DETR Optuna Training Manual

**Version 0.1.0** — see [CHANGELOG.md](meta/CHANGELOG.md) for release notes. The version is
defined in [`meta/version.py`](meta/version.py) and logged at the start of each run.

## Overview

An **Optuna-based hyperparameter optimization pipeline** for training RF-DETR object
detection models. Each Optuna *trial* augments the dataset, trains an RF-DETR model,
evaluates it on the validation and test splits, and returns a score. Optuna uses those
scores to search the configured hyperparameter space across many trials and track the
best configuration.

A single trial runs this pipeline (see `objective()` in [`rfdetr_train.py`](rfdetr_train.py)):

1. **Trial setup** – overlay this trial's Optuna suggestions onto a per-trial copy of the config
2. **Preprocessing** – augment the split dataset into a YOLO-format "final" dataset
3. **Training** – launch the RF-DETR worker subprocess on the prepared data
4. **Validation** – extract the best-epoch validation metrics
5. **Testing** – run inference + evaluation on the held-out test set
6. **Scoring** – combine validation/test metrics into the objective score
7. **Logging** – persist params + metrics to JSON/CSV and the Optuna store

---

## Project Structure

```
optuna_pipeline_rfdetr/
├── rfdetr_train.py              # MAIN entry point — run this
├── config.py                    # ALL configuration (the Config class)
├── rf-detr-nano.pth             # your pretrained RF-DETR weights
├── split_dataset/               # INPUT: train/valid/test splits (YOLO format, pre-augmentation)
│   ├── train/   (image1.tif + image1.txt together in one folder)
│   ├── valid/
│   └── test/
├── Final_dataset/               # OUTPUT: augmented dataset (auto-generated each trial)
├── runs/                        # per-trial training outputs
└── utils/
    ├── config_loader.py         # loads config.py -> TrainingConfig; Optuna search-space logic
    ├── preprocessing_utils.py   # image augmentation / upscaling
    ├── rf_detr_distributed_worker.py  # the subprocess that actually trains RF-DETR
    ├── rf_detr_extract_utils.py # parse metrics.csv from training output
    ├── rf_detr_prediction_utils.py    # run inference on the test set
    ├── yolo_evaluation_utils.py # evaluate predictions vs. ground truth
    ├── data_logging_utils.py    # write results to JSON/CSV
    └── optuna_utils.py          # Optuna study persistence
```

The dataset splits use `train` / `valid` / `test`, and each split holds the images
(`.tif`) and YOLO-format labels (`.txt`) **in the same folder**.

---

## Quick Start

### Prerequisites
- Python 3.10+ with PyTorch + CUDA
- `pip install -r requirements.txt`

### Run

```bash
python rfdetr_train.py
```

Run it **from the project root** (so `utils` is importable). All settings come from
[`config.py`](config.py) — there are no command-line arguments.

To accumulate more trials, just run it again: results append to the CSV/JSON and the
Optuna study resumes from `*_optuna_storage.json`.

---

## Configuration (`config.py`)

All configuration lives in a single `Config` class. Each **dict attribute** of `Config`
is a config *section*, exposed on the loaded config object as `config.<section>`:

| Section | Purpose |
|---|---|
| `study` | study name, class names, and `optimization_target_class` |
| `paths` | all file/dir paths, resolved against `config.py`'s own directory |
| `preprocessing_config` | input/training image sizes, label editing, and the brightness/contrast/sharpness augmentation values |
| `rfdetr_parameters` | parameters passed straight to RF-DETR `model.train()` (lr, epochs, loss coefs, `aug_config`, ...) |
| `rfdetr_dataset` | test threshold, IoU threshold, GPU count |
| `optuna` | `n_trials`, `n_jobs`, the `optimize` switch, and the `search_space` |

Notes:
- **Paths own their resolution.** `paths["root"]` is `config.py`'s directory; every other
  path is built from it. The loader takes paths as-is.
- **No hidden defaults.** Whatever you write in `config.py` is what the pipeline uses;
  a missing required section/key fails fast at load with a clear error.
- **`rfdetr_parameters`** is annotated with each parameter's RF-DETR default in a comment,
  so you can see at a glance where you deviate.
- **`aug_config`** is one of the RF-DETR augmentation presets, imported at the top of
  `config.py` (`AUG_CONSERVATIVE` / `AUG_AGGRESSIVE` / `AUG_AERIAL` / `AUG_INDUSTRIAL`).
  It's the actual preset object, not a string.

To change a *fixed* hyperparameter (not tuned), just edit its value in the relevant
section — e.g. set `"epochs": 50` in `rfdetr_parameters`.

---

## Tuning Hyperparameters with Optuna

Optuna tuning is controlled entirely by the `optuna` section.

### 1. Turn optimization on

```python
optuna = {
    "n_trials": 20,        # how many trials Optuna runs
    "n_jobs": 1,
    "optimize": True,      # <-- must be True, or the search space is ignored
    "search_space": { ... },
}
```

If `optimize` is `False`, every trial uses the fixed config values (no suggestions drawn).

### 2. Declare what to tune in `search_space`

The `search_space` **mirrors the config tree**. The top-level keys are section names
(they must match a `Config` section, e.g. `"rfdetr_parameters"`), and below that you
name the parameter(s) to tune. Each *leaf* is a **parameter spec** — a dict containing a
`"type"` key:

```python
"search_space": {
    "rfdetr_parameters": {
        "lr":           {"type": "float", "low": 1e-5, "high": 1e-3, "log": True},
        "weight_decay": {"type": "float", "low": 1e-6, "high": 1e-3, "log": True},
        "lr_drop":      {"type": "int",   "low": 10,   "high": 100,  "step": 10},
    },
    "preprocessing_config": {
        "brightness":   {"type": "float", "low": -0.3, "high": 0.3},
    },
}
```

**Supported spec types** (mirroring Optuna's `suggest_*`):

| `type` | Required keys | Optional keys |
|---|---|---|
| `float` | `low`, `high` | `log` (bool) |
| `int` | `low`, `high` | `step` (int) |
| `categorical` | `choices` (list) | — |

### 3. Tuning nested values (e.g. inside `aug_config`)

The search space is **recursive**: any dict *without* a `"type"` key is treated as a
container to descend into. So you can tune values nested deep inside a section — for
example, an augmentation probability inside `aug_config`:

```python
"search_space": {
    "rfdetr_parameters": {
        "aug_config": {
            "HorizontalFlip": {"p": {"type": "float", "low": 0.0, "high": 1.0}},
            "ColorJitter":    {"brightness": {"type": "float", "low": 0.0, "high": 0.5}},
        },
    },
}
```

This descends `rfdetr_parameters → aug_config → HorizontalFlip → p` and tunes that leaf.

### How it works (so the behaviour is predictable)

- Each trial gets a **deep-ish copy** of the config with suggestions overlaid; the base
  config (including the shared `aug_config` preset) is never mutated.
- The Optuna parameter name is the **dotted path** (e.g.
  `rfdetr_parameters.aug_config.HorizontalFlip.p`), which keeps names unique and is also
  the key used in the logged results.
- A search-space section/path that doesn't match the config is **warned and skipped**
  (it won't crash the run).

The tuned value automatically flows into the trial because the pipeline reads everything
from the per-trial config — no other code changes needed.

---

## Modifying the Pipeline & Adding New Tunable Hyperparameters

### A. Tune a parameter that already exists in the config

If the parameter is already a key in some `Config` section, **no code changes are
needed** — just add it to `search_space` at the matching path (see above). This covers
all of `rfdetr_parameters`, `preprocessing_config`, `rfdetr_dataset`, and anything nested
within them.

### B. Add a brand-new config section, then tune it

Sections are auto-discovered: any **dict attribute** of `Config` becomes
`config.<section>`. To add one:

1. Add the dict to `Config` in `config.py`:
   ```python
   class Config:
       my_section = {
           "some_param": 0.5,
       }
   ```
2. Read it where you need it in `objective()` (or a helper), e.g.
   `config.my_section["some_param"]`.
3. Tune it by mirroring the name in `search_space`:
   ```python
   "search_space": {
       "my_section": {"some_param": {"type": "float", "low": 0.0, "high": 1.0}},
   }
   ```
   The section name in `search_space` **must equal** the `Config` attribute name.

> A section must be a **dict** — that's how the loader distinguishes sections from helper
> attributes. A scalar top-level attribute would be ignored.

### C. Change what the objective actually does

`objective()` in [`rfdetr_train.py`](rfdetr_train.py) is the per-trial recipe, split into
the 7 numbered steps from the Overview. Common edits:

- **Augmentation pipeline** — the `A.Compose([...])` block (step 2) builds the
  Albumentations transforms from `preprocessing_config`. Add/reorder transforms here; to
  make a new transform parameter tunable, read it from `preprocessing_config` and add it
  to `search_space`.
- **Scoring** — step 6 computes the value Optuna optimizes. It currently combines the
  target class's validation AP and test mAP50:
  ```python
  target_idx = config.study["optimization_target_class"] or 0
  target_class = classes[target_idx]
  score = 1 / (1 / validation_results[f"val/AP/{target_class}"]
               + 1 / test_results[f"{target_class}_map_50"])
  ```
  Change this expression to optimize a different metric (e.g. average over all classes).
- **What gets logged** — step 7 builds `combined_params` and `combined_metrics`. Add
  entries here to capture extra values in the CSV/JSON outputs.

### D. Add a new spec *type*

The suggestion logic lives in `TrainingConfig._suggest_value()` in
[`utils/config_loader.py`](utils/config_loader.py). It maps a spec's `"type"` to the
matching `trial.suggest_*` call. Add an `elif` branch there to support a new type. The
recursion/tree-walking in `_apply_search_space()` doesn't need to change.

---

## Outputs

Per trial:
- **`runs/trial_<n>/`** — training outputs (`metrics.csv`, `checkpoint_best_total.pth`,
  `test_predictions/`).

Accumulated across trials (named after `study.name`):
- **`<study>_output.csv`** — one row per trial: params + validation/test metrics (append mode).
- **`<study>_output.json`** — detailed per-trial params and metrics.
- **`<study>_optuna_storage.json`** — Optuna study state, used to resume.

---

## Troubleshooting

**Config fails to load** — the loader fails fast with a clear message; check the named
section/key exists in `config.py` (no defaults are injected).

**A search-space entry seems ignored** — confirm `optuna["optimize"]` is `True`, and that
the section name matches a `Config` section exactly and the leaf has a `"type"` key. A
mismatch is logged as a warning and skipped.

**CUDA out of memory** — lower `batch_size` (and/or `resolution`) in `rfdetr_parameters`,
or raise `grad_accum_steps`.

**Trials fail immediately** — check the `split_dataset` format (images + YOLO `.txt`
labels together per split) and that `rf-detr-nano.pth` exists at the configured path.
```
