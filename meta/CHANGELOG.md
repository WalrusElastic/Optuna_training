# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2026-06-30

Internal restructuring to expose the pipeline's moving parts and simplify configuration.
No change to what a trial computes - same dataset prep, training, scoring, and outputs.

### Changed
- **The loader returns the `Config` class directly - the `TrainingConfig` wrapper is gone.**
  `ConfigLoader.load()` now validates and returns the `Config` class itself, and per-trial
  Optuna suggestions are overlaid onto that class *in place* (each tunable section rebuilt
  from a pristine snapshot taken at load, so suggestions never accumulate across trials).
  No wrapper object is instantiated. `build_trial_config()` is now a `ConfigLoader`
  static method. Note: in-place mutation assumes trials run sequentially (`n_jobs == 1`).
- **Per-trial / derived paths moved out of `config.py` into the training script.** The
  `data.yaml`, training-params JSON, and training-worker-script paths are now built in
  `rfdetr_train.py`, and *all* per-trial path resolution is collated in one block at the
  top of `objective()` (the TRIAL SETUP section) as the single source of truth. `config.py`
  keeps only the root, dataset, and study-named output paths.
- **Preprocessing is now inlined in `objective()`.** `PreprocessingUtils.generate_transform`
  and `preprocess_image` are no longer called; the augment-and-write loop spells out each
  step (image read -> min-max normalize -> cubic upscale -> 16-bit→8-bit -> albumentations
  transform -> write image + copy labels) so the preprocessing flow is visible end to end.
- **README**: documented that a new config section must be merged into `combined_params` in
  step 7 to appear in the CSV/JSON outputs, and refreshed references to the new loader API.

### Removed
- The `TrainingConfig` class (the loader uses the `Config` class as the config object).
- The unused `edit_labels` polygon-augmentation path from the per-image preprocessing loop.

## [0.1.0] - 2026-06-24

Refactor of the pipeline as originally pulled from the repo (an unversioned baseline built
around a modular, convoluted `rf_detr_train.py` with YAML config and a flat Optuna search
space). This release establishes versioning and captures all changes made to that baseline.

### Added
- **Any config section can be Optuna-tuned, not just `rfdetr_parameters`.** The search
  space routes by section name to any auto-discovered section (`preprocessing_config`,
  `rfdetr_dataset`, `study`, ...), whereas the baseline could only tune RF-DETR params.
- **Recursive / nested search space**: within a section, tune values nested arbitrarily
  deep (e.g. inside `aug_config`). Suggestions are overlaid on a per-trial config copy
  (base config left pristine) and Optuna parameter names use the dotted config path.
- **Fail-fast config validation** with clear errors for missing sections/keys.
- **Versioning**: `meta/version.py` + this `CHANGELOG.md`; the version is logged at the
  start of each run.

### Changed
- **Configuration moved from `config.yaml` to `config.py`** — a single `Config` class
  whose dict attributes are auto-discovered as sections. Native Python types (no YAML
  `1e-5`-as-string quirks), inline comments documenting RF-DETR defaults, paths resolved
  against the config file's own directory, and output filenames derived from the study name.
- **`config_loader.py` rewritten and moved to `utils/`** — full section auto-discovery
  with no hardcoded sections and no injected defaults (previously each section was
  hand-parsed with `setdefault` fallbacks).
- **Per-trial Optuna logic moved onto the config object** as
  `TrainingConfig.build_trial_config()`.
- **`aug_config` is now the imported RF-DETR preset object**, not a string name resolved later.
- **Config layout tidied**: `classes` and `optimization_target_class` moved under `study`;
  the `augmentations` section merged into `preprocessing_config`.
- **`objective()` cleaned up**: config extraction centralized at the top, result collation
  at the end, consistent sectioning and logging.
- **README rewritten** for the new architecture (configuration, tuning, extending the pipeline).

### Removed
- Legacy modular pipeline: `rf_detr_train.py`, `cli.py`, `preparetraining.py`, and
  `utils/rf_detr_trainer.py` — replaced by the single entry point `rfdetr_train.py`.
- `config.yaml` (replaced by `config.py`) and the root-level `config_loader.py`
  (moved to `utils/`).
- CLI argument parsing — overrides are now made directly in `config.py`.
