# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
