"""
Configuration loader: imports the ``Config`` class from config.py, validates it, and
applies Optuna search-space suggestions to it per trial. RF-DETR only - no YOLO deps.

The ``Config`` class is used *directly* as the config object: its public dict attributes
are the config sections (``config.paths``, ``config.rfdetr_parameters``, ...). No wrapper
class is instantiated. Per-trial Optuna suggestions are overlaid onto the ``Config``
class's sections in place, rebuilt each trial from a pristine snapshot of the base values
(taken at load) so suggestions never accumulate across trials.
"""

import copy
import importlib.util
import logging
from pathlib import Path
from typing import Any, Dict

import optuna

logger = logging.getLogger(__name__)

# Returned by _suggest_value for an unrecognized parameter type (the caller skips it).
_SKIP = object()

# Sections that must be present for the pipeline to run.
_REQUIRED_SECTIONS = ("study", "paths")


class ConfigLoader:
    """Loads, validates, and per-trial tunes the RF-DETR ``Config`` class from config.py."""

    @staticmethod
    def load(config_path: Path) -> type:
        """
        Import config.py and return its ``Config`` class, validated and ready to use.

        Each public dict attribute of ``Config`` is a config section, read as
        ``config.<section>``; non-dict attributes (helper constants) are ignored.

        Args:
            config_path: Path to config.py.

        Returns:
            The ``Config`` class itself (not an instance) - use its attributes directly.

        Raises:
            FileNotFoundError: If the config file doesn't exist.
            ValueError: If the file defines no ``Config`` class or is missing sections.
        """
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        spec = importlib.util.spec_from_file_location("rfdetr_config", config_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        config = getattr(module, "Config", None)
        if config is None:
            raise ValueError(f"{config_path} must define a Config class")

        ConfigLoader._validate(config)

        # Snapshot the base values of every tunable section so each trial overlays its
        # suggestions onto the original config, not onto the previous trial's values.
        search_space = config.optuna.get("search_space", {})
        config._base_sections = {
            section: copy.deepcopy(getattr(config, section))
            for section in search_space
            if isinstance(getattr(config, section, None), dict)
        }

        logger.info(f"Config loaded from {config_path}")
        return config

    @staticmethod
    def _validate(config: type) -> None:
        """
        Check required sections and a few invariants, failing fast with a clear message
        instead of letting consumers hit an opaque KeyError/AttributeError later.

        Raises:
            ValueError: If a required section is missing or the config is inconsistent.
        """
        sections = {
            name
            for name, value in vars(config).items()
            if not name.startswith("_") and isinstance(value, dict)
        }
        if not sections:
            raise ValueError("Config defines no config sections")

        missing = [s for s in _REQUIRED_SECTIONS if s not in sections]
        if missing:
            raise ValueError(f"Config is missing required section(s): {missing}")

        classes = config.study.get("classes")
        if not classes:
            raise ValueError("study.classes is empty or missing")

        target = config.study.get("optimization_target_class")
        if target is not None and target >= len(classes):
            raise ValueError(
                f"study.optimization_target_class={target} but only {len(classes)} classes defined"
            )

        epochs = config.rfdetr_parameters.get("epochs")
        if epochs is not None and epochs < 1:
            raise ValueError("epochs must be >= 1")

        for path_key in ("split_dataset", "pretrained_model_weights"):
            if path_key not in config.paths:
                raise ValueError(f"Config 'paths' is missing required entry: '{path_key}'")
            if not config.paths[path_key].exists():
                logger.warning(f"Path does not exist: {path_key}={config.paths[path_key]}")

    @staticmethod
    def build_trial_config(config: type, trial: optuna.trial.Trial) -> Dict[str, Any]:
        """
        Overlay this trial's Optuna suggestions onto the ``Config`` class in place.

        No new config object is created: the suggested sections are rebuilt from their
        pristine snapshots (``config._base_sections``) and assigned straight back onto the
        ``Config`` class, so the rest of the pipeline just reads ``config.<section>`` and
        sees this trial's values. Rebuilding from the snapshot keeps suggestions from
        accumulating across trials.

        ``optuna.search_space`` mirrors the config tree, nested by section and then as deep
        as needed. A dict with a ``"type"`` key is a leaf parameter spec; any other dict is
        a container to descend into (e.g. to reach values inside
        ``rfdetr_parameters.aug_config``).

        Note:
            Mutating the shared ``Config`` class in place assumes trials run sequentially
            (``optuna.n_jobs == 1``).

        Example::

            "search_space": {
                "rfdetr_parameters": {
                    "lr": {"type": "float", "low": 1e-5, "high": 1e-3, "log": True},
                    "aug_config": {
                        "HorizontalFlip": {"p": {"type": "float", "low": 0.0, "high": 1.0}},
                    },
                },
            }

        Returns:
            The suggested values, keyed by dotted path
            (e.g. "rfdetr_parameters.aug_config.HorizontalFlip.p").
        """
        suggested: Dict[str, Any] = {}

        if not config.optuna["optimize"]:
            return suggested

        for section, spec_tree in config.optuna["search_space"].items():
            base_section = config._base_sections.get(section)
            if base_section is None:
                logger.warning(f"[Trial {trial.number}] Unknown search-space section '{section}', skipping")
                continue
            updated = ConfigLoader._apply_search_space(trial, spec_tree, base_section, section, suggested)
            setattr(config, section, updated)

        return suggested

    @staticmethod
    def _apply_search_space(
        trial: optuna.trial.Trial,
        spec_node: Dict[str, Any],
        base_node: Dict[str, Any],
        path: str,
        suggested: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Recursively overlay a search-space subtree onto the matching config subtree,
        copying on the way down so ``base_node`` (the pristine snapshot) is never mutated.
        ``path`` is the dotted key prefix, used as both the Optuna parameter name and the
        logging key. Returns the new (copied) config node.
        """
        result = dict(base_node)  # copy-on-write at this level
        for key, sub_spec in spec_node.items():
            sub_path = f"{path}.{key}"

            if not isinstance(sub_spec, dict):
                logger.warning(f"[Trial {trial.number}] Invalid search-space entry at '{sub_path}', skipping")
                continue

            if "type" in sub_spec:
                # Leaf parameter spec: draw a value from Optuna.
                value = ConfigLoader._suggest_value(trial, sub_path, sub_spec)
                if value is _SKIP:
                    continue
                result[key] = value
                suggested[sub_path] = value
                logger.info(f"[Trial {trial.number}] Optuna suggest: {sub_path} = {value}")
            else:
                # Nested container: descend into the matching config sub-dict.
                child = base_node.get(key)
                if not isinstance(child, dict):
                    logger.warning(f"[Trial {trial.number}] No nested config dict at '{sub_path}', skipping")
                    continue
                result[key] = ConfigLoader._apply_search_space(trial, sub_spec, child, sub_path, suggested)

        return result

    @staticmethod
    def _suggest_value(trial: optuna.trial.Trial, name: str, spec: Dict[str, Any]) -> Any:
        """Draw one value from Optuna for a leaf spec; returns _SKIP for an unknown type."""
        ptype = spec["type"]
        if ptype == "float":
            return trial.suggest_float(
                name, float(spec["low"]), float(spec["high"]), log=bool(spec.get("log", False))
            )
        if ptype == "int":
            return trial.suggest_int(
                name, int(spec["low"]), int(spec["high"]), step=int(spec.get("step", 1))
            )
        if ptype == "categorical":
            return trial.suggest_categorical(name, spec["choices"])

        logger.warning(f"[Trial {trial.number}] Unknown search-space type '{ptype}' for '{name}', skipping")
        return _SKIP
