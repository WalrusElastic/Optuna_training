"""
Configuration loader: builds a validated config object from the ``Config`` class in
config.py. RF-DETR only - no YOLO dependencies.
"""

import copy
import importlib.util
import logging
from pathlib import Path
from typing import Any, Dict, Tuple

import optuna

logger = logging.getLogger(__name__)

# Returned by _suggest_value for an unrecognized parameter type (the caller skips it).
_SKIP = object()


class ConfigLoader:
    """Loads and validates the RF-DETR training configuration from a Python config file."""

    @staticmethod
    def load(config_path: Path) -> "TrainingConfig":
        """
        Load config from a Python file that defines a ``Config`` class.

        Each public dict attribute of ``Config`` is treated as a config section; non-dict
        attributes (e.g. helper constants) are ignored.

        Args:
            config_path: Path to config.py

        Returns:
            A validated TrainingConfig.

        Raises:
            FileNotFoundError: If the config file doesn't exist.
            ValueError: If the file defines no ``Config`` class or no sections.
        """
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        spec = importlib.util.spec_from_file_location("rfdetr_config", config_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        config_cls = getattr(module, "Config", None)
        if config_cls is None:
            raise ValueError(f"{config_path} must define a Config class")

        raw_config = {
            name: value
            for name, value in vars(config_cls).items()
            if not name.startswith("_") and isinstance(value, dict)
        }
        if not raw_config:
            raise ValueError(f"{config_path} Config defines no config sections")

        config = TrainingConfig(raw_config)
        config._validate()
        logger.info(f"Config loaded from {config_path}")
        return config


class TrainingConfig:
    """
    RF-DETR training configuration: paths, parameters, and study metadata.

    Built from the section dicts of the Config class. Every section is exposed as a
    same-named attribute (``config.paths``, ``config.rfdetr_parameters``, ...), passed
    through exactly as defined in config.py (which owns path resolution).
    """

    def __init__(self, raw_config: Dict[str, Any]):
        """
        Args:
            raw_config: Mapping of section name -> section dict (from the Config class).
        """
        self._raw = raw_config

        # Auto-discovery: expose every section as a same-named attribute, as-is
        # (no defaults injected, no path rewriting). Read a section as config.<section>.
        for name, section in raw_config.items():
            setattr(self, name, section)

    def __repr__(self) -> str:
        study = self.study
        return (
            f"TrainingConfig(study={study.get('name')}, "
            f"classes={len(study.get('classes', []))}, "
            f"target_class={study.get('optimization_target_class')})"
        )

    def _validate(self) -> None:
        """
        Check required sections and a few invariants, failing fast with a clear message
        instead of letting consumers hit an opaque KeyError/AttributeError later.

        Raises:
            ValueError: If a required section is missing or the config is inconsistent.
        """
        required_sections = ("study", "paths", "optuna", "rfdetr_parameters")
        missing = [s for s in required_sections if s not in self._raw]
        if missing:
            raise ValueError(f"Config is missing required section(s): {missing}")

        classes = self.study.get("classes")
        if not classes:
            raise ValueError("study.classes is empty or missing")

        target = self.study.get("optimization_target_class")
        if target is not None and target >= len(classes):
            raise ValueError(
                f"study.optimization_target_class={target} but only {len(classes)} classes defined"
            )

        epochs = self.rfdetr_parameters.get("epochs")
        if epochs is not None and epochs < 1:
            raise ValueError("epochs must be >= 1")

        for path_key in ("split_dataset", "pretrained_model_weights"):
            if path_key not in self.paths:
                raise ValueError(f"Config 'paths' is missing required entry: '{path_key}'")
            if not self.paths[path_key].exists():
                logger.warning(f"Path does not exist: {path_key}={self.paths[path_key]}")

    def build_trial_config(self, trial: optuna.trial.Trial) -> Tuple["TrainingConfig", Dict[str, Any]]:
        """
        Build a per-trial config by overlaying Optuna suggestions onto this one.

        ``optuna.search_space`` mirrors the config tree, nested by section and then as deep
        as needed. A dict with a ``"type"`` key is a leaf parameter spec; any other dict is
        a container to descend into (e.g. to reach values inside ``rfdetr_parameters.aug_config``).
        Suggestions are applied to copies made on the way down, so the base config is never
        mutated.

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
            (trial_config, suggested): the per-trial config and the chosen values, keyed by
            dotted path (e.g. "rfdetr_parameters.aug_config.HorizontalFlip.p").
        """
        trial_config = copy.copy(self)
        suggested: Dict[str, Any] = {}

        if not self.optuna["optimize"]:
            return trial_config, suggested

        for section, spec_tree in self.optuna["search_space"].items():
            if not hasattr(self, section):
                logger.warning(f"[Trial {trial.number}] Unknown search-space section '{section}', skipping")
                continue
            new_section = self._apply_search_space(trial, spec_tree, getattr(self, section), section, suggested)
            setattr(trial_config, section, new_section)

        return trial_config, suggested

    def _apply_search_space(
        self,
        trial: optuna.trial.Trial,
        spec_node: Dict[str, Any],
        base_node: Dict[str, Any],
        path: str,
        suggested: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Recursively overlay a search-space subtree onto the matching config subtree,
        copying on the way down so ``base_node`` is never mutated. ``path`` is the dotted
        key prefix, used as both the Optuna parameter name and the logging key. Returns the
        new (copied) config node.
        """
        result = dict(base_node)  # copy-on-write at this level
        for key, sub_spec in spec_node.items():
            sub_path = f"{path}.{key}"

            if not isinstance(sub_spec, dict):
                logger.warning(f"[Trial {trial.number}] Invalid search-space entry at '{sub_path}', skipping")
                continue

            if "type" in sub_spec:
                # Leaf parameter spec: draw a value from Optuna.
                value = self._suggest_value(trial, sub_path, sub_spec)
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
                result[key] = self._apply_search_space(trial, sub_spec, child, sub_path, suggested)

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
