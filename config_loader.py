"""
Configuration loader: reads YAML and provides validated config object.
RF-DETR only - no YOLO dependencies.
"""

import logging
import os
from os import path
from pathlib import Path
from typing import Dict, List, Optional, Any
from rfdetr.datasets.aug_config import AUG_CONSERVATIVE, AUG_AGGRESSIVE, AUG_AERIAL, AUG_INDUSTRIAL
import yaml

logger = logging.getLogger(__name__)


class ConfigLoader:
    """
    Loads and validates RF-DETR training configuration from YAML.
    
    Single Responsibility: Load and validate config files only.
    """

    @staticmethod
    def load(config_path: Path) -> "TrainingConfig":
        """
        Load config from YAML file.
        
        Args:
            config_path: Path to config.yaml file
            
        Returns:
            TrainingConfig object with all settings
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If config is invalid
        """
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, 'r') as f:
            raw_config = yaml.safe_load(f)

        if not raw_config:
            raise ValueError("Config file is empty")

        root: Path = Path(os.path.dirname(os.path.realpath(__file__)))
        config = TrainingConfig(raw_config, root) #config_path.parent
        config._validate()
        logger.info(f"Config loaded from {root}")
        return config


class TrainingConfig:
    """
    Centralized configuration for RF-DETR training: paths, parameters, and class metadata.
    
    Loads from YAML and provides dict-like access (paths, rfdetr_parameters, etc.).
    
    Interface Segregation: Only expose what consumers need (no unused fields).
    """

    def __init__(self, raw_config: Dict[str, Any], root_dir: Path):
        """
        Initialize from raw YAML dict.
        
        Args:
            raw_config: Parsed YAML dict
            root_dir: Root directory (for resolving relative paths)
        """
        self.root = root_dir
        self._raw = raw_config

        # Extract top-level sections
        self.study_name: str = raw_config.get("study", {}).get("name", "test_study")
        self.classes: List[str] = raw_config.get("classes", [])
        self.optimization_target_class: Optional[int] = raw_config.get("optimization_target_class")

        # Build paths dict (auto-generate output files if not set)
        path_config = raw_config.get("paths", {})
        self.paths: Dict[str, Path] = {
            "root": self.root,
            "pretrained_model_weights": self.root / path_config.get("pretrained_model_weights", "model.pth"),
            "split_dataset": self.root / path_config.get("split_dataset", "split_dataset"),
            "final_dataset": self.root / path_config.get("final_dataset", "Final_dataset"),
            "data_yaml": self.root / path_config.get("final_dataset", "Final_dataset") / path_config.get("data_yaml", "data.yaml"),
            "runs_dir": self.root / path_config.get("runs_dir", "runs"),
            "optuna_json": self.root / (path_config.get("optuna_json") or f"{self.study_name}_optuna_storage.json"),
            "output_csv": self.root / (path_config.get("output_csv") or f"{self.study_name}_output.csv"),
            "output_json": self.root / (path_config.get("output_json") or f"{self.study_name}_output.json"),
            "params_json": self.root / path_config.get("params_json", "training_params.json"),
            "training_worker_script": self.root / path_config.get("training_worker_script", "rf_detr_distributed_worker.py"),
        }

        # Preprocessing parameters
        # Gets values from yaml, if not sets a default value
        prep_config = raw_config.get("preprocessing", {})
        self.preprocessing_config: Dict[str, Any] = prep_config.copy()
        self.preprocessing_config.setdefault("input_size", 512)
        self.preprocessing_config.setdefault("training_size", 1024)
        self.preprocessing_config.setdefault("edit_labels", False)

        augmentation_config = raw_config.get("augmentations", {})
        self.augmentation_parameters: Dict[str, float] = augmentation_config.copy()
        self.augmentation_parameters.setdefault("brightness", 0)
        self.augmentation_parameters.setdefault("contrast", 0)
        self.augmentation_parameters.setdefault("sharpness", 0)

        # Optuna parameters
        optuna_config = raw_config.get("optuna", {})
        self.optuna_parameters: Dict[str, Any] = {
            "n_trials": optuna_config.get("n_trials", 1),
            "n_jobs": optuna_config.get("n_jobs", 1),
        }
        # Optuna search space for hyperparameter optimization
        self.optuna_search_space = optuna_config.get("search_space") or {}

        # RF-DETR training parameters - keep generic so new keys can be added in YAML
        rfdetr_config = raw_config.get("rfdetr_training", {})
        self.rfdetr_parameters: Dict[str, Any] = rfdetr_config.copy()
        # sensible defaults for commonly used keys
        ds = self.rfdetr_parameters.get("dataset_dir")
        if not ds or ds == "None":
            self.rfdetr_parameters["dataset_dir"] = self.paths["final_dataset"]  # Will be set to final_dataset if None
        self.rfdetr_parameters.setdefault("epochs", 100)
        self.rfdetr_parameters.setdefault("batch_size", 4)
        self.rfdetr_parameters.setdefault("lr", 1e-4)
        self.rfdetr_parameters.setdefault("early_stopping_patience", 10)

        Default_Augs = {
            "AUG_CONSERVATIVE": AUG_CONSERVATIVE,
            "AUG_AGGRESSIVE": AUG_AGGRESSIVE,
            "AUG_AERIAL": AUG_AERIAL,
            "AUG_INDUSTRIAL": AUG_INDUSTRIAL,
        }

        aug = self.rfdetr_parameters.get("aug_config")

        if aug in Default_Augs:
            self.rfdetr_parameters["aug_config"] = Default_Augs[aug]


        # RF-DETR dataset parameters - keep generic
        rfdetr_dataset_config = raw_config.get("rfdetr_dataset", {})
        self.rfdetr_dataset: Dict[str, Any] = rfdetr_dataset_config.copy()
        self.rfdetr_dataset.setdefault("test_threshold", 0.03)
        self.rfdetr_dataset.setdefault("iou_threshold", 0.5)
        self.rfdetr_dataset.setdefault("num_gpus", 1)
        self.rfdetr_dataset.setdefault("num_trials", 1)

    def _validate(self) -> None:
        """
        Validate configuration for correctness.
        
        Raises:
            ValueError: If config is invalid
        """
        if not self.classes:
            raise ValueError("No classes defined in config")

        if self.optimization_target_class is not None:
            if self.optimization_target_class >= len(self.classes):
                raise ValueError(
                    f"optimization_target_class={self.optimization_target_class} "
                    f"but only {len(self.classes)} classes defined"
                )

        if self.rfdetr_parameters.get("epochs", 1) < 1:
            raise ValueError("epochs must be >= 1")

        required_paths = ["split_dataset", "pretrained_model_weights"]
        for path_key in required_paths:
            if not self.paths[path_key].exists():
                logger.warning(f"Path does not exist: {path_key}={self.paths[path_key]}")

    def __repr__(self) -> str:
        return (
            f"TrainingConfig(study={self.study_name}, "
            f"classes={len(self.classes)}, "
            f"target_class={self.optimization_target_class})"
        )
