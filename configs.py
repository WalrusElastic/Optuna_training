import logging
import os
from pathlib import Path
from typing import Dict, List



logger = logging.getLogger(__name__)


class TrainingConfig:
    """
    Centralized configuration for paths, training parameters and class metadata.
    """

    def __init__(self):
        # Study / logging configuration
        self.study_name: str = "test_study"  # Name of the study

        # Central list of all class names used in training / evaluation
        self.classes: List[str] = ["Class_1", "Class_2", "Class_3", "Class_4"]

        # Root dir where this config file (and training scripts) are located
        root: Path = Path(os.path.dirname(os.path.realpath(__file__)))

        self.paths: Dict[str, Path] = {
            "root": root,
            "pretrained_model_weights": root / "rf-detr-nano.pth", # Path to model weights
            "runs_dir": root / "runs", # Path to store training runs
            "split_dataset": root / "split_dataset", # Path to dataset after splitting into train/val/test
            "final_dataset": root / "Final_dataset", # Path to dataset after preprocessing
            "yolo_yaml": root / "Final_dataset" / "data.yaml", # Path to YOLO data.yaml file, to be generated
            "optuna_json": root / f"{self.study_name}_optuna_storage.json", # Path to Optuna storage json file
            "output_csv": root / f"{self.study_name}_output.csv", # Path to output CSV file with study
            "output_json": root / f"{self.study_name}_output.json", # Path to output JSON file with study
        }

        # Default parameters for YOLO training, to be overridden by Optuna trials
        self.rfdetr_parameters: Dict = {
            # Basic training parameters
            "dataset_dir": str(self.paths["final_dataset"]),
            "output_dir": None, #NOTE To be set to trial-specific output dir during training
            "device": "cuda",
            "batch_size": 1,
            "grad_accum_steps": 4,
            "epochs": 3, #NOTE Set to 3 for testing
            "resolution": 512, #NOTE: Set to 512 for testing
            "early_stopping": True,
            "early_stopping_patience": 10,
            "early_stopping_min_delta": 0.5, #NOTE Set to 0.5 for testing. Defaults to 0.001

            # # Learning rates
            "lr": 1e-4,
            "lr_encoder": 1.5e-4,
            "weight_decay": 1e-4,
            "lr_drop": 11,
            "clip_max_norm": 0.1,
            "lr_vit_layer_decay": 0.8,
            "lr_component_decay": 1.0,

            # # Drop parameters
            "dropout": 0,
            "drop_path": 0,
            "drop_mode": "standard",
            "drop_schedule": "constant",
            "cutoff_epoch": 0,


            # # Matcher parameters
            "set_cost_class": 2,
            "set_cost_bbox": 5,
            "set_cost_giou": 2,

            # # Loss coefficients
            "cls_loss_coef": 2,
            "bbox_loss_coef": 5,
            "giou_loss_coef": 2,
            "focal_alpha": 0.25,
        }

        # Additional parameters that are not directly part of YOLO training but are relevant for the overall pipeline. Can be overridden by Optuna trials if needed.
        self.additional_parameters: Dict = {
            "brightness": -0.12655,
            "contrast": 0.18471,
            "sharpness": 0.15792,
        }

