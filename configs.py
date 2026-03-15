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
            "default_model_weights": root / "yolo11n-seg.pt", # Path to model weights
            "runs_dir": root / "runs", # Path to store training runs
            "split_dataset": root / "split_dataset", # Path to dataset after splitting into train/val/test
            "final_dataset": root / "Final_dataset", # Path to dataset after preprocessing
            "yolo_yaml": root / "data.yaml", # Path to YOLO data.yaml file, to be generated
            "optuna_json": root / f"{self.study_name}_optuna_storage.json", # Path to Optuna storage json file
            "output_csv": root / f"{self.study_name}_output.csv", # Path to output CSV file with study
            "output_json": root / f"{self.study_name}_output.json", # Path to output JSON file with study
        }

        # Default parameters for YOLO training, to be overridden by Optuna trials
        self.yolo_parameters: Dict = {
            "data": str(self.paths["yolo_yaml"]),
            "epochs": 1, # NOTE: change back once done
            "patience": 50,
            "batch": 8,
            "save": True,
            "save_period": 0,
            "device": 0,  # NOTE: change back once done
            "imgsz": 1024,
            "project": str(self.paths["runs_dir"]),
            "exist_ok": True,
            "optimizer": "adamW",
            "verbose": True,
            "seed": 42,
            "resume": False,
            "freeze": None,
            "dropout": 0.03,
            "val": True,
            "plots": True,
            "augment": True,
            "amp": False,
            "cache": True,
            "workers": 0,
            "lr0": 0.00029076,
            "lrf": 0.000326,
            "momentum": 0.70496,
            "weight_decay": 0.1051539,
            "hsv_h": 0.015,
            "hsv_s": 0.7,
            "hsv_v": 0.4,
            "degrees": 0.0,
            "translate": 0.1,
            "scale": 0.5,
            "shear": 0.0,
            "perspective": 0.0,
            "flipud": 0.0,
            "fliplr": 0.5,
            "bgr": 0,
            "mosaic": 0.7251009743589250,
            "mixup": 0.3047758740425330,
            "cutmix": 0.19502787808654200,
            "copy_paste": 0.08030030302605730,
            "copy_paste_mode": "mixup",
            "box": 4.668834351,
            "cls": 1.299902183,
            "dfl": 3.942576859,
        }

        # Additional parameters that are not directly part of YOLO training but are relevant for the overall pipeline. Can be overridden by Optuna trials if needed.
        self.additional_parameters: Dict = {
            "brightness": -0.12655,
            "contrast": 0.18471,
            "sharpness": 0.15792,
        }

        # weight scale for YOLOWeightedDataset, to be used for class balancing in the dataset. 1.0 means the weight for each class is directly proportional to the inverse of its frequency. >1 exaggerates the weighting effect, <1 smooths it out.
        self.yolo_dataset_parameters: Dict = {
            "weight_scale": 1.2,
        }


