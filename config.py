"""
Training configuration for the Optuna RF-DETR pipeline.

Plain-Python config expressed as a class (easy to read / comment, with IDE support).
Each public *dict* attribute of Config is a config section, exposed on the loaded
TrainingConfig as ``config.<section>``. Paths are resolved here against this file's
directory (``paths["root"]``); the loader takes them as-is. Only dict attributes are
treated as sections; any non-dict attributes would be ignored as helpers.
"""

from pathlib import Path

# Options: AUG_CONSERVATIVE, AUG_AGGRESSIVE, AUG_AERIAL, AUG_INDUSTRIAL
from rfdetr.datasets.aug_configs import AUG_AGGRESSIVE #NOTE, aug_config has been changed to aug_configs in the rfdetr 1.8.1


class Config:
    study = {
        "name": "test_study",
        "classes": [
            "Class_1",
            "Class_2",
            "Class_3",
            "Class_4",
        ],
        # Index of the class used for the primary optimization metric (mAP50-95).
        # If None, falls back to class index 0.
        "optimization_target_class": 2,  # Class_3
    }

    # Paths are resolved against this file's directory (paths["root"]); the loader takes
    # them as-is. "root" doubles as the project root used for subprocess working dirs.
    paths = {"root": Path(__file__).resolve().parent}
    paths.update({
        "pretrained_model_weights": paths["root"] / "rf-detr-nano.pth",
        "split_dataset": paths["root"] / "split_dataset",
        "final_dataset": paths["root"] / "Final_dataset",
        "data_yaml": paths["root"] / "Final_dataset" / "data.yaml",
        "runs_dir": paths["root"] / "runs",
        # Output artifacts named after the study.
        "optuna_json": paths["root"] / f"{study['name']}_optuna_storage.json",
        "output_csv": paths["root"] / f"{study['name']}_output.csv",
        "output_json": paths["root"] / f"{study['name']}_output.json",
        "params_json": paths["root"] / "training_params.json",
        "training_worker_script": paths["root"] / "utils" / "rf_detr_distributed_worker.py",
    })

    preprocessing_config = {
        "input_size": 256,
        "training_size": 256,
        "edit_labels": False,
        # Albumentation parameters (applied per trial in the A.Compose pipeline).
        "brightness": -0.12655,
        "contrast": 0.18471,
        "sharpness": 0.15792,
    }

    rfdetr_parameters = {
        # dataset_dir is set by the training script (= paths.final_dataset), not here.
        # Inline comments show the rfdetr default for each parameter.
        "devices": "auto",                 # default: 1
        "batch_size": 1,                   # default: 4
        "grad_accum_steps": 4,             # default: 4
        "epochs": 1,                       # default: 100
        "resolution": 512,                 # default: 384
        "early_stopping": True,            # default: False
        "early_stopping_patience": 1,      # default: 10
        "early_stopping_min_delta": 0.5,   # default: 0.001
        "lr": 1.0e-4,                      # default: 1e-4
        "lr_encoder": 1.5e-4,              # default: 1.5e-4
        "weight_decay": 1.0e-4,            # default: 1e-4
        "lr_drop": 11,                     # default: 100
        "clip_max_norm": 0.1,              # default: 0.1
        "lr_vit_layer_decay": 0.8,         # default: 0.8
        "lr_component_decay": 1.0,         # default: 0.7
        # rfdetr augmentation preset (imported above).        default: None
        # Options: AUG_CONSERVATIVE, AUG_AGGRESSIVE, AUG_AERIAL, AUG_INDUSTRIAL.
        "aug_config": AUG_AGGRESSIVE,
        "use_ema": False,                  # default: True
        "dropout": 0,                      # default: 0
        "drop_path": 0,                    # default: 0.0
        "drop_mode": "standard",           # default: "standard"
        "drop_schedule": "constant",       # default: "constant"
        "cutoff_epoch": 0,                 # default: 0
        "set_cost_class": 2,               # default: 2
        "set_cost_bbox": 5,                # default: 5
        "set_cost_giou": 2,                # default: 2
        "cls_loss_coef": 2,                # default: 1.0
        "bbox_loss_coef": 5,               # default: 5
        "giou_loss_coef": 2,               # default: 2
        "focal_alpha": 0.25,               # default: 0.25
        "progress_bar": "rich",              # default: None
    }

    rfdetr_dataset = {
        "test_threshold": 0.03,
        "iou_threshold": 0.5,
        "num_gpus": 1,
        "num_trials": 1,
    }

    optuna = {
        "n_trials": 1,
        "n_jobs": 1,
        "optimize": False,  # set True to apply Optuna hyperparameter optimization
        # Search space for Optuna, nested by config section. Each section name must match
        # a config key (e.g. "rfdetr_parameters") so suggestions land in the right place.
        "search_space": {
            "rfdetr_parameters": {
                "lr": {"type": "float", "low": 1.0e-5, "high": 1.0e-3, "log": True},
            },
        },
    }
