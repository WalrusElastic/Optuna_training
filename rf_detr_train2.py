"""
Optuna-based RF-DETR training optimization.

Simplified main script that orchestrates:
1. Config loading from YAML
2. Optuna study setup
3. Trial objective function execution
4. Results logging

Uses modular architecture:
- config_loader.py: Config loading (generic YAML-driven)
- cli.py: CLI argument parsing
- training_utils.preprocessing_utils: Dataset augmentation
- training_utils.data_logging_utils: Results logging
- training_utils.optuna_utils: Optuna persistence
- training_utils.rf_detr_trainer.py: Model training
"""
#NOTE logger disabled in  rf_detr_distributed_worker (model.train)
import logging
import os
import sys
import shutil
from pathlib import Path, WindowsPath
from typing import Any

import optuna
import yaml
import torch
import albumentations as A
import numpy as np

from config_loader import ConfigLoader
from cli import TrainingCLI
from utils.preprocessing_utils import PreprocessingUtils
from utils.data_logging_utils import DataLogger
from utils.optuna_utils import OptunaTrialManager
from utils.rf_detr_trainer import RFDETRTrainer

# Environment setup
os.environ["ALBUMENTATIONS_DISABLE"] = "1"
os.environ["USE_LIBUV"] = "0"

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

logger.info(f"CUDA Available: {torch.cuda.is_available()}")


# ========================== YAML SETUP ==========================

def setup_data_yaml(yaml_path: Path, dataset_path: Path, classes: list) -> None:
    """Create data.yaml for RF-DETR dataset.
    Args:
        yaml_path: Path to save data.yaml (in Final_dataset folder)
        dataset_path: Path to dataset (root containing train/valid/test)
        classes: List of class names
    """
    logger.info(f"Setting up data.yaml at {yaml_path}")
    data_yaml = {
        "path": str(dataset_path),
        "train": "train",
        "val": "valid",
        "test": "test",
        "augment": True,
        "channels": 1,
        "names": {index: name for index, name in enumerate(classes)}
    }

    with open(yaml_path, 'w') as f:
        yaml.safe_dump(data_yaml, f)

    logger.info(f"data.yaml created with {len(classes)} classes")


# ========================== DATASET PREPROCESSING ==========================

def prepare_augmented_dataset(
    config: Any,
    trial_number: int,
) -> None:
    """
    Augment and prepare the dataset for this trial.
    
    Args:
        config: TrainingConfig object
        trial_number: Current trial number (for logging)
    """
    logger.info(f"[Trial {trial_number}] Starting dataset preparation")
    
    # Define augmentation transforms
    brightness = config.augmentation_parameters["brightness"]
    contrast = config.augmentation_parameters["contrast"]
    sharpness = config.augmentation_parameters["sharpness"]

    transforms = A.Compose([
        A.RandomBrightnessContrast(
            brightness_limit=(brightness, brightness),
            contrast_limit=(contrast, contrast),
            brightness_by_max=False,
            p=1
        ),
        A.Sharpen(alpha=(sharpness, sharpness), lightness=(0.8, 0.8), p=1.0),
    ])

    # Augment and prepare dataset
    split_path = config.paths["split_dataset"]
    final_path = config.paths["final_dataset"]
    
    logger.info(f"[Trial {trial_number}] Augmenting dataset")
    if final_path.exists():
        shutil.rmtree(final_path)

    input_size = config.preprocessing_config["input_size"]
    output_size = config.preprocessing_config["training_size"]
    edit_labels = config.preprocessing_config["edit_labels"]

    for split in ['train', 'valid', 'test']:
        src_dir = split_path / split
        if not src_dir.exists():
            logger.warning(f"[Trial {trial_number}] Skipping {split} split - not found")
            continue

        logger.info(f"[Trial {trial_number}] Processing {split} split")
        images_dir = final_path / split / "images"
        labels_dir = final_path / split / "labels"
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

        tif_files = list(src_dir.glob("*.tif"))
    
        logger.info(f"[Trial {trial_number}] Found {len(tif_files)} images in {split}")
        
        for tif_path in tif_files:
            txt_path = tif_path.with_suffix('.txt')
            
            PreprocessingUtils.generate_transform(
                str(tif_path),
                str(txt_path),
                str(images_dir),
                str(labels_dir),
                transforms,
                edit_labels=edit_labels,
                iterations=1,
                input_img_size=input_size,
                output_img_size=output_size,
            )

    logger.info(f"[Trial {trial_number}] Dataset preparation completed")


# ========================== OBJECTIVE FUNCTION ==========================

def objective(trial: optuna.trial.Trial, config: Any) -> float:
    """
    Optuna objective function: train RF-DETR model and return optimization metric.
    
    Pipeline:
    1. Augment dataset
    2. Setup data YAML
    3. Prepare training parameters
    4. Train model (delegate to RFDETRTrainer)
    5. Log results
    
    Args:
        trial: Optuna trial object
        config: TrainingConfig object
        
    Returns:
        Optimization score (float)
    """
    trial_path = config.paths["runs_dir"] / f'trial_{trial.number}'

    logger.info(f"[Trial {trial.number}] ========================================")
    logger.info(f"[Trial {trial.number}] Starting trial")
    logger.info(f"[Trial {trial.number}] ========================================")

    try:
        # ========================== PREPROCESSING ==========================
        prepare_augmented_dataset(config, trial.number)

        # Setup data YAML
        setup_data_yaml(
            config.paths["data_yaml"],
            config.paths["final_dataset"],
            config.classes,
        )

        # ========================== TRAINING SETUP ==========================
        logger.info(f"[Trial {trial.number}] Preparing training parameters")
        
        training_params = config.rfdetr_parameters.copy()
        ##NOTE UPDATE VALUES FOR OPTUNA
        all_params = training_params.copy()
        all_params.update(config.preprocessing_config)
        # Apply Optuna search-space suggestions (overrides training_params)
        search_space = getattr(config, "optuna_search_space", {})
        logger.info(f"[Trial {trial.number}] Search space: {search_space}")
        for name, spec in search_space.items():
        
            param_type = spec.get("type", "float")
            if param_type == "float":
                val = trial.suggest_float(
                    name,
                    float(spec["low"]),
                    float(spec["high"]),
                    log=bool(spec.get("log", False))
                )
            elif param_type == "int":
                if "step" in spec:
                    val = trial.suggest_int(
                        name,
                        int(spec["low"]),
                        int(spec["high"]),
                        step=int(spec["step"])
                    )
                else:
                    val = trial.suggest_int(name, spec["low"], spec["high"])
            elif param_type == "categorical":
                val = trial.suggest_categorical(name, spec["choices"])
            else:
                continue
            
            training_params[name] = val


            logger.info(f"[Trial {trial.number}] Optuna suggestion: {name} = {val}")

        # Collect all parameters for logging
        

        # ========================== TRAINING ==========================
        logger.info(f"[Trial {trial.number}] Launching RF-DETR training")
        
        result = RFDETRTrainer.train(
            trial.number,
            trial_path,
            training_params,
            config,
        )

        score = result["score"]
        all_metrics = {
            "best_epoch": result["best_epoch"],
            "training_time": result["training_time"],
        }
        all_metrics.update(result["validation_results"])
        all_metrics.update(result["test_results"])

        # Convert numpy types for JSON serialization
        # corrected_metrics = {}
        # for key, val in all_metrics.items():
        #     if isinstance(val, (np.float64, np.float32)):
        #         corrected_metrics[key] = float(val)
        #     elif isinstance(val, (np.int64, np.int32)):
        #         corrected_metrics[key] = int(val)
        #     else:
        #         corrected_metrics[key] = val

        # corrected_params = {}
        # for key, val in all_params.items():
        #     if isinstance(val, (WindowsPath, Path)):
        #         corrected_params[key] = str(val)
        #     if isinstance(val, (np.float64, np.float32)):
        #         corrected_params[key] = float(val)
        #     elif isinstance(val, (np.int64, np.int32)):
        #         corrected_params[key] = int(val)
        #     else:
        #         corrected_params[key] = val
        corrected_metrics = DataLogger.make_json_safe(all_metrics)
        corrected_params = DataLogger.make_json_safe(all_params)
        print(corrected_params)
        # ========================== LOGGING ==========================
        logger.info(f"[Trial {trial.number}] Saving results")
        
        DataLogger.save_to_json(
            output_json_path=config.paths["output_json"],
            trial_number=trial.number,
            params=corrected_params,
            metrics=corrected_metrics,
        )

        DataLogger.save_to_csv(
            file_path=config.paths["output_csv"],
            data_dict={**corrected_params, **corrected_metrics},
        )

        OptunaTrialManager.save_trial_to_json(
            trial,
            config.paths["optuna_json"],
            score,
        )

        logger.info(f"[Trial {trial.number}] ========================================")
        logger.info(f"[Trial {trial.number}] Completed - Score: {score:.4f}")
        logger.info(f"[Trial {trial.number}] ========================================")

        return score

    except Exception as e:
        logger.error(f"[Trial {trial.number}] Exception: {e}", exc_info=True)
        raise


# ========================== MAIN ==========================

def main():
    """Main entry point."""
    logger.info("=" * 70)
    logger.info("Optuna-based RF-DETR Training Optimization")
    logger.info("=" * 70)

    # Parse CLI arguments
    args = TrainingCLI.parse_args()

    # Load config from YAML
    logger.info(f"Loading config from {args.config}")
    config = ConfigLoader.load(args.config)

    # Apply CLI overrides
    TrainingCLI.apply_overrides(config, args)

    logger.info(f"Study: {config.study_name}")
    logger.info(f"Classes: {config.classes}")
    logger.info(f"Optimization target class: {config.optimization_target_class}")
    logger.info(f"final_dataset path: {config.paths['final_dataset']}")
    
    # Create or load Optuna study
    logger.info("Setting up Optuna study")
    study = OptunaTrialManager.create_study_from_json(
        config.paths["optuna_json"],
        config.study_name,
    )

    # Run optimization
    logger.info(f"dataset_dir path: {config.rfdetr_parameters['dataset_dir']}")
    logger.info(f"Starting optimization ({config.optuna_parameters['n_trials']} trials)")
    try:
        study.optimize(
            lambda trial: objective(trial, config),
            n_trials=config.optuna_parameters["n_trials"],
            n_jobs=config.optuna_parameters.get("n_jobs", 1),
        )
        logger.info("Optimization completed successfully")
    except KeyboardInterrupt:
        logger.info("Optimization interrupted by user")
    except Exception as e:
        logger.error(f"Optimization failed: {e}", exc_info=True)
        raise

    logger.info("=" * 70)
    logger.info("Training script finished")
    logger.info("=" * 70)


if __name__ == '__main__':
    main()
