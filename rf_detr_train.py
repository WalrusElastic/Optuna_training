"""
Script to train RF-DETR on a dataset with Optuna-suggested parameters, within a folder.
Updates augmentation and final dataset folders for each trial and trains the model.
"""

import logging
import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances
import pandas as pd
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
import time
import math
import random
from tqdm import tqdm
import shutil
import subprocess

import matplotlib.pyplot as plt
import numpy as np
import cv2
import yaml
import torch
import seaborn as sns
import albumentations as A
from rfdetr import RFDETRNano

from configs import TrainingConfig
from utils.rf_detr_extract_utils import RFDETRExtractor
from utils.preprocessing_utils import PreprocessingUtils
from utils.data_logging_utils import DataLogger
from utils.optuna_utils import OptunaTrialManager
from utils.rf_detr_prediction_utils import RFDETRPredictor
from utils.yolo_evaluation_utils import YOLOBBSEvaluator

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))
os.environ["ALBUMENTATIONS_DISABLE"] = "1"
os.environ["USE_LIBUV"] = "0" # Disable libuv to prevent potential issues with subprocesses in Windows environments


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

logger.info(f"CUDA Available: {torch.cuda.is_available()}")

# ========================== DATA PREPROCESSING ==========================

def augment_and_prepare_final_dataset(
    transforms: A.Compose,
    split_path: Path,
    final_path: Path,
    input_size: int = 512,
    output_size: int = 1024,
    edit_labels: bool = False,
) -> None:
    """
    Augment the split dataset and write it directly in YOLO final format.

    Processes and augments all images in train/val/test from split_path,
    writing images and labels into final_path with YOLO layout (split/images, split/labels).
    In the split path, the images and labels must be in the same folder for each of the train, val and test splits.

    Args:
        transforms (A.Compose): Albumentations transformation pipeline (passed to generate_transform).
        split_path (Path): Root path containing train/val/test subdirectories.
        final_path (Path): Output root for final dataset (train/images, train/labels, etc.).
        input_size (int): Input image size for preprocessing (default: 512).
        output_size (int): Output image size after preprocessing (default: 1024).
        edit_labels (bool): Determines if the labels are edited during augmentation. NOTE: if any of the albumentation augmentations could change the polgon points, set this to True.

    Input:  split_path/train|val|test with .tif and .txt
    Output: final_path/train|val|test/images and final_path/train|val|test/labels
    """
    logger.info(f"Starting data augmentation from {split_path} to {final_path}")
    if final_path.exists():
        logger.info(f"Removing existing final dataset directory: {final_path}")
        shutil.rmtree(final_path)

    for split in ['train', 'valid', 'test']:
        src_dir = split_path / split
        if not src_dir.exists():
            logger.warning(f"Skipping {split} split - directory not found")
            continue
        
        logger.info(f"Processing {split} split from {src_dir}")
        images_dir = final_path / split / "images"
        labels_dir = final_path / split / "labels"
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

        tif_files = [f for f in os.listdir(src_dir) if f.endswith('.tif')]
        logger.info(f"Found {len(tif_files)} TIF files in {split} split")
        for file_name in tqdm([str(f) for f in tif_files], desc=f"Augmenting {split} split"):
            tif_path = str(src_dir / file_name)
            txt_path = str(src_dir / f"{os.path.splitext(file_name)[0]}.txt")
            PreprocessingUtils.generate_transform(
                tif_path,
                txt_path,
                str(images_dir),
                str(labels_dir),
                transforms,
                edit_labels=edit_labels,
                iterations=1,
                input_img_size=input_size,
                output_img_size=output_size,
            )
    
    logger.info(f"Data augmentation completed. Final dataset saved to {final_path}")

# ========================== SETTING UP YAML ==========================
    
def setup_yaml(yaml_path: Path, dataset_path: Path, classes) -> None:
    """Create data.yaml for YOLO."""
    logger.info(f"Setting up YOLO data.yaml at {yaml_path}")
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
    
    logger.info(f"YOLO data.yaml created with {len(classes)} classes")

# ========================== MAIN OBJECTIVE FUNCTION ==========================

def objective(trial: optuna.trial.Trial, config: TrainingConfig) -> float:
    """Objective function for Optuna optimization."""
    
    # ========================== SETUP ==========================
    trial_path = config.paths["runs_dir"] / f'trial_{trial.number}'
    
    logger.info(f"[Trial {trial.number}] Starting trial...")
    
    # ========================== PREPROCESSING ==========================
    logger.info(f"[Trial {trial.number}] Starting preprocessing phase")
    additional_parameters = config.additional_parameters

    # Extract additional preprocessing parameters
    brightness = additional_parameters["brightness"]
    contrast = additional_parameters["contrast"]
    sharpness = additional_parameters["sharpness"]

    transforms = A.Compose([
        A.RandomBrightnessContrast(
            brightness_limit=(brightness, brightness),
            contrast_limit=(contrast, contrast),
            brightness_by_max=False, p=1
        ),
        A.Sharpen(alpha=(sharpness, sharpness), lightness=(0.8, 0.8), p=1.0),
    ])
    
    logger.info(f"[Trial {trial.number}] Starting dataset preparation and augmentation")
    # prepare final dataset with augmentations
    augment_and_prepare_final_dataset(
        transforms, 
        config.paths["split_dataset"], 
        config.paths["final_dataset"],
        input_size=config.slice_size, 
        output_size=config.training_size #NOTE: Set to 512 for testing. Defaults to 1024
    )
    # setup yaml for YOLO training
    setup_yaml(
        config.paths["yolo_yaml"], 
        config.paths["final_dataset"], 
        config.classes)

    # ========================== TRAINING ==========================
    logger.info(f"[Trial {trial.number}] Starting training phase")
    
    # Collecting default parameters
    default_params = config.rfdetr_parameters
    training_params = {}
    training_params.update(default_params)
    training_params["output_dir"] = str(trial_path)

    # NOTE: OVERWRITE default parameters with any trial-specific suggestions here if needed, e.g.:
    # training_params["lr"] = trial.suggest_float("lr", 1e-5, 1e-3, log=True)

    # Combine default and additional parameters for logging
    combined_params = {**default_params, **additional_parameters}
    
    logger.info(f"[Trial {trial.number}] Starting RF-DETR training with {default_params['epochs']} epochs")

    trial_path.mkdir(parents=True, exist_ok=True)

    # Save training parameters to JSON for the worker script to consume
    params_json_path = config.paths["params_json"]
    if params_json_path.exists():
        logger.info(f"Removing existing params JSON file at {params_json_path}")
        params_json_path.unlink()
    with open(params_json_path, "w") as f:
        json.dump(training_params, f, indent=2)

    # Launch training subprocess (handles distributed setup internally based on OS)
    logger.info(f"[Trial {trial.number}] Launching training subprocess with {config.num_gpus} GPU(s)")

    training_start_time = time.time()
    if sys.platform == "win32":
        logger.info(f"[Trial {trial.number}] Running on Windows: using direct subprocess (no torch.distributed)")
        launch_cmd = [
            sys.executable,
            str(config.paths["training_worker_script"]),
            "--pretrain-weights",
            str(config.paths["pretrained_model_weights"]),
            "--params-json",
            str(params_json_path),
        ]
        logger.info(f"[Trial {trial.number}] Launching training: {' '.join(launch_cmd)}")
        proc = subprocess.run(
            launch_cmd,
            cwd=str(config.paths["root"]),
        )
        if proc.returncode != 0:
            logger.error(f"[Trial {trial.number}] Training subprocess failed with code {proc.returncode}")
            raise RuntimeError(
                f"Training subprocess failed with return code {proc.returncode}. "
                "Check console output above for details."
            )
    else:
        logger.info(f"[Trial {trial.number}] Running on Linux: using torch.distributed.run")
        launch_cmd = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            f"--nproc_per_node={config.num_gpus}",
            str(config.paths["training_worker_script"]),
            "--pretrain-weights",
            str(config.paths["pretrained_model_weights"]),
            "--params-json",
            str(params_json_path),
        ]
        logger.info(f"[Trial {trial.number}] Launching distributed training: {' '.join(launch_cmd)}")
        proc = subprocess.run(
            launch_cmd,
            cwd=str(config.paths["root"]),
        )
        if proc.returncode != 0:
            logger.error(f"[Trial {trial.number}] Training subprocess failed with code {proc.returncode}")
            raise RuntimeError(
                f"Training subprocess failed with return code {proc.returncode}. "
                "Check console output above for details."
            )
    logger.info(f"[Trial {trial.number}] Training completed successfully")
    training_end_time = time.time()
    training_time = training_end_time - training_start_time
    # ========================== EXTRACTING METRICS FROM RF-DETR OUTPUT ==========================
    rf_detr_val_metrics_path = trial_path/ "metrics.csv"
    best_epoch = RFDETRExtractor.get_best_epoch(
        rf_detr_val_metrics_path, 
        patience=training_params["early_stopping_patience"]
        )
    logger.info(f"[Trial {trial.number}] Best epoch identified: {best_epoch}")
    logger.info(f"[Trial {trial.number}] Final epoch identified: {best_epoch + training_params['early_stopping_patience']}")
    logger.info(f"[Trial {trial.number}] {best_epoch + training_params['early_stopping_patience']} epochs completed in {training_time:.2f} seconds, average time per epoch: {training_time/(best_epoch + training_params['early_stopping_patience']):.2f} seconds")



    logger.info(f"[Trial {trial.number}] Extracting validation RF-DETR output")  

    validation_results = RFDETRExtractor.get_validation_results(
        rf_detr_val_metrics_path, 
        best_epoch
        )
 
    # ========================== TESTING MODEL ==========================
    test_predictions_dir = trial_path / "test_predictions" 
    test_predictions_dir.mkdir(parents=True, exist_ok=True)
    model_path = trial_path / "checkpoint_best_total.pth"
    print(model_path)
    model = RFDETRNano(pretrain_weights = str(model_path))
    RFDETRPredictor.predict_image_dir_and_save_bbs_labels(
        model=model,
        classes=config.classes,
        input_dir=config.paths["final_dataset"] / "test" / "images",
        pred_labels_dir=test_predictions_dir,
        threshold=0.03, # Set a low threshold to capture more predictions for evaluation of 50-95,
        include_confidence=True
        )
    
    test_results, _, _ = YOLOBBSEvaluator.evaluate(
        pred_dir=test_predictions_dir,
        gt_dir=config.paths["final_dataset"] / "test" / "labels",
        classes=config.classes
    )

    
    # ========================== Calculating Optimization Score & Logging output ==========================
    score = 1/ (1/validation_results[f"val/AP/{config.classes[2]}"] + 1/test_results[f"{config.classes[2]}_map_50_95"])

    logger.info(f"[Trial {trial.number}] Optimization score calculated: {score:.5f}")
    
    combined_metrics = {}
    combined_metrics["best_epoch"] = best_epoch
    combined_metrics.update(validation_results)
    combined_metrics.update(test_results)
    combined_metrics['score'] = score

    # converting to float 32 and int 32 for values in combined metrics and combined params
    corrected_combined_metrics = {}
    for key in combined_metrics:
        if isinstance(combined_metrics[key], np.float64):
            corrected_combined_metrics[key] =float(combined_metrics[key])
        elif isinstance(combined_metrics[key], np.int64):
            corrected_combined_metrics[key] = int(combined_metrics[key])
        else:
            corrected_combined_metrics[key] = combined_metrics[key]

    corrected_combined_params = {}
    for key in combined_params:
        if isinstance(combined_params[key], np.float64):
            corrected_combined_params[key] = np.float32(combined_params[key])
        elif isinstance(combined_params[key], np.int64):
            corrected_combined_params[key] = np.int32(combined_params[key])
        else:
            corrected_combined_params[key] = combined_params[key]

    logger.info(f"[Trial {trial.number}] Saving results")
    
    DataLogger.save_to_json(
        output_json_path=config.paths["output_json"],
        trial_number=trial.number,
        params=corrected_combined_params,
        metrics=corrected_combined_metrics
    )
    
    DataLogger.save_to_csv(
        file_path=config.paths["output_csv"],
        data_dict={**corrected_combined_params, **corrected_combined_metrics}
    )
    
    OptunaTrialManager.save_trial_to_json(trial, config.paths["optuna_json"], score)
    logger.info(f"[Trial {trial.number}] Completed - Score: {score:.4f}")
    
    return score

def main():
    """Main entry point."""
    logger.info("="*60)
    logger.info("Starting Optuna-based RF-DETR training optimization")
    logger.info("="*60)
    
    config = TrainingConfig()
    logger.info(f"Configuration loaded - Study name: {config.study_name}")
    logger.info(f"Classes: {config.classes}")

    
    try:
        logger.info("Creating or loading Optuna study")
        study = OptunaTrialManager.create_study_from_json(config.paths["optuna_json"], config.study_name)
        logger.info(f"Starting optimization loop ({config.num_trials} trials)")
        study.optimize(
            lambda trial: objective(trial, config),
            n_trials=config.num_trials,
            n_jobs=1
        )
        logger.info("Optimization completed successfully")
    
    except KeyboardInterrupt:
        logger.info('Ctrl+c detected!')

    
    except Exception as e:
        logger.error(f"Trial failed: {e}")

if __name__ == '__main__':
    main()
