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

from configs import TrainingConfig
from utils.rf_detr_extract_utils import RFDETRExtractor
from utils.preprocessing_utils import PreprocessingUtils
from utils.data_logging_utils import DataLogger
from utils.optuna_utils import OptunaTrialManager

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

    # Extract parameters
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
    augment_and_prepare_final_dataset(
        transforms, 
        config.paths["split_dataset"], 
        config.paths["final_dataset"],
        input_size=config.slice_resolution,
        output_size=config.train_resolution #NOTE: Set to train_resolution for testing. Defaults to 1024
    )
    setup_yaml(
        config.paths["yolo_yaml"], 
        config.paths["final_dataset"], 
        config.classes)

    # ========================== TRAINING ==========================
    logger.info(f"[Trial {trial.number}] Starting training phase")
    default_params = config.rfdetr_parameters

    training_params = {}
    training_params.update(default_params)
    training_params["output_dir"] = str(trial_path)

    combined_params = {**default_params, **additional_parameters}
    
    logger.info(f"[Trial {trial.number}] Starting RF-DETR training with {default_params['epochs']} epochs")
    trial_path.mkdir(parents=True, exist_ok=True)

    params_json_path = config.paths["params_json"]
    if params_json_path.exists():
        logger.info(f"Removing existing params JSON file at {params_json_path}")
        params_json_path.unlink()
    with open(params_json_path, "w") as f:
        json.dump(training_params, f, indent=2)

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
    # ========================== EXTRACTING METRICS ==========================
    rf_detr_log_path = trial_path / "log.txt"  # Assuming RF-DETR saves a training log with metrics
    best_epoch = RFDETRExtractor.get_best_epoch(rf_detr_log_path)
    last_epoch = RFDETRExtractor.get_final_epoch(rf_detr_log_path)
    logger.info(f"[Trial {trial.number}] Best epoch identified: {best_epoch}")
    logger.info(f"[Trial {trial.number}] Final epoch identified: {last_epoch}")
    logger.info(f"[Trial {trial.number}] {last_epoch} epochs completed in {training_time:.2f} seconds, average time per epoch: {training_time/last_epoch:.2f} seconds")



    logger.info(f"[Trial {trial.number}] Extracting validation and test results from RF-DETR output")  
    

    rf_detr_restults_path = trial_path/ "results.json"  # Assuming RF-DETR saves a results.json with validation/test metrics
    validation_results, test_results = RFDETRExtractor.get_validation_and_test_results(rf_detr_restults_path)
    flattened_validation_results = RFDETRExtractor.flatten_dict(validation_results)
    flattened_test_results = RFDETRExtractor.flatten_dict(test_results)
    prefixed_validation_results = RFDETRExtractor.prefix_keys(flattened_validation_results, "VAL_")
    prefixed_test_results = RFDETRExtractor.prefix_keys(flattened_test_results, "TEST_")

    # calculate harmonic mean for optimization score
    harmonic_mean_values = [prefixed_test_results[f"TEST_{config.classes[2]}_map_50_95"], prefixed_validation_results[f"VAL_{config.classes[2]}_map_50_95"]]

    score = len(harmonic_mean_values) / sum(1.0 / v for v in harmonic_mean_values if v > 0) if all(v > 0 for v in harmonic_mean_values) else 0.0


    logger.info(f"[Trial {trial.number}] Optimization score calculated: {score:.5f}")
    
    combined_metrics = {}
    combined_metrics[best_epoch] = best_epoch
    combined_metrics.update(prefixed_validation_results)
    combined_metrics.update(prefixed_test_results)
    combined_metrics.update({"score": score})
    # ========================== LOGGING ==========================
    logger.info(f"[Trial {trial.number}] Saving results")
    
    DataLogger.save_to_json(
        output_json_path=config.paths["output_json"],
        trial_number=trial.number,
        params=combined_params,
        metrics=combined_metrics
    )
    
    DataLogger.save_to_csv(
        file_path=config.paths["output_csv"],
        data_dict={**combined_params, **combined_metrics}
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
        logger.info(f"Starting optimization loop (1 trial)")
        study.optimize(
            lambda trial: objective(trial, config),
            n_trials=1,
            n_jobs=1
        )
        logger.info("Optimization completed successfully")
    
    except KeyboardInterrupt:
        logger.info('Ctrl+c detected!')

    
    except Exception as e:
        logger.error(f"Trial failed: {e}")

if __name__ == '__main__':
    main()
