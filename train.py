"""
Script to train YOLO on a dataset with Optuna-suggested parameters, within a folder.
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
import math
import random
from tqdm import tqdm
import shutil

import matplotlib.pyplot as plt
import numpy as np
import cv2
import yaml
import torch
import seaborn as sns
import albumentations as A

from configs import TrainingConfig
from utils.preprocessing_utils import PreprocessingUtils
from utils.yolo_dataset_utils import YOLOWeightedDataset
from utils.evaluation_utils import YOLOSegmentationEvaluator
from utils.extract_yolo_data_utils import YOLODataExtractor
from utils.optuna_utils import OptunaTrialManager

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))
os.environ["ALBUMENTATIONS_DISABLE"] = "1"
import ultralytics
from ultralytics import YOLO
import ultralytics.data.build as build

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

logger.info(f"CUDA Available: {torch.cuda.is_available()}")

# ========================== DATA AUGMENTATION ==========================

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

    for split in ['train', 'val', 'test']:
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

# ========================== SETTING UP YOLO YAML ==========================
    
def setup_yaml(yaml_path: Path, dataset_path: Path, classes) -> None:
    """Create data.yaml for YOLO."""
    logger.info(f"Setting up YOLO data.yaml at {yaml_path}")
    data_yaml = {
        "path": str(dataset_path),
        "train": "train",
        "val": "val",
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
    test_path = config.paths["test_dataset"]
    
    logger.info(f"[Trial {trial.number}] Starting trial...")
    
    # ========================== PREPROCESSING ==========================
    logger.info(f"[Trial {trial.number}] Starting preprocessing phase")
    additional_parameters = config.additional_parameters

    # Insert parameters you wish to optimize here
    brightness = additional_parameters["brightness"]
    contrast = additional_parameters["contrast"]
    sharpness = additional_parameters["sharpness"]
    # Insert parameters you wish to optimize here
    # brightness = trial.suggest_float("brightness", 0.2, 0.5)
    # additional_parameters["brightness"] = brightness

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
        transforms, config.paths["split_dataset"], config.paths["final_dataset"],
        input_size=512, output_size=1024
    )
    
    # ========================== TRAINING ==========================
    logger.info(f"[Trial {trial.number}] Starting training phase")
    build.YOLODataset = YOLOWeightedDataset
    model = YOLO(str(config.paths["default_model_weights"]))
    logger.info(f"[Trial {trial.number}] Model loaded from {config.paths['default_model_weights']}")
    
    params = config.yolo_parameters
    params["name"] = f'trial_{trial.number}'
    # Insert parameters you wish to optimize here
    # mosaic = trial.suggest_float("mosaic", [0.0, 1.0])
    # params["mosaic"] = mosaic

    try:
        logger.info(f"[Trial {trial.number}] Starting YOLO training with {config.yolo_parameters['epochs']} epochs")
        results = model.train(**params)
        logger.info(f"[Trial {trial.number}] Training completed successfully")
    except RuntimeError:
        logger.warning(f"[Trial {trial.number}] Runtime error during training, suspecting tensor error during final validation due to large batch size, continuing...")

    # ========================== VALIDATION METRICS ==========================
    logger.info(f"[Trial {trial.number}] Extracting validation metrics")
    yolo_results_csv = trial_path / "results.csv"
    df = pd.read_csv(yolo_results_csv)
    
    fitness_scores = 0.1 * df["metrics/mAP50(B)"] + 0.9 * df["metrics/mAP50-95(B)"]
    best_idx = fitness_scores.idxmax()
    best_row = df.iloc[best_idx]
    logger.info(f"[Trial {trial.number}] Best validation epoch: {int(best_row['epoch'])}")
    
    additional_metrics = {
        'best_epoch': best_row["epoch"],
        'Val_P': best_row["metrics/precision(B)"],
        'Val_R': best_row["metrics/recall(B)"],
        'Val_MAP_50': best_row['metrics/mAP50(B)'],
        'Val_MAP_50-95': best_row['metrics/mAP50-95(B)'],
    }

    # Extract per-class validation metrics
    logger.info(f"[Trial {trial.number}] Loading best model for per-class metrics")
    best_pt = trial_path / 'weights' / 'best.pt'
    model = YOLO(best_pt)
    results = model.val(device=0, batch=1, data=config.paths["yolo_yaml"])
    for index in range(len(config.classes)):
        additional_metrics[f"Val_{config.classes[index]}_P"] = round(float(results.box.class_result(index)[0]), 5)
        additional_metrics[f"Val_{config.classes[index]}_R"] = round(float(results.box.class_result(index)[1]), 5)
        additional_metrics[f"Val_{config.classes[index]}_MAP_50"] = round(float(results.box.class_result(index)[2]), 5)
        additional_metrics[f"Val_{config.classes[index]}_MAP_50-95"] = round(float(results.box.class_result(index)[3]), 5)

    # Score for Optuna optimization (using mAP50-95 of the third class as primary metric)
    score = round(float(results.box.class_result(2)[3]), 5)
    logger.info(f"[Trial {trial.number}] Optimization score calculated: {score:.5f}")
    
    # Clean up unnecessary weights
    last_pt = trial_path / 'weights' / 'last.pt'
    if last_pt.exists():
        logger.info(f"[Trial {trial.number}] Removing unnecessary last.pt weights file")
        last_pt.unlink()
    
    # ========================== TEST METRICS ==========================
    logger.info(f"[Trial {trial.number}] Testing on test dataset")
    best_model_path = trial_path / "weights" / "best.pt"
    model = YOLO(str(best_model_path))
    
    pred_dir = trial_path / "test_results" / "labels"
    if pred_dir.exists():
        shutil.rmtree(pred_dir.parent)

    logger.info(f"[Trial {trial.number}] Running predictions on test dataset")
    model.predict(
        source=str(test_path / "images"),
        verbose=False,
        project=str(trial_path),
        save_txt=True,
        name="test_results",
        conf=0.1,
    )
    logger.info(f"[Trial {trial.number}] Predictions completed")

    logger.info(f"[Trial {trial.number}] Evaluating segmentation metrics on test set")
    test_metrics, all_true, all_pred = YOLOSegmentationEvaluator.evaluate_yolo_seg(
        pred_dir, test_path / "labels", config.classes, iou_threshold=0.3
    )
    logger.info(f"[Trial {trial.number}] Creating confusion matrix")
    YOLOSegmentationEvaluator.create_confusion_matrix(
        all_true, all_pred, trial_path / "test_results" / "confusion_matrix.png", config.classes
    )
    logger.info(f"[Trial {trial.number}] Test evaluation completed")
    
    # ========================== LOGGING ==========================
    logger.info(f"[Trial {trial.number}] Extracting loss graphs and saving results")
    YOLODataExtractor.extract_loss_graphs(trial_path)

    logger.info(f"[Trial {trial.number}] Saving results to JSON")
    YOLODataExtractor.save_results_to_json(
        config.paths["output_json"],
        trial.number,
        params,
        additional_parameters,
        additional_metrics,
        test_metrics,
    )

    row_to_write = {"path": str(trial_path)}
    row_to_write.update(params)
    row_to_write.update(config.additional_parameters)
    row_to_write.update(additional_metrics)
    row_to_write.update(test_metrics)
    
    logger.info(f"[Trial {trial.number}] Saving results to CSV")
    YOLODataExtractor.append_to_csv(config.paths["output_csv"], row_to_write)

    logger.info(f"[Trial {trial.number}] Saving Optuna data to JSON")
    OptunaTrialManager.save_trial_to_json(trial, config.paths["optuna_json"], score)
    
    logger.info(f"[Trial {trial.number}] Completed - Score: {score:.4f}")
    
    return score


def main():
    """Main entry point."""
    logger.info("="*60)
    logger.info("Starting Optuna-based YOLO training optimization")
    logger.info("="*60)
    
    config = TrainingConfig()
    logger.info(f"Configuration loaded - Study name: {config.study_name}")
    logger.info(f"Classes: {config.classes}")
    
    logger.info("Setting up YOLO configuration file")
    setup_yaml(config.paths["yolo_yaml"], config.paths["final_dataset"], config.classes)
    
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
        logger.info('Ctrl+c detected! Saving plots...')
        his_fig = plot_optimization_history(study)
        his_fig.show()
        
        fig = plot_param_importances(study)
        fig.show()
    
    except RuntimeError as e:
        logger.error(f"Caught YOLO tensor issue: {e}")
        logger.info("Restarting...")
        main()
    
    except Exception as e:
        logger.error(f"Trial failed: {e}")



if __name__ == '__main__':
    main()

