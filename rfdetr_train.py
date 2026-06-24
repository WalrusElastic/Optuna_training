"""
Script to train RF-DETR on a dataset with Optuna-suggested parameters, within a folder.
Updates augmentation and final dataset folders for each trial and trains the model.
"""

import json
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import albumentations as A
import optuna
import torch
import yaml
from tqdm import tqdm
from rfdetr import RFDETRNano

from meta import __version__
from utils.config_loader import ConfigLoader, TrainingConfig
from utils.rf_detr_extract_utils import RFDETRExtractor
from utils.preprocessing_utils import PreprocessingUtils
from utils.data_logging_utils import DataLogger
from utils.optuna_utils import OptunaTrialManager
from utils.rf_detr_prediction_utils import RFDETRPredictor
from utils.yolo_evaluation_utils import YOLOBBSEvaluator

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))
os.environ["ALBUMENTATIONS_DISABLE"] = "1"
os.environ["USE_LIBUV"] = "0"  # Disable libuv to prevent subprocess issues on Windows

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s',
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


# ========================== SETTING UP RFDETR YAML ==========================

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
        "names": {index: name for index, name in enumerate(classes)},
    }

    with open(yaml_path, 'w') as f:
        yaml.safe_dump(data_yaml, f)

    logger.info(f"YOLO data.yaml created with {len(classes)} classes")


# ========================== MAIN OBJECTIVE FUNCTION ==========================

def objective(trial: optuna.trial.Trial, config: TrainingConfig) -> float:
    """
    Optuna objective: run one full RF-DETR trial and return its optimization score.

    Pipeline (per trial):
        1. Trial setup   - overlay this trial's Optuna suggestions onto a per-trial config.
        2. Preprocessing - augment the split dataset into a YOLO-format final dataset.
        3. Training      - launch the RF-DETR worker subprocess on the prepared data.
        4. Validation    - extract the best-epoch validation metrics from training output.
        5. Testing       - run inference + evaluation on the held-out test set.
        6. Scoring       - combine validation/test metrics into the objective score.
        7. Logging       - persist params + metrics to JSON/CSV and the Optuna store.

    Args:
        trial: Optuna trial, providing this run's hyperparameter suggestions.
        config: Base TrainingConfig loaded from config.py.

    Returns:
        The optimization score for this trial (higher is better).
    """
    logger.info(f"[Trial {trial.number}] {'=' * 16} Starting trial {trial.number} {'=' * 16}")

    # ============================= 1. TRIAL SETUP =============================
    # Overlay this trial's Optuna suggestions onto a per-trial copy of the config,
    # then pull out the sections the body uses (single source of truth for the trial).
    trial_config, suggested_params = config.build_trial_config(trial)
    if suggested_params:
        logger.info(f"[Trial {trial.number}] Optuna suggestions this trial: {suggested_params}")

    paths = config.paths
    classes = config.study["classes"]
    trial_path = paths["runs_dir"] / f'trial_{trial.number}'

    training_params = dict(trial_config.rfdetr_parameters)
    training_params["output_dir"] = str(trial_path)
    preprocessing_config = trial_config.preprocessing_config
    rfdetr_dataset = trial_config.rfdetr_dataset

    # ============================= 2. PREPROCESSING =============================
    # Build the (fixed-per-trial) augmentation pipeline, then write the YOLO-format
    # dataset and its data.yaml.
    logger.info(f"[Trial {trial.number}] Preprocessing: augmenting split dataset into final dataset")

    brightness = preprocessing_config["brightness"]
    contrast = preprocessing_config["contrast"]
    sharpness = preprocessing_config["sharpness"]

    transforms = A.Compose([
        A.RandomBrightnessContrast(
            brightness_limit=(brightness, brightness),
            contrast_limit=(contrast, contrast),
            brightness_by_max=False, p=1
        ),
        A.Sharpen(alpha=(sharpness, sharpness), lightness=(0.8, 0.8), p=1.0),
    ])

    augment_and_prepare_final_dataset(
        transforms,
        paths["split_dataset"],
        paths["final_dataset"],
        input_size=preprocessing_config["input_size"],
        output_size=preprocessing_config["training_size"],
        edit_labels=preprocessing_config["edit_labels"],
    )
    setup_yaml(
        paths["data_yaml"],
        paths["final_dataset"],
        classes,
    )

    # ============================= 3. TRAINING =============================
    # Serialize the resolved training params, then launch the RF-DETR worker subprocess.
    logger.info(f"[Trial {trial.number}] Training: {training_params['epochs']} epochs on {rfdetr_dataset['num_gpus']} GPU(s)")

    trial_path.mkdir(parents=True, exist_ok=True)

    # dataset_dir is pipeline-determined: the augmented dataset is written to final_dataset.
    training_params["dataset_dir"] = str(paths["final_dataset"])

    # Hand the params to the worker subprocess via JSON.
    params_json_path = paths["params_json"]
    if params_json_path.exists():
        logger.info(f"[Trial {trial.number}] Removing existing params JSON at {params_json_path}")
        params_json_path.unlink()
    with open(params_json_path, "w") as f:
        json.dump(training_params, f, indent=2)

    # Launch training subprocess (handles distributed setup internally based on OS).
    training_start_time = time.time()
    if sys.platform == "win32":
        logger.info(f"[Trial {trial.number}] Running on Windows: using direct subprocess (no torch.distributed)")
        launch_cmd = [
            sys.executable,
            str(paths["training_worker_script"]),
            "--pretrain-weights",
            str(paths["pretrained_model_weights"]),
            "--params-json",
            str(params_json_path),
        ]
        logger.info(f"[Trial {trial.number}] Launching training: {' '.join(launch_cmd)}")
        proc = subprocess.run(launch_cmd, cwd=str(paths["root"]))
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
            f"--nproc_per_node={rfdetr_dataset['num_gpus']}",
            str(paths["training_worker_script"]),
            "--pretrain-weights",
            str(paths["pretrained_model_weights"]),
            "--params-json",
            str(params_json_path),
        ]
        logger.info(f"[Trial {trial.number}] Launching distributed training: {' '.join(launch_cmd)}")
        proc = subprocess.run(launch_cmd, cwd=str(paths["root"]))
        if proc.returncode != 0:
            logger.error(f"[Trial {trial.number}] Training subprocess failed with code {proc.returncode}")
            raise RuntimeError(
                f"Training subprocess failed with return code {proc.returncode}. "
                "Check console output above for details."
            )
    logger.info(f"[Trial {trial.number}] Training completed successfully")
    training_end_time = time.time()
    training_time = training_end_time - training_start_time

    # ============================= 4. VALIDATION METRICS =============================
    # Find the best epoch (by early-stopping patience) and read its validation metrics.
    logger.info(f"[Trial {trial.number}] Validation: extracting best-epoch metrics from training output")
    rf_detr_val_metrics_path = trial_path / "metrics.csv"
    best_epoch = RFDETRExtractor.get_best_epoch(
        rf_detr_val_metrics_path,
        patience=training_params["early_stopping_patience"],
    )
    logger.info(f"[Trial {trial.number}] Best epoch identified: {best_epoch}")
    logger.info(f"[Trial {trial.number}] Final epoch identified: {best_epoch + training_params['early_stopping_patience']}")
    logger.info(f"[Trial {trial.number}] {best_epoch + training_params['early_stopping_patience']} epochs completed in {training_time:.2f} seconds, average time per epoch: {training_time / (best_epoch + training_params['early_stopping_patience']):.2f} seconds")

    validation_results = RFDETRExtractor.get_validation_results(
        rf_detr_val_metrics_path,
        best_epoch,
    )

    # ============================= 5. TEST-SET EVALUATION =============================
    # Load the best checkpoint, predict on the test set, then evaluate against ground truth.
    logger.info(f"[Trial {trial.number}] Testing: running inference + evaluation on the test set")
    test_predictions_dir = trial_path / "test_predictions"
    test_predictions_dir.mkdir(parents=True, exist_ok=True)
    model_path = trial_path / "checkpoint_best_total.pth"
    model = RFDETRNano(pretrain_weights=str(model_path))
    RFDETRPredictor.predict_image_dir_and_save_bbs_labels(
        model=model,
        classes=classes,
        input_dir=paths["final_dataset"] / "test" / "images",
        pred_labels_dir=test_predictions_dir,
        threshold=rfdetr_dataset["test_threshold"],  # low threshold to capture more predictions for mAP50-95
        include_confidence=True,
    )

    test_results, _, _ = YOLOBBSEvaluator.evaluate(
        pred_dir=test_predictions_dir,
        gt_dir=paths["final_dataset"] / "test" / "labels",
        classes=classes,
    )

    # ============================= 6. SCORING =============================
    # Objective = harmonic-style combination of validation AP and test mAP for the target class.
    target_idx = config.study["optimization_target_class"] or 0
    target_class = classes[target_idx]
    score = 1 / (1 / validation_results[f"val/AP/{target_class}"] + 1 / test_results[f"{target_class}_map_50"])

    logger.info(f"[Trial {trial.number}] Optimization score for '{target_class}': {score:.5f}")

    # ============================= 7. LOGGING =============================
    # Assemble everything that gets persisted (params + metrics), in one place.
    combined_params = {**training_params, **preprocessing_config, **suggested_params}

    combined_metrics = {"best_epoch": best_epoch}
    combined_metrics.update(validation_results)
    combined_metrics.update(test_results)
    combined_metrics["score"] = score

    # Convert numpy / WindowsPath / other non-JSON types so results can be serialized.
    corrected_combined_params = DataLogger.make_json_safe(combined_params)
    corrected_combined_metrics = DataLogger.make_json_safe(combined_metrics)

    logger.info(f"[Trial {trial.number}] Saving results to JSON/CSV and the Optuna store")

    DataLogger.save_to_json(
        output_json_path=paths["output_json"],
        trial_number=trial.number,
        params=corrected_combined_params,
        metrics=corrected_combined_metrics,
    )

    DataLogger.save_to_csv(
        file_path=paths["output_csv"],
        data_dict={**corrected_combined_params, **corrected_combined_metrics},
    )

    OptunaTrialManager.save_trial_to_json(trial, paths["optuna_json"], score)

    logger.info(f"[Trial {trial.number}] {'=' * 16} Completed trial {trial.number} - score {score:.4f} {'=' * 16}")

    return score


def main():
    """Main entry point."""
    logger.info("=" * 60)
    logger.info(f"Optuna-based RF-DETR training optimization (v{__version__})")
    logger.info("=" * 60)

    config_path = Path(__file__).parent / "config.py"
    config = ConfigLoader.load(config_path)
    logger.info(f"Configuration loaded from {config_path} - Study name: {config.study['name']}")
    logger.info(f"Classes: {config.study['classes']}")

    try:
        logger.info("Creating or loading Optuna study")
        study = OptunaTrialManager.create_study_from_json(config.paths["optuna_json"], config.study["name"])
        logger.info(f"Starting optimization loop ({config.optuna['n_trials']} trials)")
        study.optimize(
            lambda trial: objective(trial, config),
            n_trials=config.optuna["n_trials"],
            n_jobs=config.optuna["n_jobs"],
        )
        logger.info("Optimization completed successfully")
    except KeyboardInterrupt:
        logger.info('Ctrl+c detected!')
    except Exception as e:
        logger.error(f"Trial failed: {e}")


if __name__ == '__main__':
    main()
