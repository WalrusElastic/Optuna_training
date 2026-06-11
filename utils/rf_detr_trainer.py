"""
RF-DETR model training and evaluation wrapper.

Single Responsibility: Train RF-DETR model and extract metrics.
"""

import logging
import json
import time
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
from rfdetr import RFDETRNano

from .rf_detr_extract_utils import RFDETRExtractor
from .rf_detr_prediction_utils import RFDETRPredictor
from .yolo_evaluation_utils import YOLOBBSEvaluator

logger = logging.getLogger(__name__)


class RFDETRTrainer:
    """
    Handles RF-DETR model training, validation, and test evaluation.
    
    Single Responsibility: Training orchestration only (no config, no preprocessing).
    """

    @staticmethod
    def train(
        trial_number: int,
        trial_path: Path,
        training_params: Dict[str, Any],
        config: Any,
    ) -> float:
        """
        Train RF-DETR model and return optimization score.
        
        Pipeline:
        1. Save training parameters to JSON
        2. Launch training subprocess
        3. Extract validation metrics
        4. Run test evaluation
        5. Calculate and return score
        
        Args:
            trial_number: Optuna trial number
            trial_path: Path to save trial outputs
            training_params: Training parameters (from config)
            config: TrainingConfig object (for paths, classes, etc.)
            
        Returns:
            Optimization score (float)
            
        Raises:
            RuntimeError: If training subprocess fails
        """
        logger.info(f"[Trial {trial_number}] Starting RF-DETR training")
        
        trial_path.mkdir(parents=True, exist_ok=True)
        training_params["output_dir"] = str(trial_path)

        # ========================== TRAINING ==========================
        
        # Save parameters for worker script
        params_json_path = config.paths["params_json"]
        if params_json_path.exists():
            params_json_path.unlink()
        with open(params_json_path, "w") as f:
            json.dump(training_params, f, indent=2)

        logger.info(f"[Trial {trial_number}] Launching training subprocess")
        training_start_time = time.time()

        RFDETRTrainer._launch_training(
            trial_number,
            config.paths["training_worker_script"],
            config.paths["pretrained_model_weights"],
            params_json_path,
            config.paths["root"],
            config.rfdetr_dataset.get("num_gpus", 1),
        )

        training_end_time = time.time()
        training_time = training_end_time - training_start_time
        logger.info(f"[Trial {trial_number}] Training completed in {training_time:.2f}s")

        # ========================== EXTRACT VALIDATION METRICS ==========================
        
        logger.info(f"[Trial {trial_number}] Extracting validation metrics")
        
        rf_detr_val_metrics_path = trial_path / "metrics.csv"
        if not rf_detr_val_metrics_path.exists():
            logger.warning(f"[Trial {trial_number}] metrics.csv not found")
            return 0.0

        best_epoch = RFDETRExtractor.get_best_epoch(
            rf_detr_val_metrics_path,
            patience=training_params.get("early_stopping_patience", 10),
        )
        logger.info(f"[Trial {trial_number}] Best epoch: {best_epoch}")

        validation_results = RFDETRExtractor.get_validation_results(
            rf_detr_val_metrics_path, best_epoch
        )

        # ========================== TEST EVALUATION ==========================
        
        logger.info(f"[Trial {trial_number}] Running test evaluation")
        
        test_predictions_dir = trial_path / "test_predictions"
        test_predictions_dir.mkdir(parents=True, exist_ok=True)

        model_path = trial_path / "checkpoint_best_total.pth"
        if not model_path.exists():
            logger.error(f"[Trial {trial_number}] Model checkpoint not found: {model_path}")
            return 0.0

        model = RFDETRNano(pretrain_weights=str(model_path))

        test_threshold = config.rfdetr_dataset.get("test_threshold", 0.03)
        RFDETRPredictor.predict_image_dir_and_save_bbs_labels(
            model=model,
            classes=config.classes,
            input_dir=config.paths["final_dataset"] / "test" / "images",
            pred_labels_dir=test_predictions_dir,
            threshold=test_threshold,
            include_confidence=True,
        )

        test_results, _, _ = YOLOBBSEvaluator.evaluate(
            pred_dir=test_predictions_dir,
            gt_dir=config.paths["final_dataset"] / "test" / "labels",
            classes=config.classes,
        )

        # ========================== CALCULATE SCORE ==========================
        
        logger.info(f"[Trial {trial_number}] Calculating optimization score")
        
        target_class_idx = config.optimization_target_class or 0
        if target_class_idx >= len(config.classes):
            target_class_idx = 0
        target_class = config.classes[target_class_idx]

        val_ap = validation_results.get(f"val/AP/{target_class}", 0.0)
        test_map = test_results.get(f"{target_class}_map_50_95", 0.0)

        # Harmonic mean of val AP and test mAP
        if val_ap > 0 and test_map > 0:
            score = 2.0 / (1.0 / val_ap + 1.0 / test_map)
        else:
            score = max(val_ap, test_map)

        logger.info(f"[Trial {trial_number}] Score: {score:.5f}")

        # Return consolidated results
        return {
            "score": score,
            "best_epoch": best_epoch,
            "validation_results": validation_results,
            "test_results": test_results,
            "training_time": training_time,
        }

    @staticmethod
    def _launch_training(
        trial_number: int,
        worker_script: Path,
        pretrained_weights: Path,
        params_json: Path,
        root_dir: Path,
        num_gpus: int,
    ) -> None:
        """
        Launch training subprocess (handles OS-specific distributed setup).
        
        Args:
            trial_number: Trial number for logging
            worker_script: Path to training worker script
            pretrained_weights: Path to pretrained model
            params_json: Path to training parameters JSON
            root_dir: Working directory for subprocess
            num_gpus: Number of GPUs for distributed training
            
        Raises:
            RuntimeError: If subprocess fails
        """
        if sys.platform == "win32":
            logger.info(f"[Trial {trial_number}] Running on Windows (no torch.distributed)")
            launch_cmd = [
                sys.executable,
                str(worker_script),
                "--pretrain-weights",
                str(pretrained_weights),
                "--params-json",
                str(params_json),
            ]
        else:
            logger.info(f"[Trial {trial_number}] Running on Linux with {num_gpus} GPU(s)")
            launch_cmd = [
                sys.executable,
                "-m",
                "torch.distributed.run",
                f"--nproc_per_node={num_gpus}",
                str(worker_script),
                "--pretrain-weights",
                str(pretrained_weights),
                "--params-json",
                str(params_json),
            ]

        logger.info(f"[Trial {trial_number}] Launching: {' '.join(launch_cmd)}")
        proc = subprocess.run(launch_cmd, cwd=str(root_dir))

        if proc.returncode != 0:
            raise RuntimeError(
                f"Training subprocess failed with return code {proc.returncode}. "
                "Check console output above for details."
            )
