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
- utils.preprocessing_utils: Dataset augmentation
- utils.data_logging_utils: Results logging
- utils.optuna_utils: Optuna persistence
- utils.rf_detr_trainer.py: Model training
- preparetraining.py: Trial preparation (dataset + params)

"""

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
from utils.data_logging_utils import DataLogger
from utils.optuna_utils import OptunaTrialManager
from utils.rf_detr_trainer import RFDETRTrainer
from preparetraining import PrepareTraining

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
        training_params, all_params = PrepareTraining.prepare_trial(trial, config)
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

        # Convert numpy, WindowsPath types for JSON serialization
        corrected_metrics = DataLogger.make_json_safe(all_metrics)
        corrected_params = DataLogger.make_json_safe(all_params)

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
    # Parse CLI arguments
    args = TrainingCLI.parse_args()

    logger.info("=" * 70)
    logger.info("Optuna-based RF-DETR Training Optimization")
    logger.info("For a list of available CLI arguments, run with --help")
    logger.info("=" * 70)

    # Load config from YAML
    logger.info(f"Loading config from {args.config}")
    config = ConfigLoader.load(args.config)

    # Apply CLI overrides
    TrainingCLI.apply_overrides(config, args)

    logger.info(f"Study: {config.study_name}")
    logger.info(f"Classes: {config.classes}")

    # Create or load Optuna study
    logger.info("Setting up Optuna study")
    study = OptunaTrialManager.create_study_from_json(
        config.paths["optuna_json"],
        config.study_name,
    )

    # Run optimization
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
