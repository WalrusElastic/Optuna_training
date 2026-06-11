"""
Command-line argument parsing for RF-DETR training.

Single Responsibility: Parse and apply CLI overrides to config.
"""

import argparse
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class TrainingCLI:
    """
    Handles command-line argument parsing and config overrides.
    
    Single Responsibility: Parse CLI args and apply to config only.
    """

    @staticmethod
    def parse_args() -> argparse.Namespace:
        """
        Parse command-line arguments.
        
        Returns:
            Namespace with parsed arguments
        """
        parser = argparse.ArgumentParser(
            description="RF-DETR training with Optuna optimization",
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )

        parser.add_argument(
            "--config",
            type=str,
            default="config.yaml",
            help="Path to config.yaml file (default: config.yaml)",
        )

        parser.add_argument(
            "--trials",
            type=int,
            help="Number of Optuna trials to run (overrides config)",
        )

        parser.add_argument(
            "--device",
            type=int,
            help="GPU device ID (sets num_gpus=1 and devices=device_id)",
        )

        parser.add_argument(
            "--batch",
            type=int,
            help="Batch size (overrides config)",
        )

        parser.add_argument(
            "--epochs",
            type=int,
            help="Number of epochs (overrides config)",
        )

        parser.add_argument(
            "--target-class",
            type=int,
            help="Class index for optimization metric (overrides config)",
        )

        parser.add_argument(
            "--lr",
            type=float,
            help="Learning rate (overrides config)",
        )

        parser.add_argument(
            "--resume",
            action="store_true",
            help="Resume from existing Optuna study",
        )

        parser.add_argument(
            "--study-name",
            type=str,
            help="Custom study name (overrides config)",
        )

        parser.add_argument(
            "--no-augment",
            action="store_true",
            help="Disable preprocessing augmentation",
        )

        parser.add_argument(
            "--verbose",
            action="store_true",
            help="Enable verbose logging",
        )

        return parser.parse_args()

    @staticmethod
    def apply_overrides(config: Any, args: argparse.Namespace) -> None:
        """
        Apply command-line overrides to config object.
        
        Args:
            config: TrainingConfig object to modify in-place
            args: Parsed arguments from parse_args()
        """
        if args.trials is not None:
            config.optuna_parameters["n_trials"] = args.trials
            logger.info(f"Override: n_trials = {args.trials}")

        if args.epochs is not None:
            config.rfdetr_parameters["epochs"] = args.epochs
            logger.info(f"Override: epochs = {args.epochs}")

        if args.batch is not None:
            config.rfdetr_parameters["batch_size"] = args.batch
            logger.info(f"Override: batch_size = {args.batch}")

        if args.lr is not None:
            config.rfdetr_parameters["lr"] = args.lr
            logger.info(f"Override: lr = {args.lr}")

        if args.device is not None:
            config.rfdetr_dataset["num_gpus"] = 1
            config.rfdetr_parameters["devices"] = args.device
            logger.info(f"Override: devices = {args.device}, num_gpus = 1")

        if args.target_class is not None:
            config.optimization_target_class = args.target_class
            logger.info(f"Override: optimization_target_class = {args.target_class}")

        if args.study_name is not None:
            config.study_name = args.study_name
            logger.info(f"Override: study_name = {args.study_name}")

        if args.no_augment:
            for key in config.preprocessing_config:
                if key in ["brightness", "contrast", "sharpness"]:
                    config.preprocessing_config[key] = 0.0
            logger.info("Override: augmentation disabled")

        if args.verbose:
            logging.getLogger().setLevel(logging.DEBUG)
            logger.info("Override: verbose logging enabled")
