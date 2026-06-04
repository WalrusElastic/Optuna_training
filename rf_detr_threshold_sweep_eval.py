"""
Evaluate an RF-DETR checkpoint on validation and test splits across confidence thresholds.

Predictions are run once per split at a low threshold with confidence scores saved to disk.
The threshold sweep then re-evaluates saved predictions by filtering on confidence, avoiding
repeated model inference for every threshold step.
"""

import logging
import os
from pathlib import Path
from typing import Dict, List

from rfdetr import RFDETRNano

from utils.data_logging_utils import DataLogger
from utils.rf_detr_prediction_utils import RFDETRPredictor
from utils.yolo_evaluation_utils import YOLOBBSEvaluator


logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ========================== USER CONFIG ==========================
CONFIG = {
    "root": Path(os.path.dirname(os.path.realpath(__file__))),
    "weights_path": r"INPUT_PATH",
    "val_images_dir": r"INPUT_PATH",
    "val_labels_dir": r"INPUT_PATH",
    "test_images_dir": r"INPUT_PATH",
    "test_labels_dir": r"INPUT_PATH",
    "output_csv": r"INPUT_PATH",
    "prediction_output_root": r"INPUT_PATH",
    # Required: provide class names in index order.
    "classes": ["Class_1", "Class_2", "Class_3", "Class_4"],
    "channels": 1,
    "iou_threshold": 0.5,
    # Low threshold used during inference to capture all detections; confidence scores are
    # saved alongside each prediction so the sweep can filter without re-running the model.
    "prediction_threshold": 0.01,
    "threshold_start": 0.05,
    "threshold_end": 0.50,
    "threshold_step": 0.05,
}


def _prefix_metrics(metrics: Dict, split_name: str) -> Dict:
    """Return a copy of metrics with every key prefixed by the split name."""
    prefix = split_name.upper()
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def _predict_split(
    model,
    classes: List[str],
    image_dir: Path,
    pred_labels_dir: Path,
    prediction_threshold: float,
    channels: int,
) -> None:
    """Run inference once and write labels with per-detection confidence scores."""
    logger.info("Running predictions for split: %s", image_dir)
    RFDETRPredictor.predict_image_dir_and_save_bbs_labels(
        model=model,
        classes=classes,
        input_dir=image_dir,
        pred_labels_dir=pred_labels_dir,
        threshold=prediction_threshold,
        channels=channels,
        include_confidence=True,
    )


def _evaluate_split_at_threshold(
    split_name: str,
    pred_labels_dir: Path,
    label_dir: Path,
    classes: List[str],
    confidence_threshold: float,
    iou_threshold: float,
) -> Dict:
    """Filter saved predictions by confidence threshold and compute metrics."""
    split_prefix = split_name.upper()
    split_row: Dict = {
        f"{split_prefix}_status": "ok",
        f"{split_prefix}_error": "",
    }

    try:
        metrics, _, _ = YOLOBBSEvaluator.evaluate(
            pred_dir=pred_labels_dir,
            gt_dir=label_dir,
            classes=classes,
            iou_threshold=iou_threshold,
            confidence_threshold=confidence_threshold,
        )
        split_row.update(_prefix_metrics(metrics, split_name))
    except Exception as exc:
        split_row[f"{split_prefix}_status"] = "error"
        split_row[f"{split_prefix}_error"] = str(exc)
        logger.exception("[%s] Failed at confidence %.2f", split_prefix, confidence_threshold)

    return split_row


def main() -> None:
    # 1) Resolve class list.
    classes_raw = CONFIG["classes"]
    if isinstance(classes_raw, list):
        classes = classes_raw
    elif isinstance(classes_raw, str):
        classes = [token.strip() for token in classes_raw.split(",") if token.strip()]
    else:
        raise TypeError("CONFIG['classes'] must be a comma-separated string or a list of class names.")
    if not classes:
        raise ValueError("CONFIG['classes'] cannot be empty.")

    # 2) Resolve paths and scalars.
    weights_path = Path(CONFIG["weights_path"])
    val_images_dir = Path(CONFIG["val_images_dir"])
    val_labels_dir = Path(CONFIG["val_labels_dir"])
    test_images_dir = Path(CONFIG["test_images_dir"])
    test_labels_dir = Path(CONFIG["test_labels_dir"])
    output_csv = Path(CONFIG["output_csv"])
    prediction_output_root = Path(CONFIG["prediction_output_root"])
    iou_threshold = float(CONFIG["iou_threshold"])
    channels = int(CONFIG["channels"])
    prediction_threshold = float(CONFIG["prediction_threshold"])

    # 3) Build confidence sweep using integer hundredths to avoid float drift.
    threshold_start = float(CONFIG["threshold_start"])
    threshold_end = float(CONFIG["threshold_end"])
    threshold_step = float(CONFIG["threshold_step"])
    if threshold_step <= 0:
        raise ValueError("threshold_step must be > 0")
    if threshold_end < threshold_start:
        raise ValueError("threshold_end must be >= threshold_start")

    start_i = int(round(threshold_start * 100))
    end_i = int(round(threshold_end * 100))
    step_i = int(round(threshold_step * 100))
    if step_i <= 0:
        raise ValueError("threshold_step resolution must be at least 0.01")
    thresholds = [v / 100 for v in range(start_i, end_i + 1, step_i)]

    # 4) Fail fast if any required input path is missing.
    for path in [weights_path, val_images_dir, val_labels_dir, test_images_dir, test_labels_dir]:
        if not path.exists():
            raise FileNotFoundError(f"Required path does not exist: {path}")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    prediction_output_root.mkdir(parents=True, exist_ok=True)

    # 5) Load model.
    logger.info("Loading RF-DETR checkpoint from: %s", weights_path)
    model = RFDETRNano(pretrain_weights=str(weights_path))

    # 6) Run inference once per split and save labels with confidence scores.
    val_pred_dir = prediction_output_root / "val" / "labels"
    test_pred_dir = prediction_output_root / "test" / "labels"

    _predict_split(model, classes, val_images_dir, val_pred_dir, prediction_threshold, channels)
    _predict_split(model, classes, test_images_dir, test_pred_dir, prediction_threshold, channels)

    # 7) Sweep over confidence thresholds — no model needed, just filter saved labels.
    for threshold in thresholds:
        logger.info("Evaluating confidence threshold %.2f", threshold)

        row: Dict = {
            "threshold": threshold,
            "iou_threshold": iou_threshold,
            "status": "ok",
            "error": "",
        }

        val_row = _evaluate_split_at_threshold(
            split_name="val",
            pred_labels_dir=val_pred_dir,
            label_dir=val_labels_dir,
            classes=classes,
            confidence_threshold=threshold,
            iou_threshold=iou_threshold,
        )
        test_row = _evaluate_split_at_threshold(
            split_name="test",
            pred_labels_dir=test_pred_dir,
            label_dir=test_labels_dir,
            classes=classes,
            confidence_threshold=threshold,
            iou_threshold=iou_threshold,
        )

        row.update(val_row)
        row.update(test_row)

        if "error" in [row.get("VAL_status"), row.get("TEST_status")]:
            row["status"] = "error"
            errors = [msg for msg in [row.get("VAL_error", ""), row.get("TEST_error", "")] if msg]
            row["error"] = " | ".join(errors)

        DataLogger.save_to_csv(output_csv, row)

    logger.info("Threshold sweep complete. Results appended to: %s", output_csv)


if __name__ == "__main__":
    main()
