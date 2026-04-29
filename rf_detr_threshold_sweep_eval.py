"""
Evaluate an RF-DETR checkpoint on validation and test splits across confidence thresholds.

This script runs prediction for each threshold in [0.05, 0.50] (step 0.05), computes
overall + per-class precision/recall metrics, and appends one row per threshold
that contains both validation and test metrics.
"""

import logging
import os
import os
from pathlib import Path
from typing import Dict, List, Tuple

from rfdetr import RFDETRNano

from utils.data_logging_utils import DataLogger
from utils.rf_detr_prediction_utils import RFDETRPredictor
from utils.yolo_evaluation_utils import YOLOBBSEvaluator


logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ========================== USER CONFIG ==========================
# Set these values directly before running the script.
CONFIG = {
    "root": Path(os.path.dirname(os.path.realpath(__file__))),
    "weights_path": Path(r"C:\Users\chenk_lsttb06\OneDrive\Documents\scvu stuff\rf_detr_testing\runs\trial_0\checkpoint_best_total.pth"),
    "val_images_dir": Path(r"C:\Users\chenk_lsttb06\OneDrive\Documents\scvu stuff\rf_detr_testing\Final_dataset\valid\images"),
    "val_labels_dir": Path(r"C:\Users\chenk_lsttb06\OneDrive\Documents\scvu stuff\rf_detr_testing\Final_dataset\valid\labels"),
    "test_images_dir": Path(r"C:\Users\chenk_lsttb06\OneDrive\Documents\scvu stuff\rf_detr_testing\Final_dataset/test/images"),
    "test_labels_dir": Path(r"C:\Users\chenk_lsttb06\OneDrive\Documents\scvu stuff\rf_detr_testing\Final_dataset/test/labels"),
    "output_csv": Path(r"C:\Users\chenk_lsttb06\OneDrive\Documents\scvu stuff\rf_detr_testing\runs\trial_0\comprehensive_threshold_sweep_results.csv"),
    "prediction_output_root": Path(r"C:\Users\chenk_lsttb06\OneDrive\Documents\scvu stuff\rf_detr_testing\threshold_eval_outputs"),
    # Required: provide class names in index order.
    "classes": ["Class_1", "Class_2", "Class_3", "Class_4"],
    "channels": 1,
    "iou_threshold": 0.5,
    "threshold_start": 0.05,
    "threshold_end": 0.50,
    "threshold_step": 0.05,
}


def _rename_metric_prefix(metrics: Dict, split_name: str) -> Dict:
    """Rename evaluator keys so CSV columns reflect the actual dataset split.

    The evaluator currently emits keys with a TEST_ prefix by default.
    When evaluating validation data, we rewrite TEST_* to VAL_* so logs are accurate
    and easy to filter downstream.
    """
    split_prefix = split_name.upper()
    renamed = {}

    for key, value in metrics.items():
        if key.startswith("TEST_"):
            renamed[f"{split_prefix}_{key[5:]}"] = value
        else:
            renamed[key] = value

    return renamed


# ========================== SPLIT EVALUATION ==========================
def _evaluate_split_at_threshold(
    model: RFDETRNano,
    classes: List[str],
    split_name: str,
    image_dir: Path,
    label_dir: Path,
    output_root: Path,
    threshold: float,
    iou_threshold: float,
    channels: int,
) -> Dict:
    """Evaluate one dataset split at one threshold and return row-ready metrics.

    Writes prediction labels, runs evaluation, and returns split-prefixed metrics.
    Any failure is captured in split-specific status/error fields.
    """
    split_output_dir = output_root / split_name
    split_output_dir.mkdir(parents=True, exist_ok=True)

    threshold_tag = f"{threshold:.2f}"
    run_dir = split_output_dir / f"thr_{threshold_tag}"
    pred_labels_dir = run_dir / "labels"
    split_prefix = split_name.upper()

    split_row = {
        f"{split_prefix}_status": "ok",
        f"{split_prefix}_error": "",
    }

    try:
        RFDETRPredictor.predict_image_dir_and_save_bbs_labels(
            model=model,
            classes=classes,
            input_dir=image_dir,
            pred_labels_dir=pred_labels_dir,
            threshold=threshold,
            channels=channels,
        )

        metrics, _, _ = YOLOBBSEvaluator.evaluate_yolo_seg(
            pred_dir=pred_labels_dir,
            gt_dir=label_dir,
            classes=classes,
            iou_threshold=iou_threshold,
        )
        split_row.update(_rename_metric_prefix(metrics, split_name))
    except Exception as exc:
        split_row[f"{split_prefix}_status"] = "error"
        split_row[f"{split_prefix}_error"] = str(exc)
        logger.exception(
            "[%s] Failed at threshold %.2f",
            split_name.upper(),
            threshold,
        )

    return split_row


def main() -> None:
    """Main execution flow.

    Sections in order:
    1) Resolve class list from CONFIG.
    2) Read and validate scalar config values.
    3) Build threshold sequence.
    4) Validate required input paths.
    5) Load model and run val/test sweeps.
    """
    # 1) Resolve classes.
    classes_raw = CONFIG["classes"]
    if isinstance(classes_raw, list):
        classes = classes_raw
    elif isinstance(classes_raw, str):
        classes = [token.strip() for token in classes_raw.split(",") if token.strip()]
    else:
        raise TypeError("CONFIG['classes'] must be a comma-separated string or a list of class names.")
    if not classes:
        raise ValueError("CONFIG['classes'] cannot be empty.")

    # 2) Read scalar/path config values.
    weights_path = Path(CONFIG["weights_path"])
    val_images_dir = Path(CONFIG["val_images_dir"])
    val_labels_dir = Path(CONFIG["val_labels_dir"])
    test_images_dir = Path(CONFIG["test_images_dir"])
    test_labels_dir = Path(CONFIG["test_labels_dir"])
    output_csv = Path(CONFIG["output_csv"])
    prediction_output_root = Path(CONFIG["prediction_output_root"])
    iou_threshold = float(CONFIG["iou_threshold"])
    channels = int(CONFIG["channels"])

    # 3) Build threshold values safely using integer hundredths to avoid float drift.
    threshold_start = float(CONFIG["threshold_start"])
    threshold_end = float(CONFIG["threshold_end"])
    threshold_step = float(CONFIG["threshold_step"])
    if threshold_step <= 0:
        raise ValueError("threshold step must be > 0")
    if threshold_end < threshold_start:
        raise ValueError("threshold end must be >= threshold start")

    start_i = int(round(threshold_start * 100))
    end_i = int(round(threshold_end * 100))
    step_i = int(round(threshold_step * 100))
    if step_i <= 0:
        raise ValueError("threshold step resolution must be at least 0.01")

    thresholds = [value / 100 for value in range(start_i, end_i + 1, step_i)]

    # 4) Fail fast if any required path is missing.
    for path in [
        weights_path,
        val_images_dir,
        val_labels_dir,
        test_images_dir,
        test_labels_dir,
    ]:
        if not path.exists():
            raise FileNotFoundError(f"Required path does not exist: {path}")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    prediction_output_root.mkdir(parents=True, exist_ok=True)

    # 5) Load model and run both split evaluations per threshold.
    logger.info("Loading RF-DETR checkpoint from: %s", weights_path)
    model = RFDETRNano(pretrain_weights=str(weights_path))

    for threshold in thresholds:
        logger.info("Running threshold %.2f for VAL and TEST", threshold)

        row = {
            "threshold": threshold,
            "iou_threshold": iou_threshold,
            "status": "ok",
            "error": "",
        }

        val_row = _evaluate_split_at_threshold(
            model=model,
            classes=classes,
            split_name="val",
            image_dir=val_images_dir,
            label_dir=val_labels_dir,
            output_root=prediction_output_root,
            threshold=threshold,
            iou_threshold=iou_threshold,
            channels=channels,
        )
        test_row = _evaluate_split_at_threshold(
            model=model,
            classes=classes,
            split_name="test",
            image_dir=test_images_dir,
            label_dir=test_labels_dir,
            output_root=prediction_output_root,
            threshold=threshold,
            iou_threshold=iou_threshold,
            channels=channels,
        )

        row.update(val_row)
        row.update(test_row)

        split_statuses = [row.get("VAL_status", "ok"), row.get("TEST_status", "ok")]
        if "error" in split_statuses:
            row["status"] = "error"
            error_messages = [
                msg
                for msg in [row.get("VAL_error", ""), row.get("TEST_error", "")]
                if msg
            ]
            row["error"] = " | ".join(error_messages)

        DataLogger.save_to_csv(output_csv, row)

    logger.info(
        "Threshold sweep complete. Results appended to: %s",
        output_csv,
    )


if __name__ == "__main__":
    main()
