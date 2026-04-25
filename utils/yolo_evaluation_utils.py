"""
YOLO segmentation evaluation: IoU, label loading, metrics, and test runner.
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd

import matplotlib.pyplot as plt

# Note: shapely and sklearn are optional dependencies for yolosegmenataion evaluation and confusion matrix plotting. In such cases, they are lazy loaded


class YOLOSegmentationEvaluator:
    """Utility class for evaluating YOLO segmentation models."""

    @staticmethod
    def polygon_iou(poly1_pts: List, poly2_pts: List) -> float:
        """Calculate IoU between two polygons."""
        try:
            from shapely.geometry import Polygon

            poly1 = Polygon(poly1_pts)
            poly2 = Polygon(poly2_pts)
            if not poly1.is_valid or not poly2.is_valid:
                return 0.0
            inter = poly1.intersection(poly2).area
            union = poly1.union(poly2).area
            return inter / union if union > 0 else 0.0
        except Exception:
            return 0.0

    @staticmethod
    def load_yolo_poly_labels(path: Path, is_pred: bool = False) -> List:
        """
        Load YOLO polygon segmentation labels from file.

        Reads YOLO format polygon labels where each line contains class ID followed by
        normalized polygon coordinates. Handles both ground truth and prediction formats.

        Args:
            path (Path): Path to YOLO label text file
            is_pred (bool): If True, treats as prediction format and removes trailing values
                           if coordinate count is odd. If False, treats as ground truth format.

        Returns:
            List: List of (class_id, polygon_points) tuples where polygon_points is a list
                  of (x, y) coordinate tuples in normalized format (0.0-1.0)
        """
        path = Path(path)
        if not path.exists() or path.stat().st_size == 0:
            return []

        objs = []
        with open(path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 4:
                    continue

                cls = int(float(parts[0]))
                coords = list(map(float, parts[1:]))

                if is_pred and len(coords) % 2 == 1:
                    coords = coords[:-1]

                points = [(coords[i], coords[i + 1]) for i in range(0, len(coords), 2)]
                objs.append((cls, points))

        return objs

    @classmethod
    def evaluate_yolo_seg(
        cls,
        pred_dir: Path,
        gt_dir: Path,
        classes: List,
        iou_threshold: float = 0.5,
    ) -> Tuple[Dict, List, List]:
        """
        Evaluate YOLO segmentation performance on predicted vs ground truth labels.

        Compares predicted polygon labels against ground truth by matching with IoU threshold.
        Returns overall and per-class metrics (precision, recall). 

        Args:
            pred_dir (Path): Directory containing predicted label files
            gt_dir (Path): Directory containing ground truth label files
            classes (List): List of class names
            iou_thresh (float): IoU threshold for matching predictions to ground truth

        Returns:
            metrics (Dict): Dictionary of evaluation metrics
            all_true (List): List of all ground truth class IDs
            all_pred (List): List of all predicted class IDs
        """
        all_true = []
        all_pred = []
        metrics = {
            "TEST_TP": 0,
            "TEST_FP": 0,
            "TEST_FN": 0,
        }

        for cls_id in classes:
            metrics[f"TEST_{cls_id}_TP"] = 0
            metrics[f"TEST_{cls_id}_FP"] = 0
            metrics[f"TEST_{cls_id}_FN"] = 0

        gt_files = [f for f in os.listdir(gt_dir) if f.endswith(".txt")]

        for file in gt_files:
            gt_path = gt_dir / file
            pred_path = pred_dir / file

            gt_objs = cls.load_yolo_poly_labels(gt_path, is_pred=False)
            pred_objs = cls.load_yolo_poly_labels(pred_path, is_pred=True)

            matched_gt = set()

            for pred_cls, pred_poly in pred_objs:
                best_iou, best_j = 0, -1

                for j, (gt_cls, gt_poly) in enumerate(gt_objs):
                    iou = cls.polygon_iou(pred_poly, gt_poly)
                    if iou > best_iou:
                        best_iou, best_j = iou, j

                if (
                    best_iou >= iou_threshold
                    and best_j != -1
                    and best_j not in matched_gt
                    and gt_objs[best_j][0] == pred_cls
                ):
                    metrics["TEST_TP"] += 1
                    metrics[f"TEST_{classes[pred_cls]}_TP"] += 1
                    all_true.append(gt_objs[best_j][0])
                    all_pred.append(pred_cls)
                    matched_gt.add(best_j)
                else:
                    metrics["TEST_FP"] += 1
                    metrics[f"TEST_{classes[pred_cls]}_FP"] += 1
                    true_label = (
                        gt_objs[best_j][0]
                        if best_iou >= iou_threshold and best_j not in matched_gt
                        else -1
                    )
                    all_true.append(true_label)

                    if best_iou >= iou_threshold and best_j not in matched_gt:
                        gt_cls = gt_objs[best_j][0]
                        metrics[f"TEST_{classes[gt_cls]}_FN"] += 1
                        matched_gt.add(best_j)

                    all_pred.append(pred_cls)

            for j, (gt_cls, _) in enumerate(gt_objs):
                if j not in matched_gt:
                    metrics["TEST_FN"] += 1
                    metrics[f"TEST_{classes[gt_cls]}_FN"] += 1
                    all_true.append(gt_cls)
                    all_pred.append(-1)

        eps = 1e-6
        metrics["TEST_precision"] = round(
            metrics["TEST_TP"] / (metrics["TEST_TP"] + metrics["TEST_FP"] + eps), 4
        )
        metrics["TEST_recall"] = round(
            metrics["TEST_TP"] / (metrics["TEST_TP"] + metrics["TEST_FN"] + eps), 4
        )

        for cls_id in classes:
            tp = metrics[f"TEST_{cls_id}_TP"]
            fp = metrics[f"TEST_{cls_id}_FP"]
            fn = metrics[f"TEST_{cls_id}_FN"]
            metrics[f"TEST_{cls_id}_precision"] = round(tp / (tp + fp + eps), 4)
            metrics[f"TEST_{cls_id}_recall"] = round(tp / (tp + fn + eps), 4)

        return metrics, all_true, all_pred

    @staticmethod
    def create_confusion_matrix(
        y_true: List, y_pred: List, save_path: Path, classes: List
    ) -> None:
        """
        Create and save confusion matrix visualization as PNG.

        Saves to run_path/test_results/confusion_matrix.png. Treats -1 as Background.

        Args:
            y_true (List): Ground truth labels
            y_pred (List): Predicted labels
            save_path (Path): Path to save the confusion matrix PNG
            classes (List): List of class names
        """
        from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
        bg_class = len(classes)
        classes_with_bg = classes + ["Background"]

        y_true_adj = [t if t != -1 else bg_class for t in y_true]
        y_pred_adj = [p if p != -1 else bg_class for p in y_pred]

        cm = confusion_matrix(
            y_true_adj, y_pred_adj, labels=range(len(classes_with_bg))
        )
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm, display_labels=classes_with_bg
        )
        disp.plot(cmap="Blues", xticks_rotation=45)
        plt.title("YOLOv11 Segmentation Confusion Matrix")
        plt.tight_layout()
        plt.savefig(save_path, dpi=200)
        plt.close()

    @staticmethod
    def extract_loss_graphs(folder_path: Path) -> None:
        """
        Extract loss curves from training results.csv and save as PNG graphs.

        Reads folder_path/results.csv, plots train/val for box, seg, cls, dfl loss,
        saves PNGs to folder_path/test_results/. Clips loss values to 10.0.
        """
        folder_path = Path(folder_path)
        file_path = folder_path / "results.csv"
        df = pd.read_csv(file_path)

        pairs = [
            ("train/box_loss", "val/box_loss"),
            ("train/seg_loss", "val/seg_loss"),
            ("train/cls_loss", "val/cls_loss"),
            ("train/dfl_loss", "val/dfl_loss"),
        ]

        output_dir = folder_path / "test_results"
        output_dir.mkdir(exist_ok=True)

        for pair in pairs:
            x_data = df["epoch"]
            y1 = df[pair[0]].clip(upper=10)
            y2 = df[pair[1]].clip(upper=10)
            plt.figure(figsize=(8, 6))
            plt.plot(x_data, y1, label=pair[0])
            plt.plot(x_data, y2, label=pair[1])
            plt.title(pair[0][6:])
            plt.legend()
            plt.grid(True)
            plt.savefig(output_dir / f"{pair[0][6:]}.png")
            plt.close()


class YOLOBBSEvaluator(YOLOSegmentationEvaluator):
    """Utility class for evaluating YOLO bbox predictions without shapely."""

    @staticmethod
    def bbox_iou(
        box1: Tuple[float, float, float, float],
        box2: Tuple[float, float, float, float],
    ) -> float:
        """Calculate IoU for two YOLO xywh boxes in normalized coordinates."""
        x1_center, y1_center, w1, h1 = box1
        x2_center, y2_center, w2, h2 = box2

        x1_min = x1_center - (w1 / 2.0)
        y1_min = y1_center - (h1 / 2.0)
        x1_max = x1_center + (w1 / 2.0)
        y1_max = y1_center + (h1 / 2.0)

        x2_min = x2_center - (w2 / 2.0)
        y2_min = y2_center - (h2 / 2.0)
        x2_max = x2_center + (w2 / 2.0)
        y2_max = y2_center + (h2 / 2.0)

        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)

        inter_w = max(0.0, inter_x_max - inter_x_min)
        inter_h = max(0.0, inter_y_max - inter_y_min)
        inter_area = inter_w * inter_h

        area1 = max(0.0, x1_max - x1_min) * max(0.0, y1_max - y1_min)
        area2 = max(0.0, x2_max - x2_min) * max(0.0, y2_max - y2_min)
        union = area1 + area2 - inter_area

        if union <= 0:
            return 0.0
        return inter_area / union

    @staticmethod
    def load_yolo_poly_labels(path: Path, is_pred: bool = False) -> List:
        """Load YOLO bbox labels as (class_id, xywh) tuples.

        Expected formats per line:
        - class x_center y_center width height
        - class x_center y_center width height confidence
        """
        path = Path(path)
        if not path.exists() or path.stat().st_size == 0:
            return []

        objects = []
        with open(path, "r", encoding="utf-8") as file_handle:
            for line in file_handle:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue

                class_id = int(float(parts[0]))
                values = list(map(float, parts[1:]))
                x_center, y_center, width, height = values[:4]
                objects.append((class_id, (x_center, y_center, width, height)))

        return objects

    @staticmethod
    def polygon_iou(poly1_pts: Tuple[float, float, float, float], poly2_pts: Tuple[float, float, float, float]) -> float:
        """Override polygon IoU with bbox IoU for bbox-based evaluation."""
        return YOLOBBSEvaluator.bbox_iou(poly1_pts, poly2_pts)




