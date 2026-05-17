"""YOLO evaluation helpers for polygon and box labels.

This module supports confidence-aware prediction labels, confidence-thresholded
matching, and COCO-style AP summaries over an IoU sweep.
"""

from abc import ABC, abstractmethod
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_IOU_SWEEP = [round(value / 100, 2) for value in range(50, 100, 5)]
DEFAULT_CONFIDENCE_SWEEP = [round(value / 100, 2) for value in range(5, 100, 2)]
EPS = 1e-9



class BaseYOLOEvaluator(ABC):
    """Base evaluator containing shared evaluation loop and utilities."""

    @classmethod
    @abstractmethod
    def load_labels(cls, path: Path, is_pred: bool = False) -> List:
        """Load label records from a text file.

        Returns a list of tuples in either:
        - (class_id, geometry)
        - (class_id, geometry, confidence)
        """
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def iou(cls, a, b) -> float:
        """Compute IoU between two geometry objects used by the subclass."""
        raise NotImplementedError()

    @staticmethod
    def _unpack_record(record: Tuple) -> Tuple[int, Any, float]:
        """Normalize a label record to (class_id, geometry, confidence)."""
        if len(record) >= 3:
            cls_id, geometry, confidence = record[:3]
        elif len(record) == 2:
            cls_id, geometry = record
            confidence = 1.0
        else:
            raise ValueError("label records must contain at least class and geometry")

        return int(cls_id), geometry, float(confidence)

    @staticmethod
    def _load_dataset(pred_dir: Path, gt_dir: Path, loader) -> List[Tuple[str, List, List]]:
        """Load and pair prediction/ground-truth labels by text filename."""
        gt_dir = Path(gt_dir)
        pred_dir = Path(pred_dir)
        gt_files = sorted(file_name for file_name in os.listdir(gt_dir) if file_name.endswith(".txt"))

        dataset = []
        for file_name in gt_files:
            gt_path = gt_dir / file_name
            pred_path = pred_dir / file_name
            dataset.append((file_name, loader(gt_path, is_pred=False), loader(pred_path, is_pred=True)))

        return dataset

    @classmethod
    def _match_predictions_to_gts(
        cls,
        gt_records: List[Tuple],
        pred_records: List[Tuple],
        iou_threshold: float,
        pred_order: List[int] | None = None,
    ) -> Tuple[List[Tuple[int, int, float]], List[int], List[int]]:
        """Match predictions to ground-truths using IoU."""
        num_preds = len(pred_records)
        num_gts = len(gt_records)

        if num_preds == 0 or num_gts == 0:
            return [], list(range(num_gts)), list(range(num_preds))

        gt_items: List[Any] = []
        for rec in gt_records:
            _, item, _ = cls._unpack_record(rec)
            gt_items.append(item)

        pred_items: List[Any] = []
        for rec in pred_records:
            _, item, _ = cls._unpack_record(rec)
            pred_items.append(item)

        ious: List[List[float]] = [[-1.0 for _ in range(num_gts)] for _ in range(num_preds)]
        for p in range(num_preds):
            for g in range(num_gts):
                ious[p][g] = float(cls.iou(pred_items[p], gt_items[g]))


        matched_pairs: List[Tuple[int, int, float]] = []

        if pred_order is None:
            while True:
                best_val = -1.0
                best_p = -1
                best_g = -1
                for p in range(num_preds):
                    for g in range(num_gts):
                        if ious[p][g] > best_val:
                            best_val = ious[p][g]
                            best_p = p
                            best_g = g

                if best_val < iou_threshold:
                    break

                matched_pairs.append((best_p, best_g, best_val))
                for g in range(num_gts):
                    ious[best_p][g] = -1.0
                for p in range(num_preds):
                    ious[p][best_g] = -1.0
        else:
            for p in pred_order:
                best_val = -1.0
                best_g = -1
                for g in range(num_gts):
                    if ious[p][g] > best_val:
                        best_val = ious[p][g]
                        best_g = g

                if best_val >= iou_threshold and best_g != -1:
                    matched_pairs.append((p, best_g, best_val))
                    for p2 in range(num_preds):
                        ious[p2][best_g] = -1.0

        matched_pred_idxs = {p for p, _, _ in matched_pairs}
        matched_gt_idxs = {g for _, g, _ in matched_pairs}

        unmatched_gt_idxs = [g for g in range(num_gts) if g not in matched_gt_idxs]
        unmatched_pred_idxs = [p for p in range(num_preds) if p not in matched_pred_idxs]

        return matched_pairs, unmatched_gt_idxs, unmatched_pred_idxs

    @classmethod
    def _evaluate_thresholded_matches(
        cls,
        dataset: List[Tuple[str, List, List]],
        classes: List,
        iou_threshold: float,
        confidence_threshold: float,
    ) -> Tuple[Dict, List, List]:
        """Compute TP/FP/FN for a single IoU threshold with optional score filter."""
        metrics = {"TP": 0, "FP": 0, "FN": 0}
        for cls_name in classes:
            metrics[f"{cls_name}_TP"] = 0
            metrics[f"{cls_name}_FP"] = 0
            metrics[f"{cls_name}_FN"] = 0
        all_true: List[int] = []
        all_pred: List[int] = []

        for _, gt_objs, pred_objs in dataset:
            if confidence_threshold is not None:
                pred_objs = [record for record in pred_objs if cls._unpack_record(record)[2] >= confidence_threshold]

            matched_pairs, unmatched_gts, unmatched_preds = cls._match_predictions_to_gts(
                gt_objs, pred_objs, iou_threshold
            )

            for pred_idx, gt_idx, _ in matched_pairs:
                pred_cls, _, _ = cls._unpack_record(pred_objs[pred_idx])
                gt_cls, _, _ = cls._unpack_record(gt_objs[gt_idx])


                if pred_cls < 0 or pred_cls >= len(classes) or gt_cls < 0 or gt_cls >= len(classes):
                    continue

                if pred_cls == gt_cls:
                    metrics["TP"] += 1
                    metrics[f"{classes[gt_cls]}_TP"] += 1
                else:
                    metrics["FP"] += 1
                    metrics[f"{classes[pred_cls]}_FP"] += 1
                    metrics["FN"] += 1
                    metrics[f"{classes[gt_cls]}_FN"] += 1

                all_true.append(gt_cls)
                all_pred.append(pred_cls)

            for pred_idx in unmatched_preds:
                try:
                    pred_cls, _, _ = cls._unpack_record(pred_objs[pred_idx])
                except Exception:
                    continue
                if pred_cls < 0 or pred_cls >= len(classes):
                    continue

                metrics["FP"] += 1
                metrics[f"{classes[pred_cls]}_FP"] += 1
                all_true.append(-1)
                all_pred.append(pred_cls)

            for gt_idx in unmatched_gts:

                gt_cls, _, _ = cls._unpack_record(gt_objs[gt_idx])

                if gt_cls < 0 or gt_cls >= len(classes):
                    continue

                metrics["FN"] += 1
                metrics[f"{classes[gt_cls]}_FN"] += 1
                all_true.append(gt_cls)
                all_pred.append(-1)

        return metrics, all_true, all_pred

    @classmethod
    def _evaluate_average_precision(
        cls,
        dataset: List[Tuple[str, List, List]],
        classes: List,
        iou_thresholds: List[float] = DEFAULT_IOU_SWEEP,
        confidence_thresholds: List[float] = DEFAULT_CONFIDENCE_SWEEP,
    ) -> Dict:
        """Compute per-class and overall AP metrics for an IoU sweep."""
        if dataset is None or not isinstance(dataset, list) or len(dataset) == 0:
            raise ValueError("dataset must be a non-empty list of (filename, gt_records, pred_records) tuples")

        def _integrate_ap(precisions: List[float], recalls: List[float]) -> float:
            if not recalls or not precisions:
                return 0.0

            if len(recalls) != len(precisions):
                raise ValueError("recalls and precisions must have the same length")

            mpre = [0.0] + [float(value) for value in precisions] + [1.0]
            mrec = [1.0] + [float(value) for value in recalls] + [0.0]


            ap = 0.0
            for index in range(len(mpre)-1):
                # calculate area based on trapeziums
                ap += (mpre[index+1] - mpre[index]) * (mrec[index] + mrec[index +1])/2

            return ap

        metrics: Dict[str, float] = {}
        per_class_ap_by_iou: Dict[str, Dict[float, float]] = {class_name: {} for class_name in classes}
        all_ap_by_iou: Dict[float, float] = {}

        for iou_threshold in iou_thresholds:
            all_precisions: List[float] = []
            all_recalls: List[float] = []
            class_precisions: Dict[str, List[float]] = {class_name: [] for class_name in classes}
            class_recalls: Dict[str, List[float]] = {class_name: [] for class_name in classes}

            for confidence_threshold in confidence_thresholds:
                summary, _, _ = cls._evaluate_thresholded_matches(
                    dataset=dataset,
                    classes=classes,
                    iou_threshold=iou_threshold,
                    confidence_threshold=confidence_threshold,
                )

                tp = float(summary["TP"])
                fp = float(summary["FP"])
                fn = float(summary["FN"])
                all_precisions.append(tp / (tp + fp + EPS))
                all_recalls.append(tp / (tp + fn + EPS))

                for class_name in classes:
                    cls_tp = float(summary[f"{class_name}_TP"])
                    cls_fp = float(summary[f"{class_name}_FP"])
                    cls_fn = float(summary[f"{class_name}_FN"])
                    class_precisions[class_name].append(cls_tp / (cls_tp + cls_fp + EPS))
                    class_recalls[class_name].append(cls_tp / (cls_tp + cls_fn + EPS))

            all_ap_by_iou[iou_threshold] = _integrate_ap(all_precisions, all_recalls)
            for class_name in classes:
                per_class_ap_by_iou[class_name][iou_threshold] = _integrate_ap(
                    class_precisions[class_name],
                    class_recalls[class_name],
                )


        class_map_50_values: List[float] = []
        class_map_50_95_values: List[float] = []
        for class_name in classes:
            ap_50 = per_class_ap_by_iou[class_name].get(0.5, 0.0)
            ap_50_95 = sum(per_class_ap_by_iou[class_name].values()) / len(per_class_ap_by_iou[class_name])
            metrics[f"{class_name}_map_50"] = round(ap_50, 4)
            metrics[f"{class_name}_map_50_95"] = round(ap_50_95, 4)
            class_map_50_values.append(ap_50)
            class_map_50_95_values.append(ap_50_95)

        all_map_50 = all_ap_by_iou.get(0.5, 0.0)
        all_map_50_95 = sum(all_ap_by_iou.values()) / len(all_ap_by_iou)
        metrics["All_map_50"] = round(all_map_50, 4)
        metrics["All_map_50_95"] = round(all_map_50_95, 4) 

        return metrics

    @classmethod
    def evaluate(
        cls,
        pred_dir: Path,
        gt_dir: Path,
        classes: List,
        iou_threshold: float = 0.5,
        confidence_threshold: float = 0.5,
        iou_thresholds: List[float] = DEFAULT_IOU_SWEEP,
        confidence_thresholds: List[float] = DEFAULT_CONFIDENCE_SWEEP,
    ) -> Tuple[Dict, List, List]:
        """Run full evaluation: thresholded matching + AP summaries."""
        dataset = cls._load_dataset(pred_dir=pred_dir, gt_dir=gt_dir, loader=cls.load_labels)

        metrics, all_true, all_pred = cls._evaluate_thresholded_matches(
            dataset=dataset,
            classes=classes,
            iou_threshold=iou_threshold,
            confidence_threshold=confidence_threshold,
        )

        epsilon = EPS
        metrics["precision"] = round(metrics["TP"] / (metrics["TP"] + metrics["FP"] + epsilon), 4)
        metrics["recall"] = round(metrics["TP"] / (metrics["TP"] + metrics["FN"] + epsilon), 4)
        metrics["f1_score"] = round(
            2 * metrics["precision"] * metrics["recall"] / (metrics["precision"] + metrics["recall"] + epsilon),
            4,
        )

        for cls_name in classes:
            tp = metrics[f"{cls_name}_TP"]
            fp = metrics[f"{cls_name}_FP"]
            fn = metrics[f"{cls_name}_FN"]
            precision = tp / (tp + fp + epsilon)
            recall = tp / (tp + fn + epsilon)
            metrics[f"{cls_name}_precision"] = round(precision, 4)
            metrics[f"{cls_name}_recall"] = round(recall, 4)
            metrics[f"{cls_name}_f1_score"] = round(
                2 * precision * recall / (precision + recall + epsilon),
                4,
            )

        metrics.update(cls._evaluate_average_precision(dataset, classes, iou_thresholds, confidence_thresholds))

        return metrics, all_true, all_pred

    @staticmethod
    def create_confusion_matrix(
        y_true: List, y_pred: List, save_path: Path, classes: List
    ) -> None:
        """Create and save a confusion matrix image.

        ``-1`` labels are mapped to an explicit background class for display.
        """
        from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

        bg_class = len(classes)
        classes_with_bg = classes + ["Background"]

        y_true_adj = [value if value != -1 else bg_class for value in y_true]
        y_pred_adj = [value if value != -1 else bg_class for value in y_pred]

        cm = confusion_matrix(y_true_adj, y_pred_adj, labels=range(len(classes_with_bg)))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes_with_bg)
        disp.plot(cmap="Blues", xticks_rotation=45)
        plt.title("YOLO Evaluation Confusion Matrix")
        plt.tight_layout()
        plt.savefig(save_path, dpi=200)
        plt.close()



class YOLOSegmentationEvaluator(BaseYOLOEvaluator):
    """Evaluator for polygon segmentation labels."""

    @staticmethod
    def _normalize_polygon_points(points: List[Tuple[float, float]]):
        """Normalize polygon points into a valid Shapely polygon when possible.

        Steps include deduplicating consecutive points, closing the ring,
        repairing invalid polygons via ``buffer(0)``, and resolving
        MultiPolygon outputs to their largest area component.
        """
        try:
            from shapely.geometry import Polygon

            cleaned_points: List[Tuple[float, float]] = []
            for point in points:
                current_point = (float(point[0]), float(point[1]))
                if not cleaned_points or current_point != cleaned_points[-1]:
                    cleaned_points.append(current_point)

            if len(cleaned_points) < 3:
                return None

            if cleaned_points[0] != cleaned_points[-1]:
                cleaned_points.append(cleaned_points[0])

            polygon = Polygon(cleaned_points)
            if polygon.is_empty:
                return None

            if not polygon.is_valid:
                polygon = polygon.buffer(0)

            if polygon.is_empty:
                return None

            if polygon.geom_type == "MultiPolygon":
                polygon = max(polygon.geoms, key=lambda geom: geom.area, default=None)
                if polygon is None or polygon.is_empty:
                    return None

            if polygon.geom_type not in {"Polygon", "MultiPolygon"}:
                return None

            return polygon if polygon.area > 0 else None
        except Exception:
            return None

    @classmethod
    def load_labels(cls, path: Path, is_pred: bool = False) -> List:
        """Load polygon labels with optional trailing prediction confidence.

        Input format per line:
        - GT: ``class x1 y1 x2 y2 ...``
        - Pred: ``class x1 y1 x2 y2 ... confidence`` (optional confidence)
        """
        path = Path(path)
        if not path.exists() or path.stat().st_size == 0:
            return []

        objects = []
        with open(path, "r", encoding="utf-8") as file_handle:
            for line in file_handle:
                parts = line.strip().split()
                if len(parts) < 7:
                    continue

                class_id = int(float(parts[0]))
                values = [float(value) for value in parts[1:]]
                confidence = None

                if is_pred and len(values) % 2 == 1:
                    confidence = values[-1]
                    values = values[:-1]

                if len(values) < 6 or len(values) % 2 != 0:
                    continue

                points = [(values[index], values[index + 1]) for index in range(0, len(values), 2)]
                polygon = cls._normalize_polygon_points(points)
                if polygon is None:
                    continue

                if confidence is None:
                    objects.append((class_id, polygon))
                else:
                    objects.append((class_id, polygon, confidence))

        return objects

    @classmethod
    def iou(cls, poly1_pts: List, poly2_pts: List) -> float:
        """Compute IoU between two polygons or polygon-like point sequences."""
        try:
            poly1 = poly1_pts if hasattr(poly1_pts, "intersection") else cls._normalize_polygon_points(poly1_pts)
            poly2 = poly2_pts if hasattr(poly2_pts, "intersection") else cls._normalize_polygon_points(poly2_pts)
            if poly1 is None or poly2 is None:
                return 0.0

            inter_area = poly1.intersection(poly2).area
            union_area = poly1.union(poly2).area
            if union_area <= 0:
                return 0.0
            return inter_area / union_area
        except Exception:
            return 0.0


class YOLOBBSEvaluator(BaseYOLOEvaluator):
    """Evaluator for bounding-box (xywh) labels."""

    @classmethod
    def load_labels(cls, path: Path, is_pred: bool = False) -> List:
        """Load bbox labels with optional trailing prediction confidence.

        Input format per line:
        - GT: ``class x_center y_center width height``
        - Pred: ``class x_center y_center width height confidence``
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
                values = [float(value) for value in parts[1:]]
                confidence = None

                if is_pred and len(values) >= 5:
                    confidence = values[4]
                    values = values[:4]

                if len(values) < 4:
                    continue

                x_center, y_center, width, height = values[:4]
                if confidence is None:
                    objects.append((class_id, (x_center, y_center, width, height)))
                else:
                    objects.append((class_id, (x_center, y_center, width, height), confidence))

        return objects

    @classmethod
    def iou(
        cls,
        box1: Tuple[float, float, float, float],
        box2: Tuple[float, float, float, float],
    ) -> float:
        """Compute IoU between two normalized ``xywh`` bounding boxes."""
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








