import os
from pathlib import Path

import shutil
import io
import requests
import supervision as sv
from PIL import Image
import numpy as np
from rfdetr import RFDETRNano

class RFDETRPredictor:
    """Class for predicting objects in images using RF-DETR model and saving annotated results.
    """

    @staticmethod
    def _normalize_and_validate_bbox(bbox, img_width: int, img_height: int):
        """Clamp xyxy box to image bounds and return normalized corners or None."""
        x1, y1, x2, y2 = bbox.tolist()
        x1 = max(0.0, min(float(x1), float(img_width)))
        y1 = max(0.0, min(float(y1), float(img_height)))
        x2 = max(0.0, min(float(x2), float(img_width)))
        y2 = max(0.0, min(float(y2), float(img_height)))

        if x2 <= x1 or y2 <= y1:
            return None

        return (
            x1 / img_width,
            y1 / img_height,
            x2 / img_width,
            y2 / img_height,
        )

    @staticmethod
    def _bbox_to_polygon_line(class_idx: int, bbox, img_width: int, img_height: int):
        """Convert one xyxy box to a YOLO polygon-label line."""
        normalized_bbox = RFDETRPredictor._normalize_and_validate_bbox(
            bbox=bbox,
            img_width=img_width,
            img_height=img_height,
        )
        if normalized_bbox is None:
            return None

        nx1, ny1, nx2, ny2 = normalized_bbox
        return (
            f"{class_idx} "
            f"{nx1:.6f} {ny1:.6f} "
            f"{nx2:.6f} {ny1:.6f} "
            f"{nx2:.6f} {ny2:.6f} "
            f"{nx1:.6f} {ny2:.6f}\n"
        )

    @staticmethod
    def predict_image_dir_and_save_seg_labels(
        model,
        classes,
        input_dir,
        pred_labels_dir,
        threshold=0.5,
        channels=1,
    ):
        """Run RF-DETR prediction and write polygon labels for one threshold.

        This is intended for evaluation workflows where downstream utilities expect
        YOLO polygon text labels.
        """
        pred_labels_dir = Path(pred_labels_dir)
        pred_labels_dir.mkdir(parents=True, exist_ok=True)

        image_paths = sorted(Path(input_dir).glob("*.tif"))
        images = [Image.open(path).convert("RGB") for path in image_paths]

        detections_list = model.predict(images, threshold=threshold, channels=channels)

        for image, image_path, detections in zip(images, image_paths, detections_list):
            img_width, img_height = image.size
            label_path = pred_labels_dir / f"{image_path.stem}.txt"

            with open(label_path, "w", encoding="utf-8") as file_handle:
                for bbox, class_id in zip(detections.xyxy, detections.class_id):
                    class_idx = int(class_id)
                    if class_idx < 0 or class_idx >= len(classes):
                        continue

                    line = RFDETRPredictor._bbox_to_polygon_line(
                        class_idx=class_idx,
                        bbox=bbox,
                        img_width=img_width,
                        img_height=img_height,
                    )
                    if line is not None:
                        file_handle.write(line)

    @staticmethod
    def predict_image_dir_and_save_bbs_labels(
        model,
        classes,
        input_dir,
        pred_labels_dir,
        threshold=0.5,
        channels=1,
        include_confidence=False,
    ):
        """Run RF-DETR prediction and write YOLO bbox labels for one threshold.

        Output format per line:
        - Without confidence: class x_center y_center width height
        - With confidence: class x_center y_center width height confidence
        """
        pred_labels_dir = Path(pred_labels_dir)
        pred_labels_dir.mkdir(parents=True, exist_ok=True)

        image_paths = sorted(Path(input_dir).glob("*.tif"))
        images = [Image.open(path).convert("RGB") for path in image_paths]

        detections_list = model.predict(images, threshold=threshold, channels=channels)

        for image, image_path, detections in zip(images, image_paths, detections_list):
            img_width, img_height = image.size
            label_path = pred_labels_dir / f"{image_path.stem}.txt"

            with open(label_path, "w", encoding="utf-8") as file_handle:
                for det_index, (bbox, class_id) in enumerate(zip(detections.xyxy, detections.class_id)):
                    class_idx = int(class_id)
                    if class_idx < 0 or class_idx >= len(classes):
                        continue

                    normalized_bbox = RFDETRPredictor._normalize_and_validate_bbox(
                        bbox=bbox,
                        img_width=img_width,
                        img_height=img_height,
                    )
                    if normalized_bbox is None:
                        continue

                    nx1, ny1, nx2, ny2 = normalized_bbox
                    x_center = (nx1 + nx2) / 2.0
                    y_center = (ny1 + ny2) / 2.0
                    width = nx2 - nx1
                    height = ny2 - ny1

                    if include_confidence:
                        confidence = float(detections.confidence[det_index])
                        file_handle.write(
                            f"{class_idx} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f} {confidence:.6f}\n"
                        )
                    else:
                        file_handle.write(
                            f"{class_idx} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"
                        )

    @staticmethod
    def predict_images_and_save_annotated_images(classes, model, input_dir, output_dir, threshold=0.5, channels=1):
        """
        Predict objects in images using RF-DETR model and save annotated images. Takes in the model, input directory of images, and output directory for annotated results. Saves both annotated images and YOLO-format labels in seperate subdirectories.

        Args:
            classes (list): List of class names.
            model: Loaded RF-DETR model.
            input_dir (str or Path): Directory containing input images (.tif files).
            output_dir (str or Path): Directory to saved annotated images and labels.
            threshold (float): Confidence threshold for detections.
            channels (int): Number of channels in images.
        """
        output_dir = Path(output_dir)
        labels_dir = output_dir/ "labels"
        images_dir = output_dir / "images"

        for path in [output_dir, labels_dir, images_dir]:
            if path.exists():
                shutil.rmtree(path)
            path.mkdir(parents=True, exist_ok=True)
        
        image_paths = list(Path(input_dir).glob("*.tif"))

        images = [Image.open(path).convert("RGB") for path in image_paths]

        detections_list = model.predict(images, threshold=threshold, channels=channels)

        for i, (image, image_path, detections) in enumerate(zip(images, image_paths, detections_list)):
            labels = [
                f"{classes[class_id]} {confidence:.2f}"
                for class_id, confidence
                in zip(detections.class_id, detections.confidence)
            ]

            annotated_image = image.copy()
            annotated_image = sv.BoxAnnotator().annotate(annotated_image, detections)
            annotated_image = sv.LabelAnnotator().annotate(annotated_image, detections, labels)
            # Convert PIL Image to numpy array for cv2.imwrite compatibility
            annotated_image = np.array(annotated_image)
            output_name = f"pred_{image_path.name}"
            with sv.ImageSink(target_dir_path=images_dir) as sink:
                sink.save_image(image=annotated_image, image_name=output_name)

            # save detection labels in YOLO format (normalized coordinates)
            label_path = labels_dir / f"{image_path.stem}.txt"
            img_width, img_height = image.size  # PIL Image dimensions
            with open(label_path, "w") as lf:
                for bbox, cid in zip(detections.xyxy, detections.class_id):
                    x1, y1, x2, y2 = bbox.tolist()
                    # Convert xyxy to normalized yolo format (x_center, y_center, width, height)
                    x_center = (x1 + x2) / (2 * img_width)
                    y_center = (y1 + y2) / (2 * img_height)
                    width = (x2 - x1) / img_width
                    height = (y2 - y1) / img_height
                    lf.write(f"{int(cid)} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")


