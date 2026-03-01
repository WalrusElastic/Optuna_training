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
    def predict_images_and_save_output(classes, model, input_dir, output_dir, threshold=0.5, channels=1):
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


