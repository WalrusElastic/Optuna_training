"""
Image and label preprocessing utilities for YOLO training pipeline.
"""

import os
import shutil
from typing import List, Tuple

import cv2
import numpy as np
import albumentations as A


class PreprocessingUtils:
    """
    Image processing, YOLO label loading/saving, and augmentation generation.
    """

    @staticmethod
    def convert_16bit_to_8bit_minmax(img_16bit: np.ndarray) -> np.ndarray:
        """Convert 16-bit image to 8-bit using min-max normalization."""
        img_float = img_16bit.astype(np.float32)
        min_val = np.min(img_float)
        max_val = np.max(img_float)

        if max_val == min_val:
            return np.zeros_like(img_float, dtype=np.uint8)

        img_8bit = 255 * (img_float - min_val) / (max_val - min_val)
        return img_8bit.astype(np.uint8)

    @staticmethod
    def minmax_norm(img: np.ndarray, strength: float) -> np.ndarray:
        """Apply min-max normalization with strength parameter."""
        img = img.astype(np.float32)
        mean = img.mean()
        out = mean + (img - mean) * (1 - strength)
        return np.clip(out, 0, 65535).astype(np.uint16)

    @staticmethod
    def apply_cubic_convolution(img: np.ndarray, scale_factor: float) -> np.ndarray:
        """Resize image using cubic convolution interpolation."""
        new_h = int(img.shape[0] * scale_factor)
        new_w = int(img.shape[1] * scale_factor)
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    
    @staticmethod
    def preprocess_image(img_path: str, input_size: int = 512, output_size: int = 1024
    ) -> np.ndarray:
        """
        Load and preprocess image with complete pipeline.

        Loads image from file, applies min-max normalization, resizes via cubic convolution,
        and converts to 8-bit format. This is the standard preprocessing for all training images.

        Args:
            img_path (str): Path to image file to load
            input_size (int): Expected input image size in pixels (default: 512)
            output_size (int): Desired output image size in pixels (default: 1024)

        Returns:
            np.ndarray: Preprocessed 8-bit image array ready for YOLO training
        """
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        img = PreprocessingUtils.minmax_norm(img, 0.5)
        scale_factor = output_size / input_size
        img = PreprocessingUtils.apply_cubic_convolution(img, scale_factor)
        img = PreprocessingUtils.convert_16bit_to_8bit_minmax(img)
        return img

    @staticmethod
    def load_yolo_labels(
        txt_path: str,
    ) -> List[Tuple[int, List[Tuple[float, float]]]]:
        """
        Load YOLO segmentation labels from file.

        Reads YOLO segmentation format labels where each line contains:
        class_id x1 y1 x2 y2 ... xn yn (normalized polygon coordinates)

        Returns list of (class_id, polygon_points) tuples. Returns empty list if file
        doesn't exist or is empty.
        """
        labels = []
        if not os.path.exists(txt_path):
            return labels

        with open(txt_path, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                parts = line.strip().split()
                cls_id = int(parts[0])
                coords = [
                    (float(parts[i]), float(parts[i + 1]))
                    for i in range(1, len(parts), 2)
                ]
                labels.append((cls_id, coords))

        return labels

    @staticmethod
    def save_augmented_labels(
        txt_path: str,
        polygon_pts: List,
        poly_sizes: List[Tuple[int, int]],
        img_size: int,
    ) -> None:
        """
        Save normalized YOLO labels to file after augmentation.

        Converts pixel-space polygon coordinates back to normalized format and saves to
        YOLO label file. Automatically removes polygons with points outside image bounds.
        """
        lines = []
        cum_sum = 0

        for cls_id, num_pts in poly_sizes:
            polygon = polygon_pts[cum_sum : cum_sum + num_pts]
            cum_sum += num_pts

            valid = all(
                0 <= pt[0] <= img_size and 0 <= pt[1] <= img_size for pt in polygon
            )
            if not valid:
                continue

            normalized_pts = [(pt[0] / img_size, pt[1] / img_size) for pt in polygon]

            line_parts = [str(cls_id)]
            for pt in normalized_pts:
                line_parts.extend([str(pt[0]), str(pt[1])])

            lines.append(" ".join(line_parts))

        with open(txt_path, "w") as f:
            for line in lines:
                f.write(line + "\n")

    @staticmethod
    def generate_transform(
        tif_path: str,
        txt_path: str,
        output_tif_folder: str,
        output_txt_folder: str,
        transforms: A.Compose,
        edit_labels: bool = False,
        iterations: int = 1,
        input_img_size: int = 512,
        output_img_size: int = 1024,
    ) -> None:
        """
        Generate augmented image and label samples.

        Applies preprocessing to image, optionally transforms polygon coordinates,
        and saves augmented versions.

        Args:
            tif_path (str): Path to source image file
            txt_path (str): Path to source YOLO label file
            output_tif_folder (str): Folder to save augmented images
            output_txt_folder (str): Folder to save augmented labels
            transforms (A.Compose): Albumentations transformation pipeline to apply
            edit_labels (bool): If True, transforms polygon coordinates; if False, copies labels unchanged
            iterations (int): Number of augmented samples to generate from single image
            input_img_size (int): Input image size (default: 512)
            output_img_size (int): Output image size after preprocessing (default: 1024)
        """
        img = PreprocessingUtils.preprocess_image(tif_path, input_img_size, output_img_size)

        polygon_pts, poly_sizes = None, None
        if edit_labels:
            initial_labels = PreprocessingUtils.load_yolo_labels(txt_path)
            if initial_labels:
                polygon_pts_norm = [
                    pt for _, coords in initial_labels for pt in coords
                ]
                polygon_pts = [
                    (pt[0] * output_img_size, pt[1] * output_img_size)
                    for pt in polygon_pts_norm
                ]
                poly_sizes = [
                    (cls_id, len(coords)) for cls_id, coords in initial_labels
                ]

        base_name = os.path.splitext(os.path.basename(tif_path))[0]

        for i in range(iterations):
            aug_txt_path = os.path.join(
                output_txt_folder, f"{base_name}_{i+1}.txt"
            )
            aug_img_path = os.path.join(
                output_tif_folder, f"{base_name}_{i+1}.tif"
            )

            for path in [aug_txt_path, aug_img_path]:
                if os.path.exists(path):
                    os.remove(path)

            if edit_labels and polygon_pts:
                aug = transforms(image=img, keypoints=polygon_pts)
                aug_img = aug["image"]
                aug_pts = aug["keypoints"]
            else:
                aug = transforms(image=img)
                aug_img = aug["image"]
                aug_pts = polygon_pts if polygon_pts else []

            cv2.imwrite(aug_img_path, aug_img)

            if edit_labels and aug_pts and poly_sizes:
                PreprocessingUtils.save_augmented_labels(
                    aug_txt_path, aug_pts, poly_sizes, output_img_size
                )
            elif not edit_labels and os.path.exists(txt_path):
                shutil.copy(txt_path, aug_txt_path)
