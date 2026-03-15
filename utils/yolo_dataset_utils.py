"""
YOLO dataset utilities for training.
"""

from typing import List

import numpy as np

from ultralytics.data.dataset import YOLODataset

import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class YOLOWeightedDataset(YOLODataset):
    """YOLO dataset with class weighting for balanced sampling."""

    default_weight_scale: float = 1.0
    # Note, this is the power applied to the class weights. 1.0 means the weight for each class is directly proportional to the inverse of its frequency. >1 exaggerates the weighting effect, <1 smooths it out.

    def __init__(self, *args, mode="train", **kwargs):

        weight_scale = self.default_weight_scale
        super(YOLOWeightedDataset, self).__init__(*args, **kwargs)

        if not hasattr(self, "labels"):
            raise AttributeError("self.labels is not defined")

        self.train_mode = "train" in self.prefix
        self.weight_scale = weight_scale
        self.count_instances()

        # Calculate class weights: higher count = lower weight
        class_weights = np.sum(self.counts) / self.counts

        # apply scale: 1.0 = original, >1 exaggerates minority-pref weighting, <1 smooths
        class_weights = np.power(class_weights, self.weight_scale)
        self.class_weights = np.array(class_weights)
        self.weights = self.calculate_weights()
        self.probabilities = self.calculate_probabilities()
        logger.info(f"YOLOWeightedDataset initialized with weight scale {self.weight_scale}.")



    def count_instances(self) -> None:
        """Count instances per class."""
        self.counts = np.zeros(len(self.data["names"]))
        for label in self.labels:
            cls = label["cls"].reshape(-1).astype(int)
            for cls_id in cls:
                self.counts[cls_id] += 1

        # Avoid division by zero
        self.counts = np.where(self.counts == 0, 1, self.counts)

    def calculate_weights(self) -> List[float]:
        """Calculate sampling weight for each image."""
        weights = []
        for label in self.labels:
            cls = label["cls"].reshape(-1).astype(int)

            if cls.size == 0:
                weights.append(1)
            else:
                weight = np.mean(self.class_weights[cls])
                weights.append(weight)

        return weights

    def calculate_probabilities(self) -> List[float]:
        """Convert weights to sampling probabilities."""
        total_weight = sum(self.weights)
        return [w / total_weight for w in self.weights]

    def __getitem__(self, index: int):
        """Get batch item with weighted sampling."""
        if self.train_mode:
            index = np.random.choice(len(self.labels), p=self.probabilities)

        return self.transforms(self.get_image_and_label(index))
