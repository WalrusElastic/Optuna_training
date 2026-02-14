"""
YOLO dataset utilities for training.
"""

import logging
from typing import List

import numpy as np

from ultralytics.data.dataset import YOLODataset

logger = logging.getLogger(__name__)


class YOLOWeightedDataset(YOLODataset):
    """YOLO dataset with class weighting for balanced sampling."""

    def __init__(self, *args, mode="train", **kwargs):
        super(YOLOWeightedDataset, self).__init__(*args, **kwargs)

        if not hasattr(self, "labels"):
            raise AttributeError("self.labels is not defined")

        self.train_mode = "train" in self.prefix
        self.count_instances()

        # Calculate class weights: higher count = lower weight
        class_weights = np.sum(self.counts) / self.counts
        self.class_weights = np.array(class_weights)
        self.weights = self.calculate_weights()
        self.probabilities = self.calculate_probabilities()

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
