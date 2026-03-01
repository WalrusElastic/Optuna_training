"""
Extract YOLO training/eval results to CSV and JSON, and loss graph PNGs.
"""

import csv
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd

logger = logging.getLogger(__name__)



class DataLogger:
    """Utility class for extracting and saving YOLO training/evaluation results."""

    @staticmethod
    def save_to_csv(file_path: Path, data_dict: Dict) -> None:
        """
        Append a dictionary as a new row to CSV file.

        Creates file with headers if it doesn't exist. Handles new fields by adding
        blank columns to existing rows. Uses UTF-8 encoding.
        """
        file_path = Path(file_path)

        if not file_path.exists():
            with open(file_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(data_dict.keys()))
                writer.writeheader()
                writer.writerow(data_dict)
            return

        with open(file_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            existing_fields = reader.fieldnames or []

        new_fields = [k for k in data_dict.keys() if k not in existing_fields]
        all_fields = existing_fields + new_fields

        with open(file_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=all_fields)
            writer.writeheader()
            for row in rows:
                for field in new_fields:
                    row[field] = ""
                writer.writerow(row)
            writer.writerow(data_dict)

    @staticmethod
    def save_to_json(
        output_json_path: Path,
        trial_number: int,
        params: Dict,
        metrics: Dict,
    ) -> None:
        """
        Save a trial's results to a JSON file using a parameter dict.

        Parameters
        ----------
        output_json_path : Path
            Location of the JSON file to update.
        params : Dict
            Dictionary containing any keys you wish to record (e.g.
            ``trial_number``, ``training_parameters``,
            ``augmentation_parameters``, ``additional_metrics``).
        metrics : Optional[Dict]
            A separate dictionary of evaluation/test metrics. These will be
            stored under the ``"test_metrics"`` key in the entry.

        The file structure is::

            {
                "trials": [
                    [trial_number, params, metrics],
                    ...
                ]
            }

        Existing files are loaded and the new entry appended; otherwise a
        new container is created.
        """
        output_json_path = Path(output_json_path)
        if metrics is None:
            metrics = {}

        if output_json_path.exists():
            with open(output_json_path, "r") as f:
                all_results = json.load(f)
        else:
            all_results = {"trials": []}


        all_results["trials"].append([trial_number, params, metrics])

        with open(output_json_path, "w") as f:
            json.dump(all_results, f, indent=2)
