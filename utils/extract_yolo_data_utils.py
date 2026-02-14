"""
Extract YOLO training/eval results to CSV and JSON, and loss graph PNGs.
"""

import csv
import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd



class YOLODataExtractor:
    """Utility class for extracting and saving YOLO training/evaluation results."""

    @staticmethod
    def append_to_csv(file_path: Path, data_dict: Dict) -> None:
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
    def save_results_to_json(
        output_json_path: Path,
        trial_number: int,
        training_params: Dict,
        additional_parameters: Dict,
        additional_metrics: Dict,
        metrics: Optional[Dict] = None,
    ) -> None:
        """
        Save trial results (parameters and metrics) to JSON file.

        Loads existing trials if file exists, appends new result, overwrites file.
        Output structure: {"trials": [{trial_number, training_parameters, ...}, ...]}.
        """
        output_json_path = Path(output_json_path)
        if metrics is None:
            metrics = {}

        if output_json_path.exists():
            with open(output_json_path, "r") as f:
                all_results = json.load(f)
        else:
            all_results = {"trials": []}

        result_entry = {
            "trial_number": trial_number,
            "training_parameters": training_params,
            "augmentation_parameters": additional_parameters,
            "additional_metrics": additional_metrics,
            "test_metrics": metrics,
        }
        all_results["trials"].append(result_entry)

        with open(output_json_path, "w") as f:
            json.dump(all_results, f, indent=2)

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
