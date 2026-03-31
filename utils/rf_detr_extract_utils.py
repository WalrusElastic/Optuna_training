"""
RF-DETR extraction utilities: best epoch and results handling (OOP format).
"""

import json
from pathlib import Path

class RFDETRExtractor:
    """
    Class for extracting RF-DETR training results and metrics.
    """

    @staticmethod
    def get_best_epoch(log_file_path):
        """
        Read the log file and return the last 'all_best_ep' value.
        
        Args:
            log_file_path (str or Path): Path to the log.txt file.
        
        Returns:
            int: The best epoch from the last entry.
        """
        log_file = Path(log_file_path)
        best_epoch = None
        
        with open(log_file, 'r') as f:
            for line in f:
                if line.strip():
                    try:
                        data = json.loads(line.strip())
                        if 'all_best_ep' in data:
                            best_epoch = data['all_best_ep']
                    except json.JSONDecodeError:
                        continue  # Skip invalid JSON lines
        
        return best_epoch
    
    @staticmethod
    def get_final_epoch(log_file_path):
        """
        Read the log file and return the last 'epoch' value.
        
        Args:
            log_file_path (str or Path): Path to the log.txt file.
        Returns:
            int: The final epoch from the last entry.
        """
        log_file = Path(log_file_path)
        final_epoch = None
        
        with open(log_file, 'r') as f:
            for line in f:
                if line.strip():
                    try:
                        data = json.loads(line.strip())
                        if 'epoch' in data:
                            final_epoch = data['epoch']
                    except json.JSONDecodeError:
                        continue  # Skip invalid JSON lines
        
        return final_epoch
    
    @staticmethod
    def combine_dictionaries(results_list):
        """
        Flatten a list of result dictionaries into a single flat dictionary.
        
        Args:
            results_list (list): List of dicts, each with 'class' and metrics.
        
        Returns:
            dict: Flattened dict with keys like 'class_1_map_50_95'.
        """
        flat_dict = {}
        for item in results_list:
            class_name = item["class"].replace(" ", "_")
            for key, value in item.items():
                if key != "class":
                    # Replace special chars in key
                    clean_key = key.replace("@", "_").replace(":", "_").replace(".", "_")
                    flat_key = f"{class_name}_{clean_key}"
                    flat_dict[flat_key] = value
        return flat_dict

    @staticmethod
    def get_validation_and_test_results(results_file_path):
        """
        Read the results.json file and return validation and test results as lists of dicts.
        
        Args:
            results_file_path (str or Path): Path to the results.json file.
        
        Returns:
            tuple: (validation_results, test_results) where each is a dictionary.
        """
        results_file = Path(results_file_path)
        
        with open(results_file, 'r') as f:
            data = json.load(f)
        
        class_map = data.get("class_map", {})
        validation_results = class_map.get("valid", [])
        test_results = class_map.get("test", [])
        
        return RFDETRExtractor.combine_dictionaries(validation_results), RFDETRExtractor.combine_dictionaries(test_results)   


    @staticmethod
    def flatten_dict(d, prefix='', sep='_'):
        """
        Recursively flatten a nested dictionary.
        
        Args:
            d (dict): The dictionary to flatten.
            prefix (str): Prefix for keys.
            sep (str): Separator for nested keys.
        
        Returns:
            dict: Flattened dictionary.
        """
        items = []
        for k, v in d.items():
            new_key = f"{prefix}{sep}{k}" if prefix else k
            if isinstance(v, dict):
                items.extend(RFDETRExtractor.flatten_dict(v, new_key, sep=sep).items())
            elif isinstance(v, list):
                for i, item in enumerate(v):
                    if isinstance(item, dict):
                        items.extend(RFDETRExtractor.flatten_dict(item, f"{new_key}{sep}{i}", sep=sep).items())
                    else:
                        items.append((f"{new_key}{sep}{i}", item))
            else:
                items.append((new_key, v))
        return dict(items)

    @staticmethod
    def prefix_keys(d: dict, prefix: str) -> dict:
        """
        Return a copy of a dictionary with `prefix` prepended to each key.

        Only top-level keys are modified. If ``prefix`` is empty the
        original dictionary is returned as a shallow copy.
        """
        if not prefix:
            return d.copy()
        return {f"{prefix}{k}": v for k, v in d.items()}