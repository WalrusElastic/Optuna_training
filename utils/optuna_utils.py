"""
Optuna trial management and study persistence utilities.
"""

import json
import optuna
from pathlib import Path
from typing import Dict


class OptunaTrialManager:
    """Utility class for managing Optuna studies and trials with JSON persistence."""

    @staticmethod
    def save_trial_to_json(trial: optuna.trial.Trial, json_file: Path, score: float) -> None:
        """
        Save a single optuna trial to JSON file.

        Description:
            Persists trial data to JSON file by appending to existing trials list.
            Enables resuming optimization from checkpoint without losing prior trial history.

        Args:
            trial (optuna.trial.Trial): Trial object to save
            json_file (Path): Output JSON file path
            score (float): Trial's objective function value

        Returns:
            None (writes to file)

        Behavior:
            - Loads existing trials, appends new trial, and overwrites file
            - Creates new file if it doesn't exist
        """
        trial_data = {
            "number": trial.number,
            "state": "COMPLETE",
            "params": trial.params,
            "distributions": {
                k: {
                    "type": type(v).__name__,
                    "low": getattr(v, "low", None),
                    "high": getattr(v, "high", None),
                    "choices": getattr(v, "choices", None),
                }
                for k, v in trial.distributions.items()
            },
            "user_attrs": trial.user_attrs,
            "system_attrs": trial.system_attrs,
            "value": score,
        }

        if not json_file.exists():
            trials = []
        else:
            with open(json_file, "r") as f:
                trials = json.load(f)
        trials.append(trial_data)

        with open(json_file, "w") as f:
            json.dump(trials, f, indent=2)

    @staticmethod
    def create_study_from_json(json_path: Path, study_name: str) -> optuna.Study:
        """
        Create or restore Optuna study from JSON file.

        Description:
            Initializes new Optuna study or loads existing trials from JSON to continue
            optimization. Reconstructs trial objects with original parameters and distributions.

        Args:
            json_path (Path): Path to JSON file containing previous trials (can be non-existent)
            study_name (str): Name of the study (used for Optuna identification)

        Returns:
            optuna.Study: Optuna study object ready for optimization

        Optimization Direction:
            - 'maximize': Higher scores are better (suitable for precision/recall metrics)

        Process:
            1. Creates new Study with specified name
            2. Loads previous trials from JSON if file exists
            3. Reconstructs distributions (Float, Int, Categorical) from saved metadata
            4. Recreates trial objects with same params, distributions, and values
            5. Adds all trials to study

        Note:
            - Uses in-memory storage (storage=None)
            - All trials must have same distribution types and ranges
            - Useful for long-running optimization that needs to resume across sessions
        """
        study = optuna.create_study(
            study_name=study_name,
            direction="maximize",
            storage=None
        )

        # Load existing trials
        if not json_path.exists():
            trials_data = []
        else:
            with open(json_path, "r") as f:
                trials_data = json.load(f)

        if trials_data:
            print(f"Loading {len(trials_data)} existing trials...")

            for t_data in trials_data:
                distributions = {}
                for k, v in t_data["distributions"].items():
                    if v["type"] == "FloatDistribution":
                        distributions[k] = optuna.distributions.FloatDistribution(
                            low=v["low"], high=v["high"]
                        )
                    elif v["type"] == "IntDistribution":
                        distributions[k] = optuna.distributions.IntDistribution(
                            low=v["low"], high=v["high"]
                        )
                    elif v["type"] == "CategoricalDistribution":
                        distributions[k] = optuna.distributions.CategoricalDistribution(
                            choices=v["choices"]
                        )

                trial = optuna.trial.create_trial(
                    params=t_data["params"],
                    distributions=distributions,
                    value=t_data["value"],
                    state=optuna.trial.TrialState[t_data["state"]],
                    user_attrs=t_data.get("user_attrs", {}),
                    system_attrs=t_data.get("system_attrs", {}),
                )
                study.add_trial(trial)

        return study
