"""
RF-DETR extraction utilities: best epoch and results handling (OOP format).
"""

import json
from pathlib import Path
from typing import Optional
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

class RFDETRExtractor:
    """
    Class for extracting RF-DETR training results and metrics.
    """

    @staticmethod
    def get_best_epoch(metrics_csv_path: str | Path, patience: int) ->int:
        """
        Determine the best epoch based on the last epoch minus a patience value.
        """
        csv_path = Path(metrics_csv_path)

        df = pd.read_csv(csv_path)

        df = df.groupby("epoch").mean(numeric_only=True).reset_index().sort_values("epoch")



        # find last epoch
        last_epoch = df["epoch"].max() 
        best_epoch = last_epoch - patience
        return best_epoch
    


    @staticmethod
    def get_validation_results(results_file_path: str | Path, best_epoch: int) -> dict:
        """
        Extract validation results for the best epoch from the metrics CSV file.

        """
        results_path = Path(results_file_path)



        df = pd.read_csv(results_path).groupby("epoch").mean(numeric_only=True).reset_index()

        matches = df[df["epoch"] == best_epoch]
        row = matches.iloc[0] if len(matches) > 0 else (df.iloc[-1] if len(df) > 0 else None)


        validation = {col.replace("val/", "val/"): float(row[col]) for col in df.columns if col.startswith("val/")}
        return validation

    @staticmethod
    def plot_metrics(
        metrics_csv: str,
        output_path: Optional[str] = None,
        loss_log_scale: bool = False,
    ) -> str:
        """Read a PTL ``CSVLogger`` metrics file and save a seaborn training plot.

        The figure contains one subplot per metric group (Loss, AP@0.50,
        AP@0.50:0.95, AR), arranged in a 2-column grid.  Only groups with at
        least one non-NaN column are shown.

        Args:
            metrics_csv: Path to the ``metrics.csv`` file produced by
                ``CSVLogger``.
            output_path: Destination for the PNG file.  Defaults to
                ``metrics_plot.png`` next to ``metrics_csv``.

        Returns:
            The absolute path where the figure was saved.

        Raises:
            ImportError: If ``matplotlib``, ``pandas``, or ``seaborn`` are not
                installed.
            FileNotFoundError: If ``metrics_csv`` does not exist.
        """

        csv_path = Path(metrics_csv)
        if not csv_path.exists():
            raise FileNotFoundError(f"metrics.csv not found: {csv_path}")

        if output_path is None:
            output_path = str(csv_path.parent / "metrics_plot.png")

        df = pd.read_csv(csv_path)
        if "epoch" not in df.columns:
            raise ValueError("metrics.csv does not contain an 'epoch' column.")
        # CSVLogger writes one row per step; aggregate to one row per epoch.
        df = df.groupby("epoch").mean(numeric_only=True).reset_index()

        def _val_cols(*patterns: str) -> list[str]:
            """Return val/ columns whose name contains any of the given patterns."""
            return [c for c in df.columns if c.startswith("val/") and any(p in c for p in patterns) and df[c].notna().any()]

        # Loss: only the aggregate scalars, not per-component breakdowns.
        loss_cols = [c for c in ("train/loss", "val/loss", "test/loss") if c in df.columns and df[c].notna().any()]

        # AP/AR: all val/ columns matching each group (base + EMA when present).
        # test/ metrics are excluded — they only appear at the final epoch as a
        # single dot which seaborn renders as a legend entry with no visible line.
        metric_groups: dict[str, list[str]] = {
            "Loss": loss_cols,
            "AP@0.50": _val_cols("mAP_50"),  # matches mAP_50 and ema_mAP_50 but not mAP_50_95
            "AP@0.50:0.95": _val_cols("mAP_50_95"),
            "AR": _val_cols("mAR"),
        }
        # Exclude mAP_50_95 hits from the AP@0.50 bucket (substring overlap).
        metric_groups["AP@0.50"] = [c for c in metric_groups["AP@0.50"] if "mAP_50_95" not in c]
        metric_groups = {k: v for k, v in metric_groups.items() if v}

        n_groups = len(metric_groups)
        n_cols = 2
        n_rows = (n_groups + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 5 * n_rows), squeeze=False)
        axes_flat = axes.flatten()

        melted = df.melt(id_vars="epoch", var_name="metric", value_name="value")

        for idx, (title, metric_list) in enumerate(metric_groups.items()):
            ax = axes_flat[idx]
            group_data = melted[melted["metric"].isin(metric_list)]
            sns.lineplot(data=group_data, x="epoch", y="value", hue="metric", marker="o", ax=ax)
            ax.set_title(title, fontsize=13, fontweight="bold")
            ax.set_xlabel("Epoch", fontsize=11)
            ax.set_ylabel(title, fontsize=11)
            ax.grid(True, alpha=0.3)
            if title == "Loss" and loss_log_scale:
                ax.set_yscale("log")

        for idx in range(n_groups, len(axes_flat)):
            axes_flat[idx].set_visible(False)

        fig.suptitle("RF-DETR Training Metrics", fontsize=14)
        fig.tight_layout()
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        return str(Path(output_path).resolve())
