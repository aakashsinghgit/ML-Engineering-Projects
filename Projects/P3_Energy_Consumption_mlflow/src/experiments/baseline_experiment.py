"""
Baseline LinearRegression experiment for household energy consumption forecasting.

Predicts Global_active_power from calendar features only (hour, dayofweek, month,
is_weekend). Deliberately *does not* use the other electrical signals (Voltage,
Global_intensity, Sub_metering_*) because they are co-measured at the same
timestamp as the target — using them would be leakage in a forecasting setup.

This is intentionally a weak baseline. The point is to establish a comparable
first run in MLflow that future experiments (lag features, tree models, etc.)
can be benchmarked against.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from datetime import datetime
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from mlflow_tracking import MLflowTracker, MLflowRun
from logger import get_logger
from exceptions import DataIngestionException
from utils import get_project_root


TARGET = "Global_active_power"


class BaselineLinearRegressionExperiment:
    """LinearRegression baseline using calendar features only."""

    def __init__(self, experiment_name: str = "baseline_models"):
        self.logger = get_logger(f"experiment_{experiment_name}")
        self.tracker = MLflowTracker(experiment_name)
        self.project_root = get_project_root()
        self.training_data: Optional[pd.DataFrame] = None
        self.validation_data: Optional[pd.DataFrame] = None

    def load_data(self) -> None:
        training_path = (
            self.project_root / "artifacts" / "data_ingested"
            / "training" / "training_data_2007_2008.csv"
        )
        validation_path = (
            self.project_root / "artifacts" / "data_ingested"
            / "validation" / "validation_data_2009_2010.csv"
        )

        for p in (training_path, validation_path):
            if not p.exists():
                raise DataIngestionException(
                    f"Data not found: {p}", file_path=str(p)
                )

        self.training_data = pd.read_csv(training_path, index_col=0, parse_dates=True)
        self.validation_data = pd.read_csv(validation_path, index_col=0, parse_dates=True)
        self.logger.info(
            f"Loaded training {self.training_data.shape}, "
            f"validation {self.validation_data.shape}"
        )

    @staticmethod
    def build_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
        idx = df.index
        return pd.DataFrame(
            {
                "hour": idx.hour,
                "dayofweek": idx.dayofweek,
                "month": idx.month,
                "is_weekend": (idx.dayofweek >= 5).astype(int),
            },
            index=idx,
        )

    def prepare_xy(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        df = df.dropna(subset=[TARGET])
        X = self.build_calendar_features(df)
        y = df[TARGET].astype(float)
        return X, y

    @staticmethod
    def evaluate(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
        return {
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "r2": float(r2_score(y_true, y_pred)),
        }

    def plot_pred_vs_actual(
        self, y_true: pd.Series, y_pred: pd.Series, name: str
    ) -> None:
        # Minute-level overlay is unreadable; resample to daily mean.
        df = pd.DataFrame({"actual": y_true, "predicted": y_pred}, index=y_true.index)
        daily = df.resample("D").mean()

        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(daily.index, daily["actual"], label="actual", linewidth=1)
        ax.plot(daily.index, daily["predicted"], label="predicted", linewidth=1)
        ax.set_title("Daily mean Global_active_power — predicted vs actual (validation)")
        ax.set_ylabel("kW")
        ax.legend()
        plt.tight_layout()
        self.tracker.log_figure(fig, f"{name}_pred_vs_actual_daily.png")
        plt.close(fig)

    def run(self, run_name: Optional[str] = None, tags: Optional[Dict[str, str]] = None) -> None:
        run_name = run_name or f"baseline_linreg_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        tags = {"phase": "baseline", "model": "linear_regression", **(tags or {})}

        with MLflowRun(self.tracker, run_name, tags):
            self.load_data()

            X_train, y_train = self.prepare_xy(self.training_data)
            X_val, y_val = self.prepare_xy(self.validation_data)

            self.tracker.log_parameters({
                "model": "LinearRegression",
                "features": ",".join(X_train.columns),
                "target": TARGET,
                "fit_intercept": True,
                "n_train": len(X_train),
                "n_val": len(X_val),
                "training_file": "training_data_2007_2008.csv",
                "validation_file": "validation_data_2009_2010.csv",
            })

            self.logger.info("Fitting LinearRegression...")
            model = LinearRegression(fit_intercept=True)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_val)
            metrics = self.evaluate(y_val, y_pred)
            self.tracker.log_metrics(metrics)
            self.logger.info(f"Validation metrics: {metrics}")

            self.plot_pred_vs_actual(
                y_val, pd.Series(y_pred, index=y_val.index), "validation"
            )

            self.tracker.log_model(model, "linear_regression", model_type="sklearn")


def main() -> None:
    logger = get_logger("baseline_main")
    logger.info("Starting baseline LinearRegression experiment")
    try:
        BaselineLinearRegressionExperiment().run()
        logger.info("Baseline experiment completed successfully")
    except Exception as e:
        logger.error(f"Baseline experiment failed: {e}")
        raise


if __name__ == "__main__":
    main()
