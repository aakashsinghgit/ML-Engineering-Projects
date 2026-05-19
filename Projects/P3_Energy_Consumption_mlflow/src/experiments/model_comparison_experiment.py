"""
Model comparison experiment for calendar-feature forecasting.

Trains the same calendar features used by the baseline experiment across
multiple model classes. Runs LinearRegression, RandomForest, and LightGBM
as nested child runs beneath a parent MLflow run so the UI groups them together.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from datetime import datetime
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import mlflow
from mlflow_tracking import MLflowTracker, MLflowRun
from logger import get_logger
from exceptions import DataIngestionException, ModelTrainingException, ModelEvaluationException
from utils import get_project_root

try:
    from lightgbm import LGBMRegressor
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

TARGET = "Global_active_power"


class ModelComparisonExperiment:
    """Compare multiple models with the same calendar feature set."""

    def __init__(self, experiment_name: str = "model_comparison"):
        self.logger = get_logger(f"experiment_{experiment_name}")
        self.tracker = MLflowTracker(experiment_name)
        self.project_root = get_project_root()
        self.training_data: Optional[pd.DataFrame] = None
        self.validation_data: Optional[pd.DataFrame] = None

        self.model_specs = [
            {
                "key": "linear_regression",
                "name": "Linear Regression",
                "constructor": lambda: LinearRegression(fit_intercept=True),
                "model_type": "sklearn",
                "params": {"fit_intercept": True},
            },
            {
                "key": "random_forest",
                "name": "Random Forest",
                "constructor": lambda: RandomForestRegressor(
                    n_estimators=100, max_depth=10, random_state=42
                ),
                "model_type": "sklearn",
                "params": {"n_estimators": 100, "max_depth": 10, "random_state": 42},
            },
        ]

        if LIGHTGBM_AVAILABLE:
            self.model_specs.append(
                {
                    "key": "lightgbm",
                    "name": "LightGBM",
                    "constructor": lambda: LGBMRegressor(
                        n_estimators=100, max_depth=10, learning_rate=0.1, random_state=42
                    ),
                    "model_type": "lightgbm",
                    "params": {
                        "n_estimators": 100,
                        "max_depth": 10,
                        "learning_rate": 0.1,
                        "random_state": 42,
                    },
                }
            )
        else:
            self.logger.warning(
                "LightGBM is not installed; LightGBM will be skipped in model comparison."
            )

    def load_data(self) -> None:
        training_path = (
            self.project_root / "artifacts" / "data_ingested"
            / "training" / "training_data_2007_2008.csv"
        )
        validation_path = (
            self.project_root / "artifacts" / "data_ingested"
            / "validation" / "validation_data_2009_2010.csv"
        )

        for path in (training_path, validation_path):
            if not path.exists():
                raise DataIngestionException(
                    f"Data not found: {path}", file_path=str(path)
                )

        self.training_data = pd.read_csv(training_path, index_col=0, parse_dates=True)
        self.validation_data = pd.read_csv(validation_path, index_col=0, parse_dates=True)
        self.logger.info(
            f"Loaded training {self.training_data.shape}, validation {self.validation_data.shape}"
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
        self, y_true: pd.Series, y_pred: pd.Series, model_key: str
    ) -> None:
        df = pd.DataFrame({"actual": y_true, "predicted": y_pred}, index=y_true.index)
        daily = df.resample("D").mean()

        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(daily.index, daily["actual"], label="actual", linewidth=1)
        ax.plot(daily.index, daily["predicted"], label="predicted", linewidth=1)
        ax.set_title(
            f"Daily mean Global_active_power — predicted vs actual ({model_key})"
        )
        ax.set_ylabel("kW")
        ax.legend()
        plt.tight_layout()

        self.tracker.log_figure(fig, f"{model_key}_pred_vs_actual_daily.png")
        plt.close(fig)

    def _run_single_model(
        self,
        model_key: str,
        model_name: str,
        constructor,
        model_type: str,
        params: Dict[str, object],
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> Dict[str, object]:
        run_name = f"model_comparison_{model_key}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        tags = {
            "phase": "model_selection",
            "step": "model_comparison",
            "model": model_key,
            "feature_set": "calendar_features",
        }

        with mlflow.start_run(run_name=run_name, nested=True, tags=tags):
            self.tracker.log_parameters(
                {
                    "model_name": model_name,
                    "model_key": model_key,
                    "feature_set": "calendar_features",
                    "target": TARGET,
                    "n_train": len(X_train),
                    "n_validation": len(X_val),
                    **{f"param_{k}": v for k, v in params.items()},
                }
            )

            self.logger.info(f"Training {model_name} ({model_key})")
            try:
                model = constructor()
                model.fit(X_train, y_train)
            except Exception as exc:
                raise ModelTrainingException(
                    f"Failed to train {model_name}: {exc}", model_name=model_key
                )

            try:
                y_pred = model.predict(X_val)
            except Exception as exc:
                raise ModelEvaluationException(
                    f"Failed to evaluate {model_name}: {exc}", evaluation_metric="prediction"
                )

            metrics = self.evaluate(y_val, y_pred)
            self.tracker.log_metrics(metrics)
            self.logger.info(f"Validation metrics for {model_name}: {metrics}")

            self.plot_pred_vs_actual(y_val, pd.Series(y_pred, index=y_val.index), model_key)
            self.tracker.log_model(model, model_key, model_type=model_type)

            run_id = mlflow.active_run().info.run_id
            return {
                "model_key": model_key,
                "model_name": model_name,
                "metrics": metrics,
                "run_id": run_id,
            }

    def run(self, run_name: Optional[str] = None, tags: Optional[Dict[str, str]] = None) -> None:
        run_name = run_name or f"model_comparison_parent_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        tags = {
            "phase": "model_selection",
            "experiment": "model_comparison",
            "feature_set": "calendar_features",
            **(tags or {}),
        }

        with MLflowRun(self.tracker, run_name, tags):
            self.load_data()
            X_train, y_train = self.prepare_xy(self.training_data)
            X_val, y_val = self.prepare_xy(self.validation_data)

            self.tracker.log_parameters(
                {
                    "comparison_type": "calendar_features",
                    "target": TARGET,
                    "n_train": len(X_train),
                    "n_validation": len(X_val),
                    "training_file": "training_data_2007_2008.csv",
                    "validation_file": "validation_data_2009_2010.csv",
                }
            )

            child_runs: List[Dict[str, object]] = []

            for spec in self.model_specs:
                child_runs.append(
                    self._run_single_model(
                        model_key=spec["key"],
                        model_name=spec["name"],
                        constructor=spec["constructor"],
                        model_type=spec["model_type"],
                        params=spec["params"],
                        X_train=X_train,
                        y_train=y_train,
                        X_val=X_val,
                        y_val=y_val,
                    )
                )

            sorted_runs = sorted(child_runs, key=lambda item: item["metrics"]["rmse"])
            best = sorted_runs[0]
            self.tracker.log_parameters(
                {
                    "best_model_key": best["model_key"],
                    "best_model_name": best["model_name"],
                    "best_model_rmse": best["metrics"]["rmse"],
                    "compared_models": ",".join([r["model_key"] for r in child_runs]),
                    "child_run_ids": ",".join([r["run_id"] for r in child_runs]),
                }
            )
            self.logger.info(
                f"Model comparison completed. Best model: {best['model_name']} ({best['metrics']})"
            )


def main() -> None:
    logger = get_logger("model_comparison_main")
    logger.info("Starting model comparison experiment")
    try:
        ModelComparisonExperiment().run()
        logger.info("Model comparison experiment completed successfully")
    except Exception as exc:
        logger.error(f"Model comparison experiment failed: {exc}")
        raise


if __name__ == "__main__":
    main()
