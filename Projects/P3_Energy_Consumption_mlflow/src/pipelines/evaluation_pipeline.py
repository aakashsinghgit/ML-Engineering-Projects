"""Evaluation pipeline for energy consumption forecasting models."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from datetime import datetime
from typing import Dict, Optional, Tuple
from copy import deepcopy

import numpy as np
import pandas as pd

from mlflow_tracking import MLflowTracker, MLflowRun
from logger import get_logger
from exceptions import DataIngestionException, ModelEvaluationException
from mlflow_config import FEATURE_ENGINEERING_CONFIG
from utils import get_project_root, load_object
from components.feature_engineering import FeatureEngineer
from components.model_trainer import ModelTrainer

TARGET = "Global_active_power"


class EvaluationPipeline:
    """Pipeline to evaluate a trained model on validation data."""

    def __init__(self, experiment_name: str = "evaluation_pipeline"):
        self.logger = get_logger(f"pipeline_{experiment_name}")
        self.tracker = MLflowTracker(experiment_name)
        self.project_root = get_project_root()
        self.validation_data: Optional[pd.DataFrame] = None
        self.feature_engineer = FeatureEngineer()
        self.model_trainer = ModelTrainer()

    def load_data(self, validation_data_file: Optional[str] = None) -> None:
        validation_path = (
            self.project_root / "artifacts" / "data_ingested"
            / "validation" / "validation_data_2009_2010.csv"
        )

        if validation_data_file:
            validation_path = self.project_root / validation_data_file

        if not validation_path.exists():
            raise DataIngestionException(
                f"Validation data not found: {validation_path}", file_path=str(validation_path)
            )

        self.validation_data = pd.read_csv(validation_path, index_col=0, parse_dates=True)
        self.logger.info(f"Loaded validation data {self.validation_data.shape} from {validation_path}")

    @staticmethod
    def _feature_config(feature_set: str) -> dict:
        config = deepcopy(FEATURE_ENGINEERING_CONFIG)

        if feature_set == "all":
            return config

        config["lag_features"]["enabled"] = feature_set in ("lag_only", "all", "seasonal_plus_lag")
        config["rolling_features"]["enabled"] = feature_set in ("rolling_only", "all")
        config["seasonal_features"]["enabled"] = feature_set in ("seasonal", "all", "seasonal_plus_lag")
        config["interaction_features"]["enabled"] = feature_set == "all"

        if feature_set == "seasonal":
            config["lag_features"]["enabled"] = False
            config["rolling_features"]["enabled"] = False
            config["interaction_features"]["enabled"] = False

        if feature_set == "lag_only":
            config["rolling_features"]["enabled"] = False
            config["seasonal_features"]["enabled"] = False
            config["interaction_features"]["enabled"] = False

        if feature_set == "rolling_only":
            config["lag_features"]["enabled"] = False
            config["seasonal_features"]["enabled"] = False
            config["interaction_features"]["enabled"] = False

        if feature_set == "seasonal_plus_lag":
            config["rolling_features"]["enabled"] = False
            config["interaction_features"]["enabled"] = False

        return config

    def prepare_features(self, df: pd.DataFrame, feature_set: str = "all") -> Tuple[pd.DataFrame, pd.Series]:
        df = df.dropna(subset=[TARGET])
        config = self._feature_config(feature_set)
        engineer = FeatureEngineer(config=config)
        X = engineer.build_all_features(df, TARGET)
        y = df.loc[X.index, TARGET].astype(float)
        self.logger.info(f"Prepared validation features ({feature_set}): {X.shape}")
        return X, y

    def load_model(self, local_model_path: Optional[str] = None):
        if local_model_path:
            model_path = Path(local_model_path)
        else:
            model_dir = self.project_root / "artifacts" / "models"
            model_files = list(model_dir.glob("*.pkl"))
            if not model_files:
                raise ModelEvaluationException(
                    "No models found in artifacts/models/",
                    evaluation_metric="model_load"
                )
            model_path = max(model_files, key=lambda p: p.stat().st_mtime)

        if not model_path.exists():
            raise ModelEvaluationException(
                f"Model file not found: {model_path}", evaluation_metric="model_load"
            )

        self.logger.info(f"Loading model from {model_path}")
        return load_object(model_path)

    def run(
        self,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        local_model_path: Optional[str] = None,
        validation_data_file: Optional[str] = None,
        feature_set: str = "all",
    ) -> Dict[str, object]:
        run_name = run_name or f"evaluation_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        tags = {"phase": "evaluation", "pipeline": "evaluation", **(tags or {})}

        model_path = Path(local_model_path) if local_model_path else None

        with MLflowRun(self.tracker, run_name, tags):
            self.load_data(validation_data_file=validation_data_file)
            X_val, y_val = self.prepare_features(self.validation_data, feature_set=feature_set)
            model = self.load_model(local_model_path=str(model_path) if model_path else None)

            try:
                y_pred = model.predict(X_val)
            except Exception as exc:
                raise ModelEvaluationException(
                    f"Prediction failed during evaluation: {exc}", evaluation_metric="prediction"
                )

            metrics = self.model_trainer.evaluate(y_val, y_pred)
            self.tracker.log_metrics(metrics)
            self.tracker.log_parameters(
                {
                    "pipeline": "evaluation_pipeline",
                    "target": TARGET,
                    "validation_data_file": str(validation_data_file) if validation_data_file else "default",
                    "feature_set": feature_set,
                    "n_validation": len(X_val),
                    "n_features": X_val.shape[1],
                    "model_path": str(model_path) if model_path else "latest",
                }
            )
            self.logger.info(f"Evaluation metrics: {metrics}")
            return {
                "run_name": run_name,
                "metrics": metrics,
                "n_samples": len(X_val),
                "n_features": X_val.shape[1],
            }


def main() -> None:
    logger = get_logger("evaluation_pipeline_main")
    logger.info("Starting evaluation pipeline")

    try:
        result = EvaluationPipeline().run()
        logger.info(f"Evaluation pipeline completed successfully: {result}")
    except Exception as exc:
        logger.error(f"Evaluation pipeline failed: {exc}")
        raise


if __name__ == "__main__":
    main()
