"""Training pipeline for energy consumption forecasting."""

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
from exceptions import DataIngestionException, ModelTrainingException
from mlflow_config import FEATURE_ENGINEERING_CONFIG
from utils import get_project_root, save_object, create_directory
from components.feature_engineering import FeatureEngineer
from components.model_trainer import ModelTrainer
from components.hyperparameter_tuner import HyperparameterTuner

TARGET = "Global_active_power"


class TrainPipeline:
    """Reusable training pipeline with feature engineering and model flexibility."""

    def __init__(self, experiment_name: str = "training_pipeline"):
        self.logger = get_logger(f"pipeline_{experiment_name}")
        self.tracker = MLflowTracker(experiment_name)
        self.project_root = get_project_root()
        self.training_data: Optional[pd.DataFrame] = None
        self.model_dir = create_directory(self.project_root / "artifacts" / "models")
        self.feature_engineer = FeatureEngineer()
        self.model_trainer = ModelTrainer()
        self.tuner = HyperparameterTuner()

    def load_data(self, training_data_file: Optional[str] = None) -> None:
        training_path = (
            self.project_root / "artifacts" / "data_ingested"
            / "training" / "training_data_2007_2008.csv"
        )

        if training_data_file:
            training_path = self.project_root / training_data_file

        if not training_path.exists():
            raise DataIngestionException(
                f"Training data not found: {training_path}", file_path=str(training_path)
            )

        self.training_data = pd.read_csv(training_path, index_col=0, parse_dates=True)
        self.logger.info(f"Loaded training data {self.training_data.shape} from {training_path}")

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
        """Build features using feature engineering."""
        df = df.dropna(subset=[TARGET])

        config = self._feature_config(feature_set)
        engineer = FeatureEngineer(config=config)
        X = engineer.build_all_features(df, TARGET)
        y = df.loc[X.index, TARGET].astype(float)

        self.logger.info(f"Features prepared ({feature_set}): {X.shape}")
        return X, y

    def run(
        self,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        model_name: str = "random_forest",
        model_params: Optional[Dict] = None,
        training_data_file: Optional[str] = None,
        feature_set: str = "all",
        tune: bool = False,
        tuning_method: str = "grid",
        tuning_grid_type: str = "coarse",
        tuning_n_iter: int = 20,
    ) -> Dict[str, object]:
        run_name = run_name or f"train_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        tags = {"phase": "training", "pipeline": "train", "model": model_name, **(tags or {})}

        with MLflowRun(self.tracker, run_name, tags):
            self.load_data(training_data_file=training_data_file)
            X_train, y_train = self.prepare_features(self.training_data, feature_set=feature_set)

            self.tracker.log_parameters(
                {
                    "pipeline": "train_pipeline",
                    "model_name": model_name,
                    "target": TARGET,
                    "training_data_file": str(training_data_file) if training_data_file else "default",
                    "n_train": len(X_train),
                    "n_features": X_train.shape[1],
                    "feature_set": feature_set,
                    "tune": tune,
                    "tuning_method": tuning_method,
                    "tuning_grid_type": tuning_grid_type,
                    "tuning_n_iter": tuning_n_iter,
                    **(model_params or {}),
                }
            )

            self.logger.info(f"Training {model_name}...")
            try:
                if tune:
                    best_params, best_score = self.tuner.tune_model(
                        model_name,
                        X_train,
                        y_train,
                        method=tuning_method,
                        grid_type=tuning_grid_type,
                        n_iter=tuning_n_iter,
                    )
                    self.logger.info(f"Best hyperparameters found: {best_params} (score={best_score:.4f})")
                    model = self.tuner.best_model
                    self.tracker.log_metrics({"tuning_best_score": best_score})
                    self.tracker.log_parameters({f"best_{k}": v for k, v in best_params.items()})
                else:
                    model = self.model_trainer.train(
                        model_name, X_train, y_train, params=model_params
                    )
            except Exception as exc:
                raise ModelTrainingException(
                    f"Failed to train model: {exc}", model_name=model_name
                )

            # Evaluate on training data
            y_pred = self.model_trainer.predict(X_train)
            metrics = self.model_trainer.evaluate(y_train, y_pred)
            self.tracker.log_metrics(metrics)
            self.logger.info(f"Training metrics: {metrics}")

            # Log feature importance if available
            feature_importance = self.model_trainer.get_feature_importance()
            if feature_importance:
                self.tracker.log_parameters({f"feature_importance_{k}": v for k, v in list(feature_importance.items())[:5]})

            model_path = self.model_dir / f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
            save_object(model, model_path)
            self.tracker.log_model(model, model_name, model_type="sklearn")

            self.logger.info(f"Saved model to {model_path}")
            return {
                "run_name": run_name,
                "model_name": model_name,
                "local_model_path": str(model_path),
                "metrics": metrics,
                "n_features": X_train.shape[1],
                "tuned": tune,
            }


def main() -> None:
    logger = get_logger("train_pipeline_main")
    logger.info("Starting training pipeline")

    try:
        result = TrainPipeline().run(
            model_name="linear_regression",
            tune = False,
            feature_set="lag_only",
            )
        logger.info(f"Training pipeline completed successfully: {result}")
    except Exception as exc:
        logger.error(f"Training pipeline failed: {exc}")
        raise


if __name__ == "__main__":
    main()
    # Quick tuning test. Keep it commented when not testing.