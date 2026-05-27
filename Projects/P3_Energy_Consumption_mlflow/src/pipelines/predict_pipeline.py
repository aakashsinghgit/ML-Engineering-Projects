"""Prediction pipeline for energy consumption forecasting models."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from typing import Dict, Optional
from copy import deepcopy

import pandas as pd

from mlflow_config import FEATURE_ENGINEERING_CONFIG
from components.feature_engineering import FeatureEngineer
from logger import get_logger
from exceptions import ModelPredictionException
from utils import get_project_root, load_object, create_directory

TARGET = "Global_active_power"


class PredictionPipeline:
    """Reusable prediction pipeline for inference with saved models."""

    def __init__(self):
        self.logger = get_logger("pipeline_predict")
        self.project_root = get_project_root()
        self.model_dir = self.project_root / "artifacts" / "models"
        self.feature_engineer = FeatureEngineer()

    def load_model(self, local_model_path: Optional[str] = None):
        if local_model_path:
            model_path = Path(local_model_path)
        else:
            model_files = list(self.model_dir.glob("*.pkl"))
            if not model_files:
                raise ModelPredictionException(
                    "No saved model found in artifacts/models/", input_data_info=str(self.model_dir)
                )
            model_path = max(model_files, key=lambda p: p.stat().st_mtime)

        if not model_path.exists():
            raise ModelPredictionException(
                f"Model not found: {model_path}", input_data_info=str(model_path)
            )

        self.logger.info(f"Loading model from {model_path}")
        return load_object(model_path)

    def load_input_data(self, input_path: str) -> pd.DataFrame:
        data_path = Path(input_path)
        if not data_path.exists():
            raise ModelPredictionException(
                f"Input data file not found: {data_path}", input_data_info=str(data_path)
            )
        df = pd.read_csv(data_path, index_col=0, parse_dates=True)
        self.logger.info(f"Loaded input data {df.shape} from {data_path}")
        return df

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

    def predict(
        self,
        input_df: pd.DataFrame,
        local_model_path: Optional[str] = None,
        output_path: Optional[str] = None,
        feature_set: str = "all",
    ) -> pd.DataFrame:
        if TARGET not in input_df.columns:
            raise ModelPredictionException(
                f"Target column '{TARGET}' missing from input data", input_data_info=str(input_df.columns.tolist())
            )

        config = self._feature_config(feature_set)
        engineer = FeatureEngineer(config=config)
        X = engineer.build_all_features(input_df, TARGET)
        model = self.load_model(local_model_path=local_model_path)

        try:
            predictions = model.predict(X)
        except Exception as exc:
            raise ModelPredictionException(
                f"Prediction failed: {exc}", input_data_info=str(input_df.shape)
            )

        result = pd.DataFrame(
            {
                TARGET: input_df.loc[X.index, TARGET].astype(float),
                "prediction": predictions,
            },
            index=X.index,
        )

        if output_path:
            output_path = Path(output_path)
            create_directory(output_path.parent)
            result.to_csv(output_path)
            self.logger.info(f"Saved predictions to {output_path}")

        return result

    def run(
        self,
        input_path: str,
        output_path: Optional[str] = None,
        local_model_path: Optional[str] = None,
        feature_set: str = "all",
    ) -> pd.DataFrame:
        input_df = self.load_input_data(input_path)
        return self.predict(
            input_df,
            local_model_path=local_model_path,
            output_path=output_path,
            feature_set=feature_set,
        )


def main() -> None:
    logger = get_logger("predict_pipeline_main")
    logger.info("Starting prediction pipeline")

    input_path = Path(get_project_root() / "artifacts" / "data_ingested" / "validation" / "validation_data_2009_2010.csv")
    output_path = get_project_root() / "artifacts" / "predictions" / "validation_predictions.csv"

    try:
        pipeline = PredictionPipeline()
        predictions = pipeline.run(str(input_path), str(output_path))
        logger.info(f"Prediction pipeline completed successfully. Predictions shape: {predictions.shape}")
    except Exception as exc:
        logger.error(f"Prediction pipeline failed: {exc}")
        raise


if __name__ == "__main__":
    main()
