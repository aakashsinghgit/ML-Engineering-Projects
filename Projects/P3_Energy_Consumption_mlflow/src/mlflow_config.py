"""
MLflow configuration settings for the Energy Consumption ML pipeline.
"""

from pathlib import Path
from typing import Dict, Any

import os

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MLFLOW_DB = PROJECT_ROOT / "mlflow.db"
MODELS_DIR = PROJECT_ROOT / "artifacts" / "models"
EXPERIMENTS_DIR = PROJECT_ROOT / "artifacts" / "experiments"

# MLflow settings - use sqlite backend by default, but respect environment override
MLFLOW_CONFIG = {
    "tracking_uri": os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{MLFLOW_DB.as_posix()}"),
    "experiment_name": "energy_consumption_forecasting",
    "default_tags": {
        "project": "energy_consumption",
        "data_type": "time_series",
        "task": "regression",
        "created_by": "ml_pipeline"
    }
}

# Experiment configurations
EXPERIMENT_CONFIGS = {
    "data_exploration": {
        "name": "data_exploration",
        "description": "Exploratory data analysis and data quality assessment",
        "tags": {"phase": "exploration", "type": "eda"}
    },
    "feature_engineering": {
        "name": "feature_engineering",
        "description": "Feature engineering experiments for time series data",
        "tags": {"phase": "feature_engineering", "type": "preprocessing"}
    },
    "model_comparison": {
        "name": "model_comparison",
        "description": "Comparison of different ML models for energy consumption forecasting",
        "tags": {"phase": "model_selection", "type": "comparison"}
    },
    "hyperparameter_tuning": {
        "name": "hyperparameter_tuning",
        "description": "Hyperparameter optimization for selected models",
        "tags": {"phase": "optimization", "type": "tuning"}
    },
    "production_model": {
        "name": "production_model",
        "description": "Final production-ready model training and validation",
        "tags": {"phase": "production", "type": "final_model"}
    }
}

# Model configurations
MODEL_CONFIGS = {
    "linear_regression": {
        "name": "Linear Regression",
        "type": "sklearn",
        "params": {
            "fit_intercept": True,
            "normalize": False
        }
    },
    "random_forest": {
        "name": "Random Forest",
        "type": "sklearn",
        "params": {
            "n_estimators": 100,
            "max_depth": 10,
            "random_state": 42
        }
    },
    "xgboost": {
        "name": "XGBoost",
        "type": "xgboost",
        "params": {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "random_state": 42
        }
    },
    "lightgbm": {
        "name": "LightGBM",
        "type": "lightgbm",
        "params": {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "random_state": 42
        }
    }
}

# Data split configurations
DATA_SPLITS = {
    "training_years": [2007, 2008],
    "validation_years": [2009, 2010],
    "test_years": [2010],  # Use 2010 as holdout test set
    "features": {
        "target_column": "Global_active_power",
        "time_column": "datetime",
        "numeric_columns": [
            "Global_active_power",
            "Global_reactive_power", 
            "Voltage",
            "Global_intensity",
            "Sub_metering_1",
            "Sub_metering_2",
            "Sub_metering_3"
        ]
    }
}

# Evaluation metrics
EVALUATION_METRICS = {
    "regression": [
        "mse", "rmse", "mae", "mape", "r2_score"
    ],
    "time_series": [
        "mse", "rmse", "mae", "mape", "r2_score", "smape"
    ]
}

# Feature engineering configurations
FEATURE_ENGINEERING_CONFIG = {
    "lag_features": {
        "enabled": True,
        "lags": [1, 2, 3, 6, 12, 24, 48, 72]  # Hours
    },
    "rolling_features": {
        "enabled": True,
        "windows": [6, 12, 24, 48, 72],  # Hours
        "functions": ["mean", "std", "min", "max"]
    },
    "seasonal_features": {
        "enabled": True,
        "hour": True,
        "day_of_week": True,
        "month": True,
        "quarter": True
    },
    "interaction_features": {
        "enabled": True,
        "combinations": [
            ["Global_active_power", "Voltage"],
            ["Global_active_power", "Global_intensity"]
        ]
    }
}

# Hyperparameter search spaces
HYPERPARAMETER_SPACES = {
    "random_forest": {
        "n_estimators": [50, 100, 200, 300],
        "max_depth": [5, 10, 15, 20, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4]
    },
    "xgboost": {
        "n_estimators": [50, 100, 200, 300],
        "max_depth": [3, 6, 9, 12],
        "learning_rate": [0.01, 0.1, 0.2, 0.3],
        "subsample": [0.8, 0.9, 1.0]
    },
    "lightgbm": {
        "n_estimators": [50, 100, 200, 300],
        "max_depth": [3, 6, 9, 12],
        "learning_rate": [0.01, 0.1, 0.2, 0.3],
        "num_leaves": [31, 50, 100, 200]
    }
}

