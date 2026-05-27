"""Model training component with support for multiple algorithms."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from typing import Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from logger import get_logger
from exceptions import ModelTrainingException

logger = get_logger("model_trainer")

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMRegressor
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False


class ModelTrainer:
    """Flexible trainer for multiple regression models."""

    def __init__(self):
        self.logger = logger
        self.models = {}
        self.trained_model = None
        self.model_type = None

    @staticmethod
    def evaluate(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate evaluation metrics."""
        return {
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "r2": float(r2_score(y_true, y_pred)),
            "mape": float(np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8)))) * 100,
        }

    def train_linear_regression(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        params: Optional[Dict[str, Any]] = None,
    ):
        """Train linear regression model."""
        params = params or {"fit_intercept": True}
        self.logger.info(f"Training LinearRegression with params: {params}")
        
        try:
            model = LinearRegression(**params)
            model.fit(X_train, y_train)
            self.trained_model = model
            self.model_type = "linear_regression"
            return model
        except Exception as exc:
            raise ModelTrainingException(
                f"LinearRegression training failed: {exc}",
                model_name="linear_regression"
            )

    def train_random_forest(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        params: Optional[Dict[str, Any]] = None,
    ):
        """Train RandomForest model."""
        params = params or {
            "n_estimators": 100,
            "max_depth": 10,
            "random_state": 42
        }
        self.logger.info(f"Training RandomForest with params: {params}")
        
        try:
            model = RandomForestRegressor(**params)
            model.fit(X_train, y_train)
            self.trained_model = model
            self.model_type = "random_forest"
            return model
        except Exception as exc:
            raise ModelTrainingException(
                f"RandomForest training failed: {exc}",
                model_name="random_forest"
            )

    def train_xgboost(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        params: Optional[Dict[str, Any]] = None,
    ):
        """Train XGBoost model."""
        if not XGBOOST_AVAILABLE:
            raise ModelTrainingException(
                "XGBoost not available. Install with: pip install xgboost",
                model_name="xgboost"
            )
        
        params = params or {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "random_state": 42
        }
        self.logger.info(f"Training XGBoost with params: {params}")
        
        try:
            model = XGBRegressor(**params, verbosity=0)
            model.fit(X_train, y_train)
            self.trained_model = model
            self.model_type = "xgboost"
            return model
        except Exception as exc:
            raise ModelTrainingException(
                f"XGBoost training failed: {exc}",
                model_name="xgboost"
            )

    def train_lightgbm(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        params: Optional[Dict[str, Any]] = None,
    ):
        """Train LightGBM model."""
        if not LIGHTGBM_AVAILABLE:
            raise ModelTrainingException(
                "LightGBM not available. Install with: pip install lightgbm",
                model_name="lightgbm"
            )
        
        params = params or {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "random_state": 42,
            "verbose": -1
        }
        self.logger.info(f"Training LightGBM with params: {params}")
        
        try:
            model = LGBMRegressor(**params)
            model.fit(X_train, y_train)
            self.trained_model = model
            self.model_type = "lightgbm"
            return model
        except Exception as exc:
            raise ModelTrainingException(
                f"LightGBM training failed: {exc}",
                model_name="lightgbm"
            )

    def train_ensemble(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        model_configs: Optional[list] = None,
    ):
        """Train ensemble of multiple models."""
        if model_configs is None:
            model_configs = [
                ("linear_regression", "linear_regression", {}),
                ("random_forest", "random_forest", {}),
            ]
            if XGBOOST_AVAILABLE:
                model_configs.append(("xgboost", "xgboost", {}))
        
        self.logger.info(f"Training ensemble with {len(model_configs)} base models")
        
        estimators = []
        
        try:
            for name, model_type, params in model_configs:
                self.logger.info(f"Training base model: {name}")
                
                if model_type == "linear_regression":
                    model = self.train_linear_regression(X_train, y_train, params)
                elif model_type == "random_forest":
                    model = self.train_random_forest(X_train, y_train, params)
                elif model_type == "xgboost":
                    model = self.train_xgboost(X_train, y_train, params)
                elif model_type == "lightgbm":
                    model = self.train_lightgbm(X_train, y_train, params)
                else:
                    raise ValueError(f"Unknown model type: {model_type}")
                
                estimators.append((name, model))
            
            # Create voting regressor
            ensemble = VotingRegressor(estimators=estimators)
            ensemble.fit(X_train, y_train)
            self.trained_model = ensemble
            self.model_type = "ensemble"
            return ensemble
            
        except Exception as exc:
            raise ModelTrainingException(
                f"Ensemble training failed: {exc}",
                model_name="ensemble"
            )

    def train(
        self,
        model_name: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        params: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """
        Train a model by name.
        
        Args:
            model_name: Name of model to train
            X_train: Training features
            y_train: Training target
            params: Model-specific parameters
            **kwargs: Additional arguments for ensemble
            
        Returns:
            Trained model
        """
        if model_name == "linear_regression":
            return self.train_linear_regression(X_train, y_train, params)
        elif model_name == "random_forest":
            return self.train_random_forest(X_train, y_train, params)
        elif model_name == "xgboost":
            return self.train_xgboost(X_train, y_train, params)
        elif model_name == "lightgbm":
            return self.train_lightgbm(X_train, y_train, params)
        elif model_name == "ensemble":
            return self.train_ensemble(X_train, y_train, kwargs.get("model_configs"))
        else:
            raise ValueError(f"Unknown model: {model_name}")

    def predict(self, X) -> np.ndarray:
        """Make predictions with trained model."""
        if self.trained_model is None:
            raise ModelTrainingException(
                "No model trained yet",
                model_name=self.model_type or "unknown"
            )
        return self.trained_model.predict(X)

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance if available."""
        if self.trained_model is None:
            return None
        
        if hasattr(self.trained_model, "feature_importances_"):
            # Tree-based models
            return dict(zip(
                [f"feature_{i}" for i in range(len(self.trained_model.feature_importances_))],
                self.trained_model.feature_importances_.tolist()
            ))
        elif hasattr(self.trained_model, "coef_"):
            # Linear models
            return dict(zip(
                [f"feature_{i}" for i in range(len(self.trained_model.coef_))],
                np.abs(self.trained_model.coef_).tolist()
            ))
        
        return None


def demo() -> None:
    """Demo model training on sample data."""
    logger.info("Starting model trainer demo")
    
    # Create sample data
    np.random.seed(42)
    X = pd.DataFrame(
        np.random.randn(100, 10),
        columns=[f"feature_{i}" for i in range(10)]
    )
    y = pd.Series(np.random.randn(100))
    
    trainer = ModelTrainer()
    
    # Train different models
    for model_name in ["linear_regression", "random_forest"]:
        model = trainer.train(model_name, X, y)
        y_pred = trainer.predict(X)
        metrics = trainer.evaluate(y, y_pred)
        logger.info(f"{model_name} metrics: {metrics}")


if __name__ == "__main__":
    demo()
