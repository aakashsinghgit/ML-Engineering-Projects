"""Hyperparameter tuning component."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from logger import get_logger
from exceptions import ModelTrainingException

logger = get_logger("hyperparameter_tuner")

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


class HyperparameterTuner:
    """Grid search and random search for hyperparameter optimization."""

    def __init__(self, n_jobs: int = -1, cv: int = 5, random_state: int = 42):
        """
        Initialize tuner.
        
        Args:
            n_jobs: Number of parallel jobs (-1 = use all cores)
            cv: Number of cross-validation folds
            random_state: Random seed
        """
        self.logger = logger
        self.n_jobs = n_jobs
        self.cv = cv
        self.random_state = random_state
        self.best_model = None
        self.best_params = None
        self.best_score = None

    def get_param_grid(self, model_name: str, grid_type: str = "coarse") -> Dict[str, List]:
        """
        Get parameter grid for a model.
        
        Args:
            model_name: Name of model
            grid_type: "coarse" (default, fast) or "fine" (thorough)
            
        Returns:
            Parameter grid dict
        """
        grids = {
            "random_forest": {
                "coarse": {
                    "n_estimators": [50, 100, 200],
                    "max_depth": [5, 10, 15],
                    "min_samples_split": [2, 5],
                },
                "fine": {
                    "n_estimators": [50, 100, 150, 200, 300],
                    "max_depth": [5, 10, 15, 20, None],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4],
                }
            },
            "xgboost": {
                "coarse": {
                    "n_estimators": [50, 100, 200],
                    "max_depth": [3, 6, 9],
                    "learning_rate": [0.01, 0.1, 0.2],
                },
                "fine": {
                    "n_estimators": [50, 100, 150, 200, 300],
                    "max_depth": [3, 6, 9, 12],
                    "learning_rate": [0.001, 0.01, 0.1, 0.2],
                    "subsample": [0.8, 0.9, 1.0],
                }
            },
            "lightgbm": {
                "coarse": {
                    "n_estimators": [50, 100, 200],
                    "max_depth": [3, 6, 9],
                    "learning_rate": [0.01, 0.1, 0.2],
                },
                "fine": {
                    "n_estimators": [50, 100, 150, 200, 300],
                    "max_depth": [3, 6, 9, 12],
                    "learning_rate": [0.001, 0.01, 0.1, 0.2],
                    "num_leaves": [31, 50, 100],
                }
            }
        }
        
        return grids.get(model_name, {}).get(grid_type, {})

    def get_base_model(self, model_name: str):
        """Get base model for tuning."""
        if model_name == "random_forest":
            return RandomForestRegressor(random_state=self.random_state, n_jobs=-1)
        elif model_name == "xgboost":
            if not XGBOOST_AVAILABLE:
                raise ModelTrainingException(
                    "XGBoost not available",
                    model_name="xgboost"
                )
            return XGBRegressor(random_state=self.random_state, verbosity=0)
        elif model_name == "lightgbm":
            if not LIGHTGBM_AVAILABLE:
                raise ModelTrainingException(
                    "LightGBM not available",
                    model_name="lightgbm"
                )
            return LGBMRegressor(random_state=self.random_state, verbose=-1)
        else:
            raise ValueError(f"Unknown model: {model_name}")

    def grid_search(
        self,
        model_name: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        param_grid: Optional[Dict] = None,
        grid_type: str = "coarse",
    ) -> Tuple[Dict, float]:
        """
        Perform grid search cross-validation.
        
        Args:
            model_name: Model to tune
            X_train: Training features
            y_train: Training target
            param_grid: Parameter grid (default from get_param_grid)
            grid_type: "coarse" or "fine"
            
        Returns:
            (best_params, best_score)
        """
        self.logger.info(f"Starting grid search for {model_name} ({grid_type})")
        
        param_grid = param_grid or self.get_param_grid(model_name, grid_type)
        base_model = self.get_base_model(model_name)
        
        try:
            search = GridSearchCV(
                base_model,
                param_grid,
                cv=self.cv,
                n_jobs=self.n_jobs,
                scoring="r2",
                verbose=1
            )
            
            search.fit(X_train, y_train)
            
            self.best_model = search.best_estimator_
            self.best_params = search.best_params_
            self.best_score = search.best_score_
            
            self.logger.info(f"Best score: {self.best_score:.4f}")
            self.logger.info(f"Best params: {self.best_params}")
            
            return self.best_params, self.best_score
            
        except Exception as exc:
            raise ModelTrainingException(
                f"Grid search failed for {model_name}: {exc}",
                model_name=model_name
            )

    def random_search(
        self,
        model_name: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        param_distributions: Optional[Dict] = None,
        n_iter: int = 20,
        grid_type: str = "coarse",
    ) -> Tuple[Dict, float]:
        """
        Perform random search cross-validation.
        
        Args:
            model_name: Model to tune
            X_train: Training features
            y_train: Training target
            param_distributions: Parameter distributions
            n_iter: Number of iterations
            grid_type: "coarse" or "fine"
            
        Returns:
            (best_params, best_score)
        """
        self.logger.info(f"Starting random search for {model_name} ({n_iter} iterations)")
        
        param_distributions = param_distributions or self.get_param_grid(model_name, grid_type)
        base_model = self.get_base_model(model_name)
        
        try:
            search = RandomizedSearchCV(
                base_model,
                param_distributions,
                n_iter=n_iter,
                cv=self.cv,
                n_jobs=self.n_jobs,
                scoring="r2",
                random_state=self.random_state,
                verbose=1
            )
            
            search.fit(X_train, y_train)
            
            self.best_model = search.best_estimator_
            self.best_params = search.best_params_
            self.best_score = search.best_score_
            
            self.logger.info(f"Best score: {self.best_score:.4f}")
            self.logger.info(f"Best params: {self.best_params}")
            
            return self.best_params, self.best_score
            
        except Exception as exc:
            raise ModelTrainingException(
                f"Random search failed for {model_name}: {exc}",
                model_name=model_name
            )

    def tune_model(
        self,
        model_name: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        method: str = "grid",
        grid_type: str = "coarse",
        **kwargs
    ) -> Tuple[Dict, float]:
        """
        Tune hyperparameters using specified method.
        
        Args:
            model_name: Model to tune
            X_train: Training features
            y_train: Training target
            method: "grid" or "random"
            grid_type: "coarse" or "fine"
            **kwargs: Additional arguments for random search
            
        Returns:
            (best_params, best_score)
        """
        if method == "grid":
            return self.grid_search(model_name, X_train, y_train, grid_type=grid_type)
        elif method == "random":
            n_iter = kwargs.get("n_iter", 20)
            return self.random_search(
                model_name, X_train, y_train,
                n_iter=n_iter, grid_type=grid_type
            )
        else:
            raise ValueError(f"Unknown tuning method: {method}")


def demo() -> None:
    """Demo hyperparameter tuning."""
    logger.info("Starting hyperparameter tuning demo")
    
    # Create sample data
    np.random.seed(42)
    X = pd.DataFrame(
        np.random.randn(200, 10),
        columns=[f"feature_{i}" for i in range(10)]
    )
    y = pd.Series(np.random.randn(200))
    
    tuner = HyperparameterTuner(cv=3, n_jobs=-1)
    
    # Tune RandomForest (coarse)
    best_params, best_score = tuner.tune_model(
        "random_forest",
        X, y,
        method="grid",
        grid_type="coarse"
    )
    logger.info(f"Best RandomForest params: {best_params}, Score: {best_score}")


if __name__ == "__main__":
    demo()
