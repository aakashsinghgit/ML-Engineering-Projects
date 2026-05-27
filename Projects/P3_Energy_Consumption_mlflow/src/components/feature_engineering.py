"""Feature engineering module for time series data."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from logger import get_logger
from exceptions import DataTransformationException

logger = get_logger("feature_engineering")


class FeatureEngineer:
    """Build advanced time-series features for energy consumption forecasting."""

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize feature engineer.
        
        Args:
            config: Feature engineering config dict with keys like 'lag_features', 'rolling_features', etc.
        """
        from mlflow_config import FEATURE_ENGINEERING_CONFIG
        self.config = config or FEATURE_ENGINEERING_CONFIG
        self.logger = logger

    def build_lag_features(self, df: pd.DataFrame, target_col: str = "Global_active_power") -> pd.DataFrame:
        """
        Create lag features (previous time steps).
        
        Args:
            df: DataFrame with target column
            target_col: Name of target column
            
        Returns:
            DataFrame with lag features
        """
        if not self.config.get("lag_features", {}).get("enabled", False):
            return pd.DataFrame(index=df.index)
        
        lags = self.config["lag_features"].get("lags", [1, 2, 3, 6, 12, 24])
        features = pd.DataFrame(index=df.index)
        
        try:
            for lag in lags:
                features[f"{target_col}_lag_{lag}h"] = df[target_col].shift(lag)
            
            self.logger.info(f"Created {len(lags)} lag features")
            return features
        except Exception as exc:
            raise DataTransformationException(
                f"Failed to build lag features: {exc}",
                transformation_step="lag_features"
            )

    def build_rolling_features(self, df: pd.DataFrame, target_col: str = "Global_active_power") -> pd.DataFrame:
        """
        Create rolling window statistics (mean, std, min, max).
        
        Args:
            df: DataFrame with target column
            target_col: Name of target column
            
        Returns:
            DataFrame with rolling features
        """
        if not self.config.get("rolling_features", {}).get("enabled", False):
            return pd.DataFrame(index=df.index)
        
        windows = self.config["rolling_features"].get("windows", [6, 12, 24, 48])
        functions = self.config["rolling_features"].get("functions", ["mean", "std", "min", "max"])
        features = pd.DataFrame(index=df.index)
        
        try:
            for window in windows:
                for func in functions:
                    col_name = f"{target_col}_rolling_{window}h_{func}"
                    features[col_name] = df[target_col].rolling(window=window).agg(func)
            
            self.logger.info(f"Created {len(windows) * len(functions)} rolling features")
            return features
        except Exception as exc:
            raise DataTransformationException(
                f"Failed to build rolling features: {exc}",
                transformation_step="rolling_features"
            )

    def build_seasonal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create calendar/seasonal features (hour, day, month, quarter, is_weekend, etc.).
        
        Args:
            df: DataFrame with datetime index
            
        Returns:
            DataFrame with seasonal features
        """
        if not self.config.get("seasonal_features", {}).get("enabled", False):
            return pd.DataFrame(index=df.index)
        
        features = pd.DataFrame(index=df.index)
        idx = df.index
        
        try:
            if self.config["seasonal_features"].get("hour", True):
                features["hour"] = idx.hour
            
            if self.config["seasonal_features"].get("day_of_week", True):
                features["dayofweek"] = idx.dayofweek
                features["is_weekend"] = (idx.dayofweek >= 5).astype(int)
            
            if self.config["seasonal_features"].get("month", True):
                features["month"] = idx.month
                features["quarter"] = idx.quarter
            
            self.logger.info(f"Created seasonal features with {features.shape[1]} columns")
            return features
        except Exception as exc:
            raise DataTransformationException(
                f"Failed to build seasonal features: {exc}",
                transformation_step="seasonal_features"
            )

    def build_interaction_features(
        self, df: pd.DataFrame, target_col: str = "Global_active_power"
    ) -> pd.DataFrame:
        """
        Create interaction features between key columns.
        
        Args:
            df: DataFrame with numeric columns
            target_col: Name of target column
            
        Returns:
            DataFrame with interaction features
        """
        if not self.config.get("interaction_features", {}).get("enabled", False):
            return pd.DataFrame(index=df.index)
        
        combinations = self.config["interaction_features"].get("combinations", [])
        features = pd.DataFrame(index=df.index)
        
        try:
            for col1, col2 in combinations:
                if col1 in df.columns and col2 in df.columns:
                    # Product interaction
                    features[f"{col1}_x_{col2}"] = df[col1] * df[col2]
                    # Ratio interaction (avoid division by zero)
                    with np.errstate(divide='ignore', invalid='ignore'):
                        ratio = df[col1] / (df[col2] + 1e-8)
                        features[f"{col1}_div_{col2}"] = ratio.replace([np.inf, -np.inf], np.nan)
            
            self.logger.info(f"Created {features.shape[1]} interaction features")
            return features
        except Exception as exc:
            raise DataTransformationException(
                f"Failed to build interaction features: {exc}",
                transformation_step="interaction_features"
            )

    def build_all_features(self, df: pd.DataFrame, target_col: str = "Global_active_power") -> pd.DataFrame:
        """
        Build all configured features.
        
        Args:
            df: Input DataFrame with datetime index and numeric columns
            target_col: Name of target column
            
        Returns:
            DataFrame with all engineered features (excludes target, excludes NaNs)
        """
        self.logger.info(f"Building all features from {df.shape} data")
        
        all_features = pd.DataFrame(index=df.index)
        
        # Build each feature type
        lag_features = self.build_lag_features(df, target_col)
        rolling_features = self.build_rolling_features(df, target_col)
        seasonal_features = self.build_seasonal_features(df)
        interaction_features = self.build_interaction_features(df, target_col)
        
        # Combine
        for feat_df in [lag_features, rolling_features, seasonal_features, interaction_features]:
            all_features = all_features.join(feat_df, how='outer')
        
        # Drop rows with NaN (caused by lags and rolling windows)
        initial_rows = all_features.shape[0]
        all_features = all_features.dropna()
        dropped_rows = initial_rows - all_features.shape[0]
        
        self.logger.info(
            f"Feature engineering complete: {all_features.shape[1]} features, "
            f"dropped {dropped_rows} rows with NaN"
        )
        
        return all_features

    def get_feature_names(self) -> List[str]:
        """Get list of all feature names that will be created."""
        features = []
        
        if self.config.get("lag_features", {}).get("enabled"):
            lags = self.config["lag_features"].get("lags", [])
            features.extend([f"Global_active_power_lag_{lag}h" for lag in lags])
        
        if self.config.get("rolling_features", {}).get("enabled"):
            windows = self.config["rolling_features"].get("windows", [])
            functions = self.config["rolling_features"].get("functions", [])
            for window in windows:
                for func in functions:
                    features.append(f"Global_active_power_rolling_{window}h_{func}")
        
        if self.config.get("seasonal_features", {}).get("enabled"):
            if self.config["seasonal_features"].get("hour"):
                features.append("hour")
            if self.config["seasonal_features"].get("day_of_week"):
                features.extend(["dayofweek", "is_weekend"])
            if self.config["seasonal_features"].get("month"):
                features.extend(["month", "quarter"])
        
        if self.config.get("interaction_features", {}).get("enabled"):
            combinations = self.config["interaction_features"].get("combinations", [])
            for col1, col2 in combinations:
                features.append(f"{col1}_x_{col2}")
                features.append(f"{col1}_div_{col2}")
        
        return features


def demo() -> None:
    """Demo feature engineering on sample data."""
    logger.info("Starting feature engineering demo")
    
    # Create sample time series data
    dates = pd.date_range("2008-01-01", periods=1000, freq="h")
    sample_data = pd.DataFrame({
        "Global_active_power": np.random.randn(1000).cumsum() + 1.0,
        "Global_reactive_power": np.random.randn(1000).cumsum() + 0.5,
        "Voltage": np.random.normal(240, 5, 1000),
        "Global_intensity": np.random.randn(1000).cumsum() + 5.0,
    }, index=dates)
    
    # Build features
    engineer = FeatureEngineer()
    features = engineer.build_all_features(sample_data)
    
    logger.info(f"Sample features shape: {features.shape}")
    logger.info(f"Feature columns: {list(features.columns[:10])}")  # Show first 10


if __name__ == "__main__":
    demo()
