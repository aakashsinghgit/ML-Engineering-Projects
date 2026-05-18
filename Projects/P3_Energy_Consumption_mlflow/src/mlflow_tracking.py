"""
MLflow tracking and experiment management for the Energy Consumption ML pipeline.
Provides comprehensive experiment tracking, model logging, and artifact management.
"""

import os
import mlflow
import mlflow.sklearn
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import pandas as pd
import numpy as np
from datetime import datetime

# Import our custom modules
from logger import get_logger
from exceptions import MLflowException
from utils import get_project_root, get_timestamp


class MLflowTracker:
    """
    Comprehensive MLflow tracking class for ML experiments.
    Handles experiment creation, run tracking, model logging, and artifact management.
    """
    
    def __init__(self, experiment_name: str = "energy_consumption_forecasting", 
                 tracking_uri: Optional[str] = None):
        """
        Initialize MLflow tracker.
        
        Args:
            experiment_name (str): Name of the MLflow experiment
            tracking_uri (Optional[str]): MLflow tracking URI (defaults to local)
        """
        self.logger = get_logger("mlflow_tracker")
        self.experiment_name = experiment_name
        
        # Set up MLflow tracking - sqlite store at project root so the UI
        # (started with `mlflow ui --backend-store-uri sqlite:///mlflow.db`
        # from the P3 folder) and this script share the same store.
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        else:
            project_root = get_project_root()
            db_path = (project_root / "mlflow.db").as_posix()
            mlflow.set_tracking_uri(f"sqlite:///{db_path}")
        
        self.logger.info(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")
        
        # Create or get experiment
        self.experiment_id = self._setup_experiment()
        self.logger.info(f"Experiment '{experiment_name}' ready (ID: {self.experiment_id})")
    
    def _setup_experiment(self) -> str:
        """Create or get existing experiment."""
        try:
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(self.experiment_name)
                self.logger.info(f"Created new experiment: {self.experiment_name}")
            else:
                experiment_id = experiment.experiment_id
                self.logger.info(f"Using existing experiment: {self.experiment_name}")
            
            return experiment_id
        except Exception as e:
            raise MLflowException(
                f"Failed to setup experiment: {str(e)}",
                mlflow_operation="experiment_setup"
            )
    
    def start_run(self, run_name: Optional[str] = None, 
                  tags: Optional[Dict[str, str]] = None) -> mlflow.ActiveRun:
        """
        Start a new MLflow run.
        
        Args:
            run_name (Optional[str]): Name for the run
            tags (Optional[Dict[str, str]]): Tags for the run
        
        Returns:
            mlflow.ActiveRun: Active run object
        """
        try:
            if run_name is None:
                run_name = f"run_{get_timestamp()}"
            
            run = mlflow.start_run(
                experiment_id=self.experiment_id,
                run_name=run_name,
                tags=tags or {}
            )
            
            self.logger.info(f"Started MLflow run: {run_name}")
            return run
            
        except Exception as e:
            raise MLflowException(
                f"Failed to start run: {str(e)}",
                mlflow_operation="start_run"
            )
    
    def log_parameters(self, params: Dict[str, Any]) -> None:
        """
        Log parameters to the current run.
        
        Args:
            params (Dict[str, Any]): Parameters to log
        """
        try:
            mlflow.log_params(params)
            self.logger.info(f"Logged {len(params)} parameters")
        except Exception as e:
            raise MLflowException(
                f"Failed to log parameters: {str(e)}",
                mlflow_operation="log_params"
            )
    
    def log_metrics(self, metrics: Dict[str, float]) -> None:
        """
        Log metrics to the current run.
        
        Args:
            metrics (Dict[str, float]): Metrics to log
        """
        try:
            mlflow.log_metrics(metrics)
            self.logger.info(f"Logged {len(metrics)} metrics")
        except Exception as e:
            raise MLflowException(
                f"Failed to log metrics: {str(e)}",
                mlflow_operation="log_metrics"
            )
    
    def log_data_info(self, data_info: Dict[str, Any], data_name: str = "dataset") -> None:
        """
        Log data information as parameters.
        
        Args:
            data_info (Dict[str, Any]): Data information dictionary
            data_name (str): Name prefix for data parameters
        """
        try:
            params = {}
            for key, value in data_info.items():
                if isinstance(value, (int, float, str, bool)):
                    params[f"{data_name}_{key}"] = value
                else:
                    params[f"{data_name}_{key}"] = str(value)
            
            mlflow.log_params(params)
            self.logger.info(f"Logged data info for {data_name}")
        except Exception as e:
            raise MLflowException(
                f"Failed to log data info: {str(e)}",
                mlflow_operation="log_data_info"
            )
    
    def log_model(self, model: Any, model_name: str, 
                  model_type: str = "sklearn") -> None:
        """
        Log a trained model.
        
        Args:
            model (Any): Trained model object
            model_name (str): Name for the model
            model_type (str): Type of model (sklearn, xgboost, lightgbm, pytorch, tensorflow)
        """
        try:
            # Lazy-import framework flavors via importlib to avoid the
            # `import mlflow.X` statement turning `mlflow` into a local
            # name and shadowing the module-level import.
            import importlib

            if model_type == "sklearn":
                mlflow.sklearn.log_model(model, model_name)
            elif model_type in ("xgboost", "lightgbm", "pytorch", "tensorflow"):
                flavor = importlib.import_module(f"mlflow.{model_type}")
                flavor.log_model(model, model_name)
            else:
                mlflow.sklearn.log_model(model, model_name)
            
            self.logger.info(f"Logged {model_type} model: {model_name}")
        except Exception as e:
            raise MLflowException(
                f"Failed to log model: {str(e)}",
                mlflow_operation="log_model"
            )
    
    def log_artifacts(self, artifacts_dir: Union[str, Path], 
                      artifact_path: Optional[str] = None) -> None:
        """
        Log artifacts (files) to the current run.
        
        Args:
            artifacts_dir (Union[str, Path]): Directory containing artifacts
            artifact_path (Optional[str]): Path within the run to store artifacts
        """
        try:
            artifacts_dir = Path(artifacts_dir)
            if artifacts_dir.exists():
                mlflow.log_artifacts(str(artifacts_dir), artifact_path)
                self.logger.info(f"Logged artifacts from: {artifacts_dir}")
            else:
                self.logger.warning(f"Artifacts directory not found: {artifacts_dir}")
        except Exception as e:
            raise MLflowException(
                f"Failed to log artifacts: {str(e)}",
                mlflow_operation="log_artifacts"
            )
    
    def log_figure(self, figure, figure_name: str) -> None:
        """
        Log a matplotlib figure.
        
        Args:
            figure: Matplotlib figure object
            figure_name (str): Name for the figure
        """
        try:
            mlflow.log_figure(figure, figure_name)
            self.logger.info(f"Logged figure: {figure_name}")
        except Exception as e:
            raise MLflowException(
                f"Failed to log figure: {str(e)}",
                mlflow_operation="log_figure"
            )
    
    def log_dataframe(self, df: pd.DataFrame, df_name: str, 
                     format: str = "csv") -> None:
        """
        Log a pandas DataFrame as an artifact.
        
        Args:
            df (pd.DataFrame): DataFrame to log
            df_name (str): Name for the DataFrame
            format (str): Format to save (csv, parquet, json)
        """
        try:
            if format == "csv":
                mlflow.log_table(df, df_name)
            else:
                # Save to temporary file and log as artifact
                temp_path = Path(f"/tmp/{df_name}.{format}")
                if format == "parquet":
                    df.to_parquet(temp_path)
                elif format == "json":
                    df.to_json(temp_path)
                
                mlflow.log_artifact(str(temp_path), df_name)
                temp_path.unlink()  # Clean up temp file
            
            self.logger.info(f"Logged DataFrame: {df_name} ({format})")
        except Exception as e:
            raise MLflowException(
                f"Failed to log DataFrame: {str(e)}",
                mlflow_operation="log_dataframe"
            )
    
    def end_run(self, status: str = "FINISHED") -> None:
        """
        End the current MLflow run.
        
        Args:
            status (str): Run status (FINISHED, FAILED, KILLED)
        """
        try:
            mlflow.end_run(status=status)
            self.logger.info(f"Ended MLflow run with status: {status}")
        except Exception as e:
            raise MLflowException(
                f"Failed to end run: {str(e)}",
                mlflow_operation="end_run"
            )
    
    def get_best_run(self, metric_name: str, ascending: bool = True) -> Optional[Dict]:
        """
        Get the best run based on a metric.
        
        Args:
            metric_name (str): Name of the metric to optimize
            ascending (bool): Whether lower values are better
        
        Returns:
            Optional[Dict]: Best run information
        """
        try:
            experiment = mlflow.get_experiment(self.experiment_id)
            runs = mlflow.search_runs(experiment_ids=[self.experiment_id])
            
            if runs.empty:
                return None
            
            if metric_name not in runs.columns:
                self.logger.warning(f"Metric '{metric_name}' not found in runs")
                return None
            
            # Filter out runs without the metric
            runs_with_metric = runs.dropna(subset=[metric_name])
            if runs_with_metric.empty:
                return None
            
            best_run = runs_with_metric.loc[runs_with_metric[metric_name].idxmin() if ascending 
                                          else runs_with_metric[metric_name].idxmax()]
            
            return best_run.to_dict()
        except Exception as e:
            raise MLflowException(
                f"Failed to get best run: {str(e)}",
                mlflow_operation="get_best_run"
            )
    
    def compare_runs(self, run_ids: List[str]) -> pd.DataFrame:
        """
        Compare multiple runs.
        
        Args:
            run_ids (List[str]): List of run IDs to compare
        
        Returns:
            pd.DataFrame: Comparison results
        """
        try:
            runs = mlflow.search_runs(run_ids=run_ids)
            self.logger.info(f"Compared {len(run_ids)} runs")
            return runs
        except Exception as e:
            raise MLflowException(
                f"Failed to compare runs: {str(e)}",
                mlflow_operation="compare_runs"
            )


# Convenience functions for quick MLflow operations
def start_experiment(experiment_name: str = "energy_consumption_forecasting") -> MLflowTracker:
    """
    Quick function to start an MLflow experiment.
    
    Args:
        experiment_name (str): Name of the experiment
    
    Returns:
        MLflowTracker: Configured tracker instance
    """
    return MLflowTracker(experiment_name)


def log_model_performance(model, model_name: str, metrics: Dict[str, float],
                         params: Dict[str, Any], model_type: str = "sklearn") -> None:
    """
    Quick function to log a complete model with performance metrics.
    
    Args:
        model: Trained model
        model_name (str): Name for the model
        metrics (Dict[str, float]): Performance metrics
        params (Dict[str, Any]): Model parameters
        model_type (str): Type of model
    """
    tracker = MLflowTracker()
    
    with tracker.start_run(run_name=f"{model_name}_{get_timestamp()}"):
        tracker.log_parameters(params)
        tracker.log_metrics(metrics)
        tracker.log_model(model, model_name, model_type)
        tracker.end_run()


# Context manager for easy run management
class MLflowRun:
    """Context manager for MLflow runs."""
    
    def __init__(self, tracker: MLflowTracker, run_name: Optional[str] = None,
                 tags: Optional[Dict[str, str]] = None):
        self.tracker = tracker
        self.run_name = run_name
        self.tags = tags
        self.run = None
    
    def __enter__(self):
        self.run = self.tracker.start_run(self.run_name, self.tags)
        return self.tracker
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        status = "FAILED" if exc_type else "FINISHED"
        self.tracker.end_run(status)
