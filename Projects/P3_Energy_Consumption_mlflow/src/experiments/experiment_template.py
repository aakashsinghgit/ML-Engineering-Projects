"""
Template for MLflow-based experimentation with energy consumption data.
This template provides a structured approach to data exploration and model experimentation.
"""

import sys
from pathlib import Path

# Add src/ to path so shared modules (mlflow_tracking, logger, utils, ...) import cleanly
sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, Any, List, Tuple

from mlflow_tracking import MLflowTracker, MLflowRun
from mlflow_config import MLFLOW_CONFIG, DATA_SPLITS, FEATURE_ENGINEERING_CONFIG
from logger import get_logger
from exceptions import DataIngestionException, MLflowException
from utils import get_project_root, get_dataframe_info, validate_data_quality


class EnergyConsumptionExperiment:
    """
    Base class for energy consumption ML experiments.
    Provides common functionality for data loading, preprocessing, and MLflow tracking.
    """
    
    def __init__(self, experiment_name: str):
        """
        Initialize experiment.
        
        Args:
            experiment_name (str): Name of the experiment
        """
        self.logger = get_logger(f"experiment_{experiment_name}")
        self.tracker = MLflowTracker(experiment_name)
        self.project_root = get_project_root()
        self.data_splits = DATA_SPLITS
        
        # Load data
        self.training_data = None
        self.validation_data = None
        self.complete_data = None
        
        self.logger.info(f"Initialized experiment: {experiment_name}")
    
    def load_data(self) -> None:
        """Load training and validation data."""
        try:
            # Load training data
            training_path = (self.project_root / "artifacts" / "data_ingested" / 
                           "training" / "training_data_2007_2008.csv")
            
            if not training_path.exists():
                raise DataIngestionException(
                    f"Training data not found: {training_path}",
                    file_path=str(training_path)
                )
            
            self.training_data = pd.read_csv(training_path, index_col=0, parse_dates=True)
            self.logger.info(f"Loaded training data: {self.training_data.shape}")
            
            # Load validation data
            validation_path = (self.project_root / "artifacts" / "data_ingested" / 
                             "validation" / "validation_data_2009_2010.csv")
            
            if not validation_path.exists():
                raise DataIngestionException(
                    f"Validation data not found: {validation_path}",
                    file_path=str(validation_path)
                )
            
            self.validation_data = pd.read_csv(validation_path, index_col=0, parse_dates=True)
            self.logger.info(f"Loaded validation data: {self.validation_data.shape}")
            
            # Load complete dataset
            complete_path = (self.project_root / "artifacts" / "data_ingested" / 
                           "complete_dataset_2006_2010.csv")
            
            if complete_path.exists():
                self.complete_data = pd.read_csv(complete_path, index_col=0, parse_dates=True)
                self.logger.info(f"Loaded complete data: {self.complete_data.shape}")
            
        except Exception as e:
            raise DataIngestionException(f"Failed to load data: {str(e)}")
    
    def log_data_summary(self, data_name: str, data: pd.DataFrame) -> None:
        """
        Log data summary to MLflow.
        
        Args:
            data_name (str): Name for the data
            data (pd.DataFrame): DataFrame to summarize
        """
        try:
            # Get comprehensive data info
            data_info = get_dataframe_info(data)
            
            # Log basic info
            self.tracker.log_data_info(data_info, data_name)
            
            # Log data quality validation
            quality_report = validate_data_quality(data)
            self.tracker.log_parameters({
                f"{data_name}_quality_valid": quality_report['is_valid'],
                f"{data_name}_quality_issues": str(quality_report['issues'])
            })
            
            self.logger.info(f"Logged data summary for {data_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to log data summary: {str(e)}")
    
    def create_time_series_plots(self, data: pd.DataFrame, data_name: str) -> None:
        """
        Create and log time series plots.
        
        Args:
            data (pd.DataFrame): Time series data
            data_name (str): Name for the data
        """
        try:
            # Set up plotting style
            plt.style.use('seaborn-v0_8')
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(f'Time Series Analysis - {data_name}', fontsize=16)
            
            # Plot 1: Global Active Power over time
            axes[0, 0].plot(data.index, data['Global_active_power'])
            axes[0, 0].set_title('Global Active Power Over Time')
            axes[0, 0].set_ylabel('Power (kW)')
            
            # Plot 2: Hourly average power
            hourly_avg = data.groupby(data.index.hour)['Global_active_power'].mean()
            axes[0, 1].plot(hourly_avg.index, hourly_avg.values)
            axes[0, 1].set_title('Average Power by Hour of Day')
            axes[0, 1].set_xlabel('Hour')
            axes[0, 1].set_ylabel('Average Power (kW)')
            
            # Plot 3: Daily average power
            daily_avg = data.groupby(data.index.date)['Global_active_power'].mean()
            axes[1, 0].plot(daily_avg.index, daily_avg.values)
            axes[1, 0].set_title('Average Power by Day')
            axes[1, 0].set_ylabel('Average Power (kW)')
            
            # Plot 4: Power distribution
            axes[1, 1].hist(data['Global_active_power'], bins=50, alpha=0.7)
            axes[1, 1].set_title('Power Distribution')
            axes[1, 1].set_xlabel('Power (kW)')
            axes[1, 1].set_ylabel('Frequency')
            
            plt.tight_layout()
            
            # Log the figure
            self.tracker.log_figure(fig, f"{data_name}_time_series_analysis")
            plt.close(fig)
            
            self.logger.info(f"Created time series plots for {data_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to create time series plots: {str(e)}")
    
    def create_correlation_heatmap(self, data: pd.DataFrame, data_name: str) -> None:
        """
        Create and log correlation heatmap.
        
        Args:
            data (pd.DataFrame): Data for correlation analysis
            data_name (str): Name for the data
        """
        try:
            # Calculate correlation matrix
            corr_matrix = data.corr()
            
            # Create heatmap
            plt.figure(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                       square=True, linewidths=0.5)
            plt.title(f'Feature Correlation Matrix - {data_name}')
            plt.tight_layout()
            
            # Log the figure
            self.tracker.log_figure(plt.gcf(), f"{data_name}_correlation_heatmap")
            plt.close()
            
            # Log correlation values
            self.tracker.log_parameters({
                f"{data_name}_max_correlation": corr_matrix.max().max(),
                f"{data_name}_min_correlation": corr_matrix.min().min()
            })
            
            self.logger.info(f"Created correlation heatmap for {data_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to create correlation heatmap: {str(e)}")
    
    def run_experiment(self, run_name: str, tags: Dict[str, str] = None) -> None:
        """
        Run the experiment with MLflow tracking.
        
        Args:
            run_name (str): Name for the run
            tags (Dict[str, str]): Tags for the run
        """
        try:
            with MLflowRun(self.tracker, run_name, tags):
                # Load data
                self.load_data()
                
                # Log data summaries
                if self.training_data is not None:
                    self.log_data_summary("training", self.training_data)
                    self.create_time_series_plots(self.training_data, "training")
                    self.create_correlation_heatmap(self.training_data, "training")
                
                if self.validation_data is not None:
                    self.log_data_summary("validation", self.validation_data)
                    self.create_time_series_plots(self.validation_data, "validation")
                    self.create_correlation_heatmap(self.validation_data, "validation")
                
                if self.complete_data is not None:
                    self.log_data_summary("complete", self.complete_data)
                
                # Log experiment metadata
                self.tracker.log_parameters({
                    "experiment_type": "data_exploration",
                    "data_source": "household_power_consumption",
                    "training_years": str(self.data_splits["training_years"]),
                    "validation_years": str(self.data_splits["validation_years"])
                })
                
                self.logger.info(f"Experiment '{run_name}' completed successfully")
                
        except Exception as e:
            self.logger.error(f"Experiment '{run_name}' failed: {str(e)}")
            raise


def main():
    """Main function to run data exploration experiment."""
    logger = get_logger("experiment_main")
    logger.info("Starting Energy Consumption Data Exploration Experiment")
    
    try:
        # Create and run experiment
        experiment = EnergyConsumptionExperiment("data_exploration")
        experiment.run_experiment(
            run_name=f"data_exploration_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            tags={"phase": "exploration", "type": "eda"}
        )
        
        logger.info("Data exploration experiment completed successfully!")
        
    except Exception as e:
        logger.error(f"Data exploration experiment failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()

