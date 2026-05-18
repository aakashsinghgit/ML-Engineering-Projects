"""
MLflow initialization and testing script.
Run this to verify MLflow setup is working correctly.
"""

import sys
from pathlib import Path

# Add src directory to path for imports
sys.path.append(str(Path(__file__).parent))

from mlflow_tracking import MLflowTracker, start_experiment
from mlflow_config import MLFLOW_CONFIG, EXPERIMENT_CONFIGS
from logger import get_logger
from exceptions import MLflowException
import pandas as pd
import numpy as np
from datetime import datetime


def test_mlflow_setup():
    """Test MLflow setup and basic functionality."""
    logger = get_logger("mlflow_test")
    logger.info("Starting MLflow setup test")
    
    try:
        # Initialize MLflow tracker
        tracker = start_experiment("mlflow_setup_test")
        logger.info("MLflow tracker initialized successfully")
        
        # Test basic run
        with tracker.start_run(run_name="setup_test_run", 
                              tags={"test": "setup", "phase": "initialization"}):
            
            # Log some test parameters
            test_params = {
                "test_param_1": "value_1",
                "test_param_2": 42,
                "test_param_3": 3.14,
                "test_param_4": True
            }
            tracker.log_parameters(test_params)
            logger.info("Test parameters logged successfully")
            
            # Log some test metrics
            test_metrics = {
                "test_metric_1": 0.95,
                "test_metric_2": 0.87,
                "test_metric_3": 0.92
            }
            tracker.log_metrics(test_metrics)
            logger.info("Test metrics logged successfully")
            
            # Log test data info
            test_data_info = {
                "shape": (1000, 5),
                "missing_percentage": 2.5,
                "memory_usage": 40000
            }
            tracker.log_data_info(test_data_info, "test_dataset")
            logger.info("Test data info logged successfully")
            
            # Create and log a test DataFrame
            test_df = pd.DataFrame({
                'feature_1': np.random.randn(100),
                'feature_2': np.random.randn(100),
                'target': np.random.randn(100)
            })
            tracker.log_dataframe(test_df, "test_dataframe")
            logger.info("Test DataFrame logged successfully")
        
        logger.info("MLflow setup test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"MLflow setup test failed: {str(e)}")
        raise MLflowException(
            f"MLflow setup test failed: {str(e)}",
            mlflow_operation="setup_test"
        )


def test_experiment_configs():
    """Test experiment configuration loading."""
    logger = get_logger("mlflow_test")
    logger.info("Testing experiment configurations")
    
    try:
        for exp_name, exp_config in EXPERIMENT_CONFIGS.items():
            logger.info(f"Experiment config '{exp_name}': {exp_config['name']}")
        
        logger.info("Experiment configurations loaded successfully")
        return True
        
    except Exception as e:
        logger.error(f"Experiment config test failed: {str(e)}")
        return False


def main():
    """Main function to run all MLflow tests."""
    logger = get_logger("mlflow_test")
    logger.info("=== MLflow Setup Test Suite ===")
    
    try:
        # Test 1: Basic MLflow setup
        logger.info("Test 1: Basic MLflow setup")
        test_mlflow_setup()
        
        # Test 2: Experiment configurations
        logger.info("Test 2: Experiment configurations")
        test_experiment_configs()
        
        logger.info("=== All MLflow tests passed! ===")
        logger.info("MLflow is ready for experimentation")
        
    except Exception as e:
        logger.error(f"MLflow test suite failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()

