import os
import sys
import logging
from datetime import datetime
from pathlib import Path


class Logger:
    """
    Custom logger class for ML pipeline operations.
    Provides both file and console logging with different log levels.
    """
    
    def __init__(self, name: str = "ml_pipeline", log_level: str = "INFO"):
        """
        Initialize the logger with file and console handlers.
        
        Args:
            name (str): Name of the logger
            log_level (str): Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        # Prevent duplicate handlers
        if self.logger.handlers:
            self.logger.handlers.clear()
        
        # Create logs directory
        self.logs_dir = self._create_logs_directory()
        
        # Setup handlers
        self._setup_file_handler()
        self._setup_console_handler()
        
        # Set format
        self._setup_formatter()
    
    def _create_logs_directory(self) -> Path:
        """Create logs directory if it doesn't exist."""
        project_root = Path(__file__).resolve().parents[2]
        logs_dir = project_root / "artifacts" / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        return logs_dir
    
    def _setup_file_handler(self):
        """Setup file handler for logging to file."""
        timestamp = datetime.now().strftime("%m_%d_%Y_%H_%M")
        log_file = self.logs_dir / f"{timestamp}.log"
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        self.logger.addHandler(file_handler)
    
    def _setup_console_handler(self):
        """Setup console handler for logging to console."""
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        self.logger.addHandler(console_handler)
    
    def _setup_formatter(self):
        """Setup formatter for log messages."""
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        for handler in self.logger.handlers:
            handler.setFormatter(formatter)
    
    def debug(self, message: str):
        """Log debug message."""
        self.logger.debug(message)
    
    def info(self, message: str):
        """Log info message."""
        self.logger.info(message)
    
    def warning(self, message: str):
        """Log warning message."""
        self.logger.warning(message)
    
    def error(self, message: str):
        """Log error message."""
        self.logger.error(message)
    
    def critical(self, message: str):
        """Log critical message."""
        self.logger.critical(message)
    
    def log_data_info(self, data_shape: tuple, data_info: str = ""):
        """Log data information in a structured way."""
        self.info(f"Data Info - Shape: {data_shape}, Details: {data_info}")
    
    def log_model_info(self, model_name: str, model_params: dict = None):
        """Log model information."""
        if model_params:
            self.info(f"Model: {model_name}, Parameters: {model_params}")
        else:
            self.info(f"Model: {model_name}")
    
    def log_performance(self, metric_name: str, metric_value: float):
        """Log model performance metrics."""
        self.info(f"Performance - {metric_name}: {metric_value:.4f}")
    
    def log_file_operation(self, operation: str, file_path: str, success: bool = True):
        """Log file operations."""
        status = "SUCCESS" if success else "FAILED"
        self.info(f"File Operation - {operation}: {file_path} [{status}]")


# Global logger instance
logger = Logger()


def get_logger(name: str = None) -> Logger:
    """
    Get a logger instance.
    
    Args:
        name (str): Optional name for the logger
        
    Returns:
        Logger: Logger instance
    """
    if name:
        return Logger(name)
    return logger


# Convenience functions for quick logging
def log_info(message: str):
    """Quick info logging."""
    logger.info(message)


def log_error(message: str):
    """Quick error logging."""
    logger.error(message)


def log_warning(message: str):
    """Quick warning logging."""
    logger.warning(message)


def log_debug(message: str):
    """Quick debug logging."""
    logger.debug(message)