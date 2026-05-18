"""
Custom exception classes for ML pipeline operations.
Provides specific error handling for different stages of the ML pipeline.
"""


class MLPipelineException(Exception):
    """Base exception class for ML pipeline operations."""
    
    def __init__(self, message: str, error_code: str = None):
        """
        Initialize ML pipeline exception.
        
        Args:
            message (str): Error message
            error_code (str): Optional error code for categorization
        """
        self.message = message
        self.error_code = error_code
        super().__init__(self.message)
    
    def __str__(self):
        if self.error_code:
            return f"[{self.error_code}] {self.message}"
        return self.message


class DataIngestionException(MLPipelineException):
    """Exception raised during data ingestion operations."""
    
    def __init__(self, message: str, file_path: str = None):
        """
        Initialize data ingestion exception.
        
        Args:
            message (str): Error message
            file_path (str): Path to the file that caused the error
        """
        self.file_path = file_path
        super().__init__(message, "DATA_INGESTION_ERROR")
    
    def __str__(self):
        base_msg = super().__str__()
        if self.file_path:
            return f"{base_msg} - File: {self.file_path}"
        return base_msg


class DataValidationException(MLPipelineException):
    """Exception raised during data validation operations."""
    
    def __init__(self, message: str, validation_type: str = None):
        """
        Initialize data validation exception.
        
        Args:
            message (str): Error message
            validation_type (str): Type of validation that failed
        """
        self.validation_type = validation_type
        super().__init__(message, "DATA_VALIDATION_ERROR")
    
    def __str__(self):
        base_msg = super().__str__()
        if self.validation_type:
            return f"{base_msg} - Validation Type: {self.validation_type}"
        return base_msg


class DataTransformationException(MLPipelineException):
    """Exception raised during data transformation operations."""
    
    def __init__(self, message: str, transformation_step: str = None):
        """
        Initialize data transformation exception.
        
        Args:
            message (str): Error message
            transformation_step (str): Step in transformation that failed
        """
        self.transformation_step = transformation_step
        super().__init__(message, "DATA_TRANSFORMATION_ERROR")
    
    def __str__(self):
        base_msg = super().__str__()
        if self.transformation_step:
            return f"{base_msg} - Step: {self.transformation_step}"
        return base_msg


class ModelTrainingException(MLPipelineException):
    """Exception raised during model training operations."""
    
    def __init__(self, message: str, model_name: str = None):
        """
        Initialize model training exception.
        
        Args:
            message (str): Error message
            model_name (str): Name of the model that failed to train
        """
        self.model_name = model_name
        super().__init__(message, "MODEL_TRAINING_ERROR")
    
    def __str__(self):
        base_msg = super().__str__()
        if self.model_name:
            return f"{base_msg} - Model: {self.model_name}"
        return base_msg


class ModelEvaluationException(MLPipelineException):
    """Exception raised during model evaluation operations."""
    
    def __init__(self, message: str, evaluation_metric: str = None):
        """
        Initialize model evaluation exception.
        
        Args:
            message (str): Error message
            evaluation_metric (str): Metric that failed during evaluation
        """
        self.evaluation_metric = evaluation_metric
        super().__init__(message, "MODEL_EVALUATION_ERROR")
    
    def __str__(self):
        base_msg = super().__str__()
        if self.evaluation_metric:
            return f"{base_msg} - Metric: {self.evaluation_metric}"
        return base_msg


class ModelPredictionException(MLPipelineException):
    """Exception raised during model prediction operations."""
    
    def __init__(self, message: str, input_data_info: str = None):
        """
        Initialize model prediction exception.
        
        Args:
            message (str): Error message
            input_data_info (str): Information about input data that caused error
        """
        self.input_data_info = input_data_info
        super().__init__(message, "MODEL_PREDICTION_ERROR")
    
    def __str__(self):
        base_msg = super().__str__()
        if self.input_data_info:
            return f"{base_msg} - Input Data: {self.input_data_info}"
        return base_msg


class FileOperationException(MLPipelineException):
    """Exception raised during file operations."""
    
    def __init__(self, message: str, file_path: str = None, operation: str = None):
        """
        Initialize file operation exception.
        
        Args:
            message (str): Error message
            file_path (str): Path to the file involved in the operation
            operation (str): Type of file operation (read, write, delete, etc.)
        """
        self.file_path = file_path
        self.operation = operation
        super().__init__(message, "FILE_OPERATION_ERROR")
    
    def __str__(self):
        base_msg = super().__str__()
        details = []
        if self.file_path:
            details.append(f"File: {self.file_path}")
        if self.operation:
            details.append(f"Operation: {self.operation}")
        
        if details:
            return f"{base_msg} - {', '.join(details)}"
        return base_msg


class ConfigurationException(MLPipelineException):
    """Exception raised for configuration-related errors."""
    
    def __init__(self, message: str, config_key: str = None):
        """
        Initialize configuration exception.
        
        Args:
            message (str): Error message
            config_key (str): Configuration key that caused the error
        """
        self.config_key = config_key
        super().__init__(message, "CONFIGURATION_ERROR")
    
    def __str__(self):
        base_msg = super().__str__()
        if self.config_key:
            return f"{base_msg} - Config Key: {self.config_key}"
        return base_msg


class MLflowException(MLPipelineException):
    """Exception raised for MLflow-related operations."""
    
    def __init__(self, message: str, mlflow_operation: str = None):
        """
        Initialize MLflow exception.
        
        Args:
            message (str): Error message
            mlflow_operation (str): MLflow operation that failed
        """
        self.mlflow_operation = mlflow_operation
        super().__init__(message, "MLFLOW_ERROR")
    
    def __str__(self):
        base_msg = super().__str__()
        if self.mlflow_operation:
            return f"{base_msg} - Operation: {self.mlflow_operation}"
        return base_msg


# Utility functions for exception handling
def handle_exception(exception: Exception, logger=None, reraise: bool = True):
    """
    Handle exceptions with proper logging.
    
    Args:
        exception (Exception): The exception to handle
        logger: Logger instance for logging the exception
        reraise (bool): Whether to reraise the exception after logging
    
    Returns:
        bool: True if exception was handled successfully
    """
    if logger:
        logger.error(f"Exception occurred: {str(exception)}")
        logger.debug(f"Exception details: {type(exception).__name__}", exc_info=True)
    
    if reraise:
        raise exception
    
    return True


def validate_file_exists(file_path: str, operation: str = "access") -> bool:
    """
    Validate that a file exists and raise appropriate exception if not.
    
    Args:
        file_path (str): Path to the file to validate
        operation (str): Operation being performed on the file
    
    Returns:
        bool: True if file exists
    
    Raises:
        FileOperationException: If file doesn't exist
    """
    import os
    
    if not os.path.exists(file_path):
        raise FileOperationException(
            f"File not found: {file_path}",
            file_path=file_path,
            operation=operation
        )
    
    return True


def validate_dataframe_not_empty(df, operation: str = "process") -> bool:
    """
    Validate that a DataFrame is not empty.
    
    Args:
        df: Pandas DataFrame to validate
        operation (str): Operation being performed on the DataFrame
    
    Returns:
        bool: True if DataFrame is not empty
    
    Raises:
        DataValidationException: If DataFrame is empty
    """
    if df.empty:
        raise DataValidationException(
            f"DataFrame is empty for operation: {operation}",
            validation_type="empty_dataframe"
        )
    
    return True
