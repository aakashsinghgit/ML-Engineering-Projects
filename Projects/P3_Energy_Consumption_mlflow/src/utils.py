"""
Utility functions for ML pipeline operations.
Contains common helper functions used across the ML pipeline.
"""

import os
import json
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime
import yaml


def get_project_root() -> Path:
    """
    Get the project root directory.
    
    Returns:
        Path: Path to the project root directory
    """
    return Path(__file__).resolve().parents[1]


def create_directory(path: Union[str, Path], parents: bool = True) -> Path:
    """
    Create a directory if it doesn't exist.
    
    Args:
        path (Union[str, Path]): Path to the directory
        parents (bool): Whether to create parent directories
    
    Returns:
        Path: Path object of the created directory
    """
    path = Path(path)
    path.mkdir(parents=parents, exist_ok=True)
    return path


def save_object(obj: Any, file_path: Union[str, Path]) -> None:
    """
    Save an object to a file using pickle.
    
    Args:
        obj (Any): Object to save
        file_path (Union[str, Path]): Path where to save the object
    """
    file_path = Path(file_path)
    create_directory(file_path.parent)
    
    with open(file_path, 'wb') as file_obj:
        pickle.dump(obj, file_obj)


def load_object(file_path: Union[str, Path]) -> Any:
    """
    Load an object from a file using pickle.
    
    Args:
        file_path (Union[str, Path]): Path to the file
    
    Returns:
        Any: Loaded object
    """
    file_path = Path(file_path)
    
    with open(file_path, 'rb') as file_obj:
        return pickle.load(file_obj)


def save_json(data: Dict, file_path: Union[str, Path]) -> None:
    """
    Save data to a JSON file.
    
    Args:
        data (Dict): Data to save
        file_path (Union[str, Path]): Path where to save the data
    """
    file_path = Path(file_path)
    create_directory(file_path.parent)
    
    with open(file_path, 'w') as file_obj:
        json.dump(data, file_obj, indent=4)


def load_json(file_path: Union[str, Path]) -> Dict:
    """
    Load data from a JSON file.
    
    Args:
        file_path (Union[str, Path]): Path to the file
    
    Returns:
        Dict: Loaded data
    """
    file_path = Path(file_path)
    
    with open(file_path, 'r') as file_obj:
        return json.load(file_obj)


def save_yaml(data: Dict, file_path: Union[str, Path]) -> None:
    """
    Save data to a YAML file.
    
    Args:
        data (Dict): Data to save
        file_path (Union[str, Path]): Path where to save the data
    """
    file_path = Path(file_path)
    create_directory(file_path.parent)
    
    with open(file_path, 'w') as file_obj:
        yaml.dump(data, file_obj, default_flow_style=False)


def load_yaml(file_path: Union[str, Path]) -> Dict:
    """
    Load data from a YAML file.
    
    Args:
        file_path (Union[str, Path]): Path to the file
    
    Returns:
        Dict: Loaded data
    """
    file_path = Path(file_path)
    
    with open(file_path, 'r') as file_obj:
        return yaml.safe_load(file_obj)


def get_timestamp() -> str:
    """
    Get current timestamp as string.
    
    Returns:
        str: Timestamp in format YYYY-MM-DD_HH-MM-SS
    """
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def get_date() -> str:
    """
    Get current date as string.
    
    Returns:
        str: Date in format YYYY-MM-DD
    """
    return datetime.now().strftime("%Y-%m-%d")


def calculate_missing_percentage(df: pd.DataFrame) -> float:
    """
    Calculate the percentage of missing values in a DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame to analyze
    
    Returns:
        float: Percentage of missing values
    """
    total_cells = df.shape[0] * df.shape[1]
    missing_cells = df.isnull().sum().sum()
    return (missing_cells / total_cells) * 100 if total_cells > 0 else 0.0


def get_dataframe_info(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Get comprehensive information about a DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame to analyze
    
    Returns:
        Dict[str, Any]: Dictionary containing DataFrame information
    """
    info = {
        'shape': df.shape,
        'columns': list(df.columns),
        'dtypes': df.dtypes.to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'missing_percentage': calculate_missing_percentage(df),
        'memory_usage': df.memory_usage(deep=True).sum(),
        'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
        'categorical_columns': df.select_dtypes(include=['object', 'category']).columns.tolist(),
        'datetime_columns': df.select_dtypes(include=['datetime64']).columns.tolist()
    }
    
    return info


def detect_outliers_iqr(df: pd.DataFrame, column: str, factor: float = 1.5) -> Tuple[int, int]:
    """
    Detect outliers using IQR method.
    
    Args:
        df (pd.DataFrame): DataFrame containing the data
        column (str): Column name to analyze
        factor (float): IQR factor for outlier detection
    
    Returns:
        Tuple[int, int]: Number of lower and upper outliers
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - factor * IQR
    upper_bound = Q3 + factor * IQR
    
    lower_outliers = len(df[df[column] < lower_bound])
    upper_outliers = len(df[df[column] > upper_bound])
    
    return lower_outliers, upper_outliers


def detect_outliers_zscore(df: pd.DataFrame, column: str, threshold: float = 3.0) -> int:
    """
    Detect outliers using Z-score method.
    
    Args:
        df (pd.DataFrame): DataFrame containing the data
        column (str): Column name to analyze
        threshold (float): Z-score threshold for outlier detection
    
    Returns:
        int: Number of outliers
    """
    z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
    outliers = len(df[z_scores > threshold])
    
    return outliers


def calculate_correlation_matrix(df: pd.DataFrame, method: str = 'pearson') -> pd.DataFrame:
    """
    Calculate correlation matrix for numeric columns.
    
    Args:
        df (pd.DataFrame): DataFrame to analyze
        method (str): Correlation method ('pearson', 'kendall', 'spearman')
    
    Returns:
        pd.DataFrame: Correlation matrix
    """
    numeric_df = df.select_dtypes(include=[np.number])
    return numeric_df.corr(method=method)


def get_high_correlations(corr_matrix: pd.DataFrame, threshold: float = 0.8) -> List[Tuple[str, str, float]]:
    """
    Get highly correlated feature pairs.
    
    Args:
        corr_matrix (pd.DataFrame): Correlation matrix
        threshold (float): Correlation threshold
    
    Returns:
        List[Tuple[str, str, float]]: List of highly correlated pairs
    """
    high_corr_pairs = []
    
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_value = corr_matrix.iloc[i, j]
            if abs(corr_value) >= threshold:
                high_corr_pairs.append((
                    corr_matrix.columns[i],
                    corr_matrix.columns[j],
                    corr_value
                ))
    
    return high_corr_pairs


def split_dataframe_by_date(df: pd.DataFrame, date_column: str, 
                          train_years: List[int], val_years: List[int]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split DataFrame by date into training and validation sets.
    
    Args:
        df (pd.DataFrame): DataFrame to split
        date_column (str): Name of the date column
        train_years (List[int]): Years for training data
        val_years (List[int]): Years for validation data
    
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Training and validation DataFrames
    """
    df[date_column] = pd.to_datetime(df[date_column])
    
    train_data = df[df[date_column].dt.year.isin(train_years)]
    val_data = df[df[date_column].dt.year.isin(val_years)]
    
    return train_data, val_data


def create_feature_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a summary of features in the DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame to summarize
    
    Returns:
        pd.DataFrame: Feature summary
    """
    summary = []
    
    for column in df.columns:
        col_info = {
            'feature': column,
            'dtype': str(df[column].dtype),
            'count': df[column].count(),
            'missing': df[column].isnull().sum(),
            'missing_pct': (df[column].isnull().sum() / len(df)) * 100,
            'unique': df[column].nunique(),
            'unique_pct': (df[column].nunique() / len(df)) * 100
        }
        
        if df[column].dtype in ['int64', 'float64']:
            col_info.update({
                'mean': df[column].mean(),
                'std': df[column].std(),
                'min': df[column].min(),
                'max': df[column].max(),
                'median': df[column].median()
            })
        
        summary.append(col_info)
    
    return pd.DataFrame(summary)


def validate_data_quality(df: pd.DataFrame, 
                         min_rows: int = 100,
                         max_missing_pct: float = 50.0,
                         min_unique_pct: float = 1.0) -> Dict[str, Any]:
    """
    Validate data quality and return quality metrics.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        min_rows (int): Minimum number of rows required
        max_missing_pct (float): Maximum allowed missing percentage
        min_unique_pct (float): Minimum required unique percentage
    
    Returns:
        Dict[str, Any]: Data quality validation results
    """
    quality_report = {
        'is_valid': True,
        'issues': [],
        'metrics': {}
    }
    
    # Check minimum rows
    if len(df) < min_rows:
        quality_report['is_valid'] = False
        quality_report['issues'].append(f"Too few rows: {len(df)} < {min_rows}")
    
    # Check missing values
    missing_pct = calculate_missing_percentage(df)
    quality_report['metrics']['missing_percentage'] = missing_pct
    
    if missing_pct > max_missing_pct:
        quality_report['is_valid'] = False
        quality_report['issues'].append(f"Too many missing values: {missing_pct:.2f}% > {max_missing_pct}%")
    
    # Check for constant columns
    constant_cols = []
    for col in df.columns:
        unique_pct = (df[col].nunique() / len(df)) * 100
        if unique_pct < min_unique_pct:
            constant_cols.append(col)
    
    if constant_cols:
        quality_report['is_valid'] = False
        quality_report['issues'].append(f"Constant columns found: {constant_cols}")
    
    quality_report['metrics']['constant_columns'] = constant_cols
    quality_report['metrics']['total_rows'] = len(df)
    quality_report['metrics']['total_columns'] = len(df.columns)
    
    return quality_report


def format_bytes(bytes_value: int) -> str:
    """
    Format bytes into human readable format.
    
    Args:
        bytes_value (int): Number of bytes
    
    Returns:
        str: Formatted string
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.2f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.2f} PB"


def get_file_size(file_path: Union[str, Path]) -> str:
    """
    Get file size in human readable format.
    
    Args:
        file_path (Union[str, Path]): Path to the file
    
    Returns:
        str: Formatted file size
    """
    file_path = Path(file_path)
    if file_path.exists():
        return format_bytes(file_path.stat().st_size)
    return "File not found"
