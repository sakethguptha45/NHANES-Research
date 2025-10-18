"""
Helper utility functions for LLM vs Traditional ML comparison project.

This module contains common utility functions used across different modules
in the project, including data validation, file operations, and common
mathematical operations.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import json

from .config import FEATURE_COLUMNS, TARGET_COLUMN, ID_COLUMN
from .logger import get_logger

logger = get_logger(__name__)

def validate_dataframe(df: pd.DataFrame, expected_columns: List[str]) -> bool:
    """
    Validate that a DataFrame has the expected columns and data types.
    
    Args:
        df: DataFrame to validate
        expected_columns: List of expected column names
    
    Returns:
        True if validation passes, False otherwise
    """
    try:
        # Check if all expected columns exist
        missing_columns = set(expected_columns) - set(df.columns)
        if missing_columns:
            logger.error(f"Missing columns: {missing_columns}")
            return False
        
        # Check for completely empty columns
        empty_columns = df.columns[df.isnull().all()].tolist()
        if empty_columns:
            logger.warning(f"Empty columns detected: {empty_columns}")
        
        # Check data types
        numeric_columns = FEATURE_COLUMNS + [TARGET_COLUMN]
        for col in numeric_columns:
            if col in df.columns:
                if not pd.api.types.is_numeric_dtype(df[col]):
                    logger.warning(f"Column {col} is not numeric: {df[col].dtype}")
        
        logger.info(f"DataFrame validation passed. Shape: {df.shape}")
        return True
        
    except Exception as e:
        logger.error(f"DataFrame validation failed: {e}")
        return False

def save_results(results: Dict[str, Any], filepath: Path) -> None:
    """
    Save results dictionary to JSON file.
    
    Args:
        results: Dictionary containing results to save
        filepath: Path where to save the results
    """
    try:
        # Ensure directory exists
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert numpy types to Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        # Convert results
        json_results = json.loads(json.dumps(results, default=convert_numpy))
        
        # Save to file
        with open(filepath, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        logger.info(f"Results saved to {filepath}")
        
    except Exception as e:
        logger.error(f"Failed to save results to {filepath}: {e}")

def load_results(filepath: Path) -> Dict[str, Any]:
    """
    Load results from JSON file.
    
    Args:
        filepath: Path to the results file
    
    Returns:
        Dictionary containing loaded results
    """
    try:
        with open(filepath, 'r') as f:
            results = json.load(f)
        
        logger.info(f"Results loaded from {filepath}")
        return results
        
    except Exception as e:
        logger.error(f"Failed to load results from {filepath}: {e}")
        return {}

def calculate_class_balance(y: pd.Series) -> Dict[str, float]:
    """
    Calculate class balance statistics.
    
    Args:
        y: Target variable series
    
    Returns:
        Dictionary with class balance information
    """
    class_counts = y.value_counts()
    total_samples = len(y)
    
    balance_info = {
        'total_samples': total_samples,
        'class_counts': class_counts.to_dict(),
        'class_proportions': (class_counts / total_samples).to_dict(),
        'majority_class': class_counts.index[0],
        'minority_class': class_counts.index[-1],
        'imbalance_ratio': class_counts.iloc[0] / class_counts.iloc[-1]
    }
    
    logger.info(f"Class balance - Majority: {balance_info['majority_class']} "
                f"({balance_info['class_proportions'][balance_info['majority_class']]:.3f}), "
                f"Minority: {balance_info['minority_class']} "
                f"({balance_info['class_proportions'][balance_info['minority_class']]:.3f})")
    
    return balance_info

def format_percentage(value: float, decimals: int = 3) -> str:
    """
    Format a decimal value as a percentage string.
    
    Args:
        value: Decimal value to format
        decimals: Number of decimal places
    
    Returns:
        Formatted percentage string
    """
    return f"{value * 100:.{decimals}f}%"

def create_summary_table(results: Dict[str, Dict[str, float]], 
                        metric: str = 'accuracy') -> pd.DataFrame:
    """
    Create a summary table from results dictionary.
    
    Args:
        results: Dictionary containing model results
        metric: Metric to include in the summary
    
    Returns:
        DataFrame with summary information
    """
    summary_data = []
    
    for model_name, metrics in results.items():
        summary_data.append({
            'Model': model_name,
            'Accuracy': format_percentage(metrics.get('accuracy', 0)),
            'AUC': f"{metrics.get('auc', 0):.3f}" if 'auc' in metrics else 'N/A'
        })
    
    return pd.DataFrame(summary_data)

def check_ollama_connection() -> bool:
    """
    Check if Ollama is running and accessible.
    
    Returns:
        True if Ollama is accessible, False otherwise
    """
    import requests
    from .config import OLLAMA_BASE_URL
    
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/tags", timeout=5)
        if response.status_code == 200:
            logger.info("Ollama connection successful")
            return True
        else:
            logger.error(f"Ollama connection failed: {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"Ollama connection failed: {e}")
        return False

def get_available_ollama_models() -> List[str]:
    """
    Get list of available Ollama models.
    
    Returns:
        List of available model names
    """
    import requests
    from .config import OLLAMA_BASE_URL
    
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get('models', [])
            model_names = [model['name'] for model in models]
            logger.info(f"Available Ollama models: {model_names}")
            return model_names
        else:
            logger.error(f"Failed to get Ollama models: {response.status_code}")
            return []
    except Exception as e:
        logger.error(f"Failed to get Ollama models: {e}")
        return []
