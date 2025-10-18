"""
Data loader module for LLM vs Traditional ML comparison project.

This module handles loading and basic preprocessing of the NHANES health data,
including data validation, type checking, and initial data exploration.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import warnings

from utils.config import TOTAL_DATA_PATH, FEATURE_COLUMNS, TARGET_COLUMN, ID_COLUMN
from utils.logger import get_logger
from utils.helpers import validate_dataframe, calculate_class_balance

logger = get_logger(__name__)

class DataLoader:
    """
    Data loader class for handling NHANES health data.
    
    This class provides methods to load, validate, and perform initial
    preprocessing on the total_data.csv file.
    """
    
    def __init__(self, data_path: Optional[Path] = None):
        """
        Initialize the DataLoader.
        
        Args:
            data_path: Path to the data file. If None, uses default from config.
        """
        self.data_path = data_path or TOTAL_DATA_PATH
        self.data: Optional[pd.DataFrame] = None
        self.validation_results: Dict[str, Any] = {}
        
        logger.info(f"DataLoader initialized with path: {self.data_path}")
    
    def load_data(self) -> pd.DataFrame:
        """
        Load the total_data.csv file.
        
        Returns:
            Loaded DataFrame
            
        Raises:
            FileNotFoundError: If the data file doesn't exist
            ValueError: If the data file is empty or corrupted
        """
        try:
            # Check if file exists
            if not self.data_path.exists():
                raise FileNotFoundError(f"Data file not found: {self.data_path}")
            
            # Load CSV file
            logger.info(f"Loading data from {self.data_path}")
            self.data = pd.read_csv(self.data_path)
            
            # Basic validation
            if self.data.empty:
                raise ValueError("Data file is empty")
            
            logger.info(f"Data loaded successfully. Shape: {self.data.shape}")
            logger.info(f"Columns: {list(self.data.columns)}")
            
            return self.data
            
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            raise
    
    def get_data_info(self) -> Dict[str, Any]:
        """
        Get comprehensive information about the loaded data.
        
        Returns:
            Dictionary containing data information
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        info = {
            'shape': self.data.shape,
            'columns': list(self.data.columns),
            'dtypes': self.data.dtypes.to_dict(),
            'memory_usage': self.data.memory_usage(deep=True).sum(),
            'missing_values': self.data.isnull().sum().to_dict(),
            'duplicate_rows': self.data.duplicated().sum(),
            'target_distribution': self.data[TARGET_COLUMN].value_counts().to_dict() if TARGET_COLUMN in self.data.columns else None
        }
        
        # Add class balance information
        if TARGET_COLUMN in self.data.columns:
            info['class_balance'] = calculate_class_balance(self.data[TARGET_COLUMN])
        
        logger.info(f"Data info collected: {info['shape'][0]} rows, {info['shape'][1]} columns")
        return info
    
    def validate_data(self) -> bool:
        """
        Validate the loaded data for quality and consistency.
        
        Returns:
            True if validation passes, False otherwise
        """
        if self.data is None:
            logger.error("No data loaded. Call load_data() first.")
            return False
        
        logger.info("Starting data validation...")
        
        # Initialize validation results
        self.validation_results = {
            'passed': True,
            'errors': [],
            'warnings': [],
            'checks_performed': []
        }
        
        # Check 1: Required columns exist
        required_columns = FEATURE_COLUMNS + [TARGET_COLUMN, ID_COLUMN]
        missing_columns = set(required_columns) - set(self.data.columns)
        
        if missing_columns:
            error_msg = f"Missing required columns: {missing_columns}"
            self.validation_results['errors'].append(error_msg)
            self.validation_results['passed'] = False
            logger.error(error_msg)
        else:
            self.validation_results['checks_performed'].append("Required columns check: PASSED")
        
        # Check 2: Data types
        numeric_columns = FEATURE_COLUMNS + [TARGET_COLUMN]
        for col in numeric_columns:
            if col in self.data.columns:
                if not pd.api.types.is_numeric_dtype(self.data[col]):
                    warning_msg = f"Column {col} is not numeric: {self.data[col].dtype}"
                    self.validation_results['warnings'].append(warning_msg)
                    logger.warning(warning_msg)
        
        self.validation_results['checks_performed'].append("Data types check: COMPLETED")
        
        # Check 3: Missing values
        missing_values = self.data.isnull().sum()
        high_missing_cols = missing_values[missing_values > len(self.data) * 0.1].index.tolist()
        
        if high_missing_cols:
            warning_msg = f"Columns with >10% missing values: {high_missing_cols}"
            self.validation_results['warnings'].append(warning_msg)
            logger.warning(warning_msg)
        
        self.validation_results['checks_performed'].append("Missing values check: COMPLETED")
        
        # Check 4: Target variable distribution
        if TARGET_COLUMN in self.data.columns:
            target_dist = self.data[TARGET_COLUMN].value_counts()
            if len(target_dist) < 2:
                error_msg = f"Target variable has only one class: {target_dist.index[0]}"
                self.validation_results['errors'].append(error_msg)
                self.validation_results['passed'] = False
                logger.error(error_msg)
            else:
                self.validation_results['checks_performed'].append("Target distribution check: PASSED")
        
        # Check 5: Duplicate rows
        duplicate_count = self.data.duplicated().sum()
        if duplicate_count > 0:
            warning_msg = f"Found {duplicate_count} duplicate rows"
            self.validation_results['warnings'].append(warning_msg)
            logger.warning(warning_msg)
        
        self.validation_results['checks_performed'].append("Duplicate rows check: COMPLETED")
        
        # Log final validation result
        if self.validation_results['passed']:
            logger.info("Data validation PASSED")
        else:
            logger.error("Data validation FAILED")
        
        return self.validation_results['passed']
    
    def get_validation_report(self) -> str:
        """
        Get a formatted validation report.
        
        Returns:
            Formatted string report
        """
        if not self.validation_results:
            return "No validation performed yet. Call validate_data() first."
        
        report = []
        report.append("=" * 60)
        report.append("DATA VALIDATION REPORT")
        report.append("=" * 60)
        
        # Overall status
        status = "PASSED" if self.validation_results['passed'] else "FAILED"
        report.append(f"Overall Status: {status}")
        report.append("")
        
        # Checks performed
        report.append("Checks Performed:")
        for check in self.validation_results['checks_performed']:
            report.append(f"  ✓ {check}")
        report.append("")
        
        # Errors
        if self.validation_results['errors']:
            report.append("Errors:")
            for error in self.validation_results['errors']:
                report.append(f"  ✗ {error}")
            report.append("")
        
        # Warnings
        if self.validation_results['warnings']:
            report.append("Warnings:")
            for warning in self.validation_results['warnings']:
                report.append(f"  ⚠ {warning}")
            report.append("")
        
        report.append("=" * 60)
        
        return "\n".join(report)
    
    def clean_data(self) -> pd.DataFrame:
        """
        Perform basic data cleaning operations.
        
        Returns:
            Cleaned DataFrame
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        logger.info("Starting data cleaning...")
        
        # Create a copy to avoid modifying original data
        cleaned_data = self.data.copy()
        
        # Remove duplicate rows
        initial_rows = len(cleaned_data)
        cleaned_data = cleaned_data.drop_duplicates()
        removed_duplicates = initial_rows - len(cleaned_data)
        
        if removed_duplicates > 0:
            logger.info(f"Removed {removed_duplicates} duplicate rows")
        
        # Handle missing values in target variable
        if TARGET_COLUMN in cleaned_data.columns:
            initial_rows = len(cleaned_data)
            cleaned_data = cleaned_data.dropna(subset=[TARGET_COLUMN])
            removed_target_nulls = initial_rows - len(cleaned_data)
            
            if removed_target_nulls > 0:
                logger.info(f"Removed {removed_target_nulls} rows with missing target values")
        
        # Update the stored data
        self.data = cleaned_data
        
        logger.info(f"Data cleaning completed. Final shape: {self.data.shape}")
        return self.data
    
    def get_feature_summary(self) -> pd.DataFrame:
        """
        Get summary statistics for all features.
        
        Returns:
            DataFrame with feature summaries
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        # Select only feature columns
        feature_data = self.data[FEATURE_COLUMNS] if all(col in self.data.columns for col in FEATURE_COLUMNS) else self.data.select_dtypes(include=[np.number])
        
        summary = feature_data.describe()
        
        # Add additional statistics
        summary.loc['missing_count'] = feature_data.isnull().sum()
        summary.loc['missing_pct'] = (feature_data.isnull().sum() / len(feature_data)) * 100
        summary.loc['unique_count'] = feature_data.nunique()
        
        logger.info("Feature summary generated")
        return summary
    
    def save_processed_data(self, output_path: Path) -> None:
        """
        Save the processed data to a file.
        
        Args:
            output_path: Path where to save the processed data
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        try:
            # Ensure directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save data
            self.data.to_csv(output_path, index=False)
            logger.info(f"Processed data saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to save processed data: {e}")
            raise


def main():
    """Main function for testing the DataLoader."""
    try:
        # Initialize data loader
        loader = DataLoader()
        
        # Load data
        data = loader.load_data()
        
        # Get data info
        info = loader.get_data_info()
        print("Data Info:")
        print(f"Shape: {info['shape']}")
        print(f"Memory usage: {info['memory_usage'] / 1024 / 1024:.2f} MB")
        
        # Validate data
        validation_passed = loader.validate_data()
        print(f"\nValidation passed: {validation_passed}")
        
        # Print validation report
        report = loader.get_validation_report()
        print(f"\n{report}")
        
        # Clean data
        cleaned_data = loader.clean_data()
        print(f"\nCleaned data shape: {cleaned_data.shape}")
        
        # Get feature summary
        summary = loader.get_feature_summary()
        print(f"\nFeature Summary:")
        print(summary.head())
        
    except Exception as e:
        logger.error(f"Error in main: {e}")
        raise


if __name__ == "__main__":
    main()
