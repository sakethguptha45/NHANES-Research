"""
Data splitter module for LLM vs Traditional ML comparison project.

This module handles data splitting for training and testing, including
stratified splits, LLM sample generation, and cross-validation setup.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any, Optional, List
from sklearn.model_selection import train_test_split, StratifiedKFold
import warnings

from utils.config import (
    TRAIN_DATA_PATH, TEST_DATA_PATH, LLM_SAMPLES_PATH,
    FEATURE_COLUMNS, TARGET_COLUMN, ID_COLUMN,
    TEST_SIZE, LLM_SAMPLE_SIZE, RANDOM_STATE
)
from utils.logger import get_logger
from utils.helpers import calculate_class_balance, save_results

logger = get_logger(__name__)

class DataSplitter:
    """
    Data splitting class for creating train/test splits and LLM samples.
    
    This class provides methods to split data into training and testing sets,
    generate samples for LLM evaluation, and set up cross-validation folds.
    """
    
    def __init__(self, test_size: float = TEST_SIZE, random_state: int = RANDOM_STATE):
        """
        Initialize the DataSplitter.
        
        Args:
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility
        """
        self.test_size = test_size
        self.random_state = random_state
        self.splits: Dict[str, Any] = {}
        
        logger.info(f"DataSplitter initialized with test_size={test_size}, random_state={random_state}")
    
    def split_data(self, data: pd.DataFrame, target_col: str = TARGET_COLUMN) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split data into training and testing sets with stratification.
        
        Args:
            data: DataFrame to split
            target_col: Name of the target column
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        logger.info("Starting data splitting...")
        
        # Prepare features and target
        feature_cols = [col for col in FEATURE_COLUMNS if col in data.columns]
        X = data[feature_cols]
        y = data[target_col]
        
        # Check if we have enough samples for stratification
        min_samples_per_class = y.value_counts().min()
        if min_samples_per_class < 2:
            logger.warning("Not enough samples per class for stratification. Using random split.")
            stratify = None
        else:
            stratify = y
        
        # Perform the split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=stratify
        )
        
        # Log split information
        logger.info(f"Data split completed:")
        logger.info(f"  Training set: {X_train.shape[0]} samples")
        logger.info(f"  Test set: {X_test.shape[0]} samples")
        logger.info(f"  Features: {X_train.shape[1]}")
        
        # Log class distribution
        train_balance = calculate_class_balance(y_train)
        test_balance = calculate_class_balance(y_test)
        
        logger.info(f"Training set class balance: {train_balance['class_proportions']}")
        logger.info(f"Test set class balance: {test_balance['class_proportions']}")
        
        # Store split information
        self.splits = {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'feature_columns': feature_cols,
            'train_balance': train_balance,
            'test_balance': test_balance
        }
        
        return X_train, X_test, y_train, y_test
    
    def create_llm_samples(self, data: pd.DataFrame, sample_size: int = LLM_SAMPLE_SIZE) -> pd.DataFrame:
        """
        Create a smaller sample for LLM evaluation.
        
        Args:
            data: Full dataset
            sample_size: Number of samples to create
            
        Returns:
            Sample DataFrame for LLM evaluation
        """
        logger.info(f"Creating LLM sample with {sample_size} samples...")
        
        # Ensure we don't exceed the available data
        actual_sample_size = min(sample_size, len(data))
        
        # Stratified sampling to maintain class balance
        if TARGET_COLUMN in data.columns:
            # Calculate samples per class
            class_counts = data[TARGET_COLUMN].value_counts()
            samples_per_class = {}
            
            for class_val, count in class_counts.items():
                # Proportional sampling
                samples_per_class[class_val] = int((count / len(data)) * actual_sample_size)
            
            # Ensure we have at least 1 sample per class
            for class_val in samples_per_class:
                if samples_per_class[class_val] == 0:
                    samples_per_class[class_val] = 1
            
            # Adjust total if needed
            total_samples = sum(samples_per_class.values())
            if total_samples > actual_sample_size:
                # Reduce largest class
                max_class = max(samples_per_class, key=samples_per_class.get)
                samples_per_class[max_class] -= (total_samples - actual_sample_size)
            
            # Sample from each class
            llm_samples = []
            for class_val, n_samples in samples_per_class.items():
                class_data = data[data[TARGET_COLUMN] == class_val]
                if len(class_data) >= n_samples:
                    sampled = class_data.sample(n=n_samples, random_state=self.random_state)
                    llm_samples.append(sampled)
                else:
                    llm_samples.append(class_data)
            
            llm_sample_data = pd.concat(llm_samples, ignore_index=True)
            
        else:
            # Random sampling if no target column
            llm_sample_data = data.sample(n=actual_sample_size, random_state=self.random_state)
        
        # Shuffle the final sample
        llm_sample_data = llm_sample_data.sample(frac=1, random_state=self.random_state).reset_index(drop=True)
        
        # Log sample information
        logger.info(f"LLM sample created: {len(llm_sample_data)} samples")
        if TARGET_COLUMN in llm_sample_data.columns:
            sample_balance = calculate_class_balance(llm_sample_data[TARGET_COLUMN])
            logger.info(f"LLM sample class balance: {sample_balance['class_proportions']}")
        
        return llm_sample_data
    
    def create_cv_folds(self, data: pd.DataFrame, n_folds: int = 5, target_col: str = TARGET_COLUMN) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Create cross-validation folds for model evaluation.
        
        Args:
            data: DataFrame to create folds from
            n_folds: Number of CV folds
            target_col: Name of the target column
            
        Returns:
            List of (train_indices, test_indices) tuples
        """
        logger.info(f"Creating {n_folds}-fold cross-validation...")
        
        # Prepare target for stratification
        y = data[target_col]
        
        # Create stratified K-fold
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=self.random_state)
        
        # Generate fold indices
        folds = []
        for fold, (train_idx, test_idx) in enumerate(skf.split(data, y)):
            folds.append((train_idx, test_idx))
            logger.info(f"Fold {fold + 1}: Train={len(train_idx)}, Test={len(test_idx)}")
        
        logger.info(f"Cross-validation folds created: {len(folds)} folds")
        return folds
    
    def save_splits(self, output_dir: Optional[Path] = None) -> None:
        """
        Save the data splits to files.
        
        Args:
            output_dir: Directory to save files. If None, uses default paths.
        """
        if not self.splits:
            logger.error("No splits available. Call split_data() first.")
            return
        
        logger.info("Saving data splits...")
        
        try:
            # Use default paths if no output directory specified
            if output_dir is None:
                train_path = TRAIN_DATA_PATH
                test_path = TEST_DATA_PATH
            else:
                train_path = output_dir / "train_data.csv"
                test_path = output_dir / "test_data.csv"
            
            # Create training data (features + target)
            train_data = self.splits['X_train'].copy()
            train_data[TARGET_COLUMN] = self.splits['y_train']
            
            # Create test data (features + target)
            test_data = self.splits['X_test'].copy()
            test_data[TARGET_COLUMN] = self.splits['y_test']
            
            # Save files
            train_data.to_csv(train_path, index=False)
            test_data.to_csv(test_path, index=False)
            
            logger.info(f"Training data saved to: {train_path}")
            logger.info(f"Test data saved to: {test_path}")
            
            # Save split metadata
            split_info = {
                'train_samples': len(train_data),
                'test_samples': len(test_data),
                'features': self.splits['feature_columns'],
                'train_balance': self.splits['train_balance'],
                'test_balance': self.splits['test_balance'],
                'test_size': self.test_size,
                'random_state': self.random_state
            }
            
            metadata_path = (output_dir or TRAIN_DATA_PATH.parent) / "split_metadata.json"
            save_results(split_info, metadata_path)
            
        except Exception as e:
            logger.error(f"Failed to save splits: {e}")
            raise
    
    def save_llm_samples(self, llm_data: pd.DataFrame, output_path: Optional[Path] = None) -> None:
        """
        Save LLM sample data to file.
        
        Args:
            llm_data: LLM sample DataFrame
            output_path: Path to save file. If None, uses default path.
        """
        if output_path is None:
            output_path = LLM_SAMPLES_PATH
        
        try:
            # Ensure directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save data
            llm_data.to_csv(output_path, index=False)
            logger.info(f"LLM sample data saved to: {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to save LLM samples: {e}")
            raise
    
    def get_split_summary(self) -> str:
        """
        Get a formatted summary of the data splits.
        
        Returns:
            Formatted string summary
        """
        if not self.splits:
            return "No splits available. Call split_data() first."
        
        summary = []
        summary.append("=" * 60)
        summary.append("DATA SPLIT SUMMARY")
        summary.append("=" * 60)
        
        # Basic information
        summary.append(f"Test Size: {self.test_size}")
        summary.append(f"Random State: {self.random_state}")
        summary.append("")
        
        # Training set info
        train_samples = len(self.splits['X_train'])
        summary.append(f"Training Set:")
        summary.append(f"  Samples: {train_samples}")
        summary.append(f"  Features: {len(self.splits['feature_columns'])}")
        summary.append(f"  Class Balance: {self.splits['train_balance']['class_proportions']}")
        summary.append("")
        
        # Test set info
        test_samples = len(self.splits['X_test'])
        summary.append(f"Test Set:")
        summary.append(f"  Samples: {test_samples}")
        summary.append(f"  Features: {len(self.splits['feature_columns'])}")
        summary.append(f"  Class Balance: {self.splits['test_balance']['class_proportions']}")
        summary.append("")
        
        # Feature list
        summary.append("Features:")
        for i, feature in enumerate(self.splits['feature_columns'], 1):
            summary.append(f"  {i:2d}. {feature}")
        
        summary.append("=" * 60)
        
        return "\n".join(summary)


def main():
    """Main function for testing the DataSplitter."""
    try:
        from .data_loader import DataLoader
        
        # Load data
        loader = DataLoader()
        data = loader.load_data()
        
        # Initialize splitter
        splitter = DataSplitter()
        
        # Split data
        X_train, X_test, y_train, y_test = splitter.split_data(data)
        
        # Print summary
        summary = splitter.get_split_summary()
        print(summary)
        
        # Create LLM samples
        llm_samples = splitter.create_llm_samples(data)
        print(f"\nLLM samples created: {len(llm_samples)}")
        
        # Save splits
        splitter.save_splits()
        splitter.save_llm_samples(llm_samples)
        
        print("\nData splitting completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in main: {e}")
        raise


if __name__ == "__main__":
    main()