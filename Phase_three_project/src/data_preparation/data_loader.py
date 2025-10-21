"""
Data preparation module for Phase 3: Advanced LLM Strategies

This module handles data loading, preprocessing, and preparation
for different LLM strategies (few-shot, fine-tuning, RAG).
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any, List, Optional
from sklearn.model_selection import train_test_split, StratifiedKFold
import warnings

from utils.config import (
    TOTAL_DATA_PATH, FEATURE_COLUMNS, TARGET_COLUMN, ID_COLUMN,
    TEST_SIZE, RANDOM_STATE, FEW_SHOT_SAMPLE_SIZE
)
from utils.helpers import setup_logger, create_few_shot_examples, create_rag_knowledge_base

logger = setup_logger(__name__)

class Phase3DataLoader:
    """
    Enhanced data loader for Phase 3 with support for different LLM strategies.
    """
    
    def __init__(self, data_path: Path = TOTAL_DATA_PATH):
        """
        Initialize the Phase 3 data loader.
        
        Args:
            data_path: Path to the total_data.csv file
        """
        self.data_path = data_path
        self.df: Optional[pd.DataFrame] = None
        logger.info(f"Phase3DataLoader initialized with path: {self.data_path}")
    
    def load_data(self) -> Optional[pd.DataFrame]:
        """
        Load the total_data.csv file.
        
        Returns:
            Loaded DataFrame or None if loading fails
        """
        if not self.data_path.exists():
            logger.error(f"Data file not found: {self.data_path}")
            return None
        
        try:
            logger.info(f"Loading data from {self.data_path}")
            self.df = pd.read_csv(self.data_path)
            logger.info(f"Data loaded successfully. Shape: {self.df.shape}")
            return self.df
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return None
    
    def clean_data(self) -> pd.DataFrame:
        """
        Clean and preprocess the data for Phase 3.
        
        Returns:
            Cleaned DataFrame
        """
        if self.df is None:
            logger.error("No data loaded. Call load_data() first.")
            return None
        
        logger.info("Starting data cleaning for Phase 3...")
        
        # Make a copy to avoid modifying original
        df_clean = self.df.copy()
        
        # Remove ID column if present in features
        if ID_COLUMN in df_clean.columns:
            df_clean = df_clean.drop(columns=[ID_COLUMN])
        
        # Handle missing values
        df_clean = df_clean.dropna()
        
        # Ensure target column is binary
        df_clean[TARGET_COLUMN] = df_clean[TARGET_COLUMN].astype(int)
        
        logger.info(f"Data cleaning completed. Final shape: {df_clean.shape}")
        logger.info(f"Target distribution: {df_clean[TARGET_COLUMN].value_counts().to_dict()}")
        
        return df_clean
    
    def split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split data into training and testing sets.
        
        Args:
            df: Cleaned DataFrame
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        logger.info("Splitting data for Phase 3...")
        
        # Prepare features and target
        X = df[FEATURE_COLUMNS]
        y = df[TARGET_COLUMN]
        
        # Stratified split to maintain class balance
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )
        
        logger.info(f"Data split completed:")
        logger.info(f"  Training set: {len(X_train)} samples")
        logger.info(f"  Test set: {len(X_test)} samples")
        logger.info(f"  Training class balance: {y_train.value_counts().to_dict()}")
        logger.info(f"  Test class balance: {y_test.value_counts().to_dict()}")
        
        return X_train, X_test, y_train, y_test
    
    def prepare_few_shot_data(self, df: pd.DataFrame) -> Tuple[List[Dict[str, Any]], pd.DataFrame]:
        """
        Prepare data for few-shot learning.
        
        Args:
            df: Cleaned DataFrame
            
        Returns:
            Few-shot examples and test data
        """
        logger.info("Preparing data for few-shot learning...")
        
        # Create few-shot examples from training data
        train_df, test_df, _, _ = self.split_data(df)
        
        # Add target column back to train_df for few-shot examples
        train_df_with_target = train_df.copy()
        train_df_with_target[TARGET_COLUMN] = df[TARGET_COLUMN].loc[train_df.index]
        few_shot_examples = create_few_shot_examples(train_df_with_target)
        
        # Sample test data for evaluation
        if len(test_df) > FEW_SHOT_SAMPLE_SIZE:
            test_sample = test_df.sample(FEW_SHOT_SAMPLE_SIZE, random_state=RANDOM_STATE)
        else:
            test_sample = test_df
        
        # Add target column back to test sample
        test_sample_with_target = test_sample.copy()
        test_sample_with_target[TARGET_COLUMN] = df[TARGET_COLUMN].loc[test_sample.index]
        
        logger.info(f"Few-shot data prepared: {len(few_shot_examples)} examples, {len(test_sample_with_target)} test samples")
        
        return few_shot_examples, test_sample_with_target
    
    def prepare_fine_tuning_data(self, df: pd.DataFrame) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Prepare data for fine-tuning.
        
        Args:
            df: Cleaned DataFrame
            
        Returns:
            Training and validation data for fine-tuning
        """
        logger.info("Preparing data for fine-tuning...")
        
        # Split data
        X_train, X_test, y_train, y_test = self.split_data(df)
        
        # Convert to training format
        train_data = []
        for idx, row in X_train.iterrows():
            train_data.append({
                'features': row.to_dict(),
                'label': int(y_train.loc[idx]),
                'text': f"Patient data: {row.to_dict()} → Health Status: {int(y_train.loc[idx])}"
            })
        
        val_data = []
        for idx, row in X_test.iterrows():
            val_data.append({
                'features': row.to_dict(),
                'label': int(y_test.loc[idx]),
                'text': f"Patient data: {row.to_dict()} → Health Status: {int(y_test.loc[idx])}"
            })
        
        logger.info(f"Fine-tuning data prepared: {len(train_data)} training, {len(val_data)} validation samples")
        
        return train_data, val_data
    
    def prepare_rag_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prepare data for RAG (Retrieval-Augmented Generation).
        
        Args:
            df: Cleaned DataFrame
            
        Returns:
            Knowledge base and test data
        """
        logger.info("Preparing data for RAG...")
        
        # Split data
        X_train, X_test, y_train, y_test = self.split_data(df)
        
        # Create knowledge base from training data
        train_df = X_train.copy()
        train_df[TARGET_COLUMN] = y_train
        knowledge_base = create_rag_knowledge_base(train_df)
        
        # Prepare test data
        test_df = X_test.copy()
        test_df[TARGET_COLUMN] = y_test
        
        # Sample test data for evaluation
        if len(test_df) > FEW_SHOT_SAMPLE_SIZE:
            test_sample = test_df.sample(FEW_SHOT_SAMPLE_SIZE, random_state=RANDOM_STATE)
        else:
            test_sample = test_df
        
        logger.info(f"RAG data prepared: {len(knowledge_base)} knowledge base cases, {len(test_sample)} test samples")
        
        return knowledge_base, test_sample
    
    def prepare_prompt_engineering_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare data for prompt engineering experiments.
        
        Args:
            df: Cleaned DataFrame
            
        Returns:
            Test data for prompt engineering
        """
        logger.info("Preparing data for prompt engineering...")
        
        # Split data
        _, X_test, _, y_test = self.split_data(df)
        
        # Prepare test data
        test_df = X_test.copy()
        test_df[TARGET_COLUMN] = y_test
        
        # Sample test data for evaluation
        if len(test_df) > FEW_SHOT_SAMPLE_SIZE:
            test_sample = test_df.sample(FEW_SHOT_SAMPLE_SIZE, random_state=RANDOM_STATE)
        else:
            test_sample = test_df
        
        logger.info(f"Prompt engineering data prepared: {len(test_sample)} test samples")
        
        return test_sample
    
    def get_data_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics of the loaded data.
        
        Returns:
            Dictionary with data summary
        """
        if self.df is None:
            return {}
        
        summary = {
            'total_samples': len(self.df),
            'features': len(FEATURE_COLUMNS),
            'target_distribution': self.df[TARGET_COLUMN].value_counts().to_dict(),
            'missing_values': self.df.isnull().sum().to_dict(),
            'data_types': self.df.dtypes.to_dict(),
            'memory_usage': self.df.memory_usage(deep=True).sum()
        }
        
        return summary
