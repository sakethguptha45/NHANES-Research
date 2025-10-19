"""
Model trainer module for LLM vs Traditional ML comparison project.

This module provides comprehensive model training functionality including
hyperparameter tuning, cross-validation, and model persistence.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import make_scorer
import joblib
import warnings

from .baseline_models import TraditionalMLModels
from utils.config import RANDOM_STATE, RESULTS_DIR
from utils.logger import get_logger
from utils.helpers import save_results

logger = get_logger(__name__)

class ModelTrainer:
    """
    Comprehensive model trainer for traditional ML models.
    
    This class provides advanced training capabilities including hyperparameter
    tuning, cross-validation, model persistence, and training history tracking.
    """
    
    def __init__(self, models: Optional[TraditionalMLModels] = None):
        """
        Initialize the ModelTrainer.
        
        Args:
            models: TraditionalMLModels instance. If None, creates new instance.
        """
        self.models = models or TraditionalMLModels()
        self.training_history: Dict[str, List[Dict[str, Any]]] = {}
        self.best_models: Dict[str, Any] = {}
        self.tuning_results: Dict[str, Dict[str, Any]] = {}
        
        logger.info("ModelTrainer initialized")
    
    def train_with_cross_validation(self, X: pd.DataFrame, y: pd.Series, 
                                  cv_folds: int = 5) -> Dict[str, Dict[str, Any]]:
        """
        Train models with cross-validation for robust evaluation.
        
        Args:
            X: Features
            y: Target
            cv_folds: Number of CV folds
            
        Returns:
            Dictionary containing CV results for each model
        """
        logger.info(f"Starting training with {cv_folds}-fold cross-validation...")
        
        # Perform cross-validation
        cv_results = self.models.cross_validate_models(X, y, cv_folds)
        
        # Train final models on full dataset
        trained_models = self.models.train_all_models(X, y)
        
        # Store training history
        for model_name in trained_models.keys():
            if model_name not in self.training_history:
                self.training_history[model_name] = []
            
            self.training_history[model_name].append({
                'method': 'cross_validation',
                'cv_folds': cv_folds,
                'cv_results': cv_results.get(model_name, {}),
                'timestamp': pd.Timestamp.now().isoformat()
            })
        
        logger.info("Cross-validation training completed")
        return cv_results
    
    def hyperparameter_tuning_comprehensive(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Dict[str, Any]]:
        """
        Perform comprehensive hyperparameter tuning for all models.
        
        Args:
            X_train: Training features
            y_train: Training target
            
        Returns:
            Dictionary containing tuning results for each model
        """
        logger.info("Starting comprehensive hyperparameter tuning...")
        
        # Define parameter grids for each model
        param_grids = self._get_parameter_grids()
        
        tuning_results = {}
        
        for model_name, param_grid in param_grids.items():
            if model_name not in self.models.models:
                logger.warning(f"Skipping {model_name} - not available")
                continue
            
            try:
                logger.info(f"Tuning hyperparameters for {model_name}...")
                
                # Use RandomizedSearchCV for efficiency
                random_search = RandomizedSearchCV(
                    self.models.models[model_name],
                    param_grid,
                    n_iter=20,  # Number of parameter settings sampled
                    cv=3,       # 3-fold CV for speed
                    scoring='accuracy',
                    n_jobs=-1,
                    random_state=RANDOM_STATE,
                    verbose=1
                )
                
                # Perform search
                random_search.fit(X_train, y_train)
                
                # Update model with best parameters
                self.models.models[model_name] = random_search.best_estimator_
                
                # Store results
                tuning_results[model_name] = {
                    'best_params': random_search.best_params_,
                    'best_score': random_search.best_score_,
                    'cv_results': random_search.cv_results_
                }
                
                # Store in training history
                if model_name not in self.training_history:
                    self.training_history[model_name] = []
                
                self.training_history[model_name].append({
                    'method': 'hyperparameter_tuning',
                    'best_params': random_search.best_params_,
                    'best_score': random_search.best_score_,
                    'timestamp': pd.Timestamp.now().isoformat()
                })
                
                logger.info(f"✅ {model_name} tuning completed: {random_search.best_score_:.4f}")
                
            except Exception as e:
                logger.error(f"❌ {model_name} tuning failed: {e}")
                continue
        
        self.tuning_results = tuning_results
        logger.info(f"Hyperparameter tuning completed for: {list(tuning_results.keys())}")
        return tuning_results
    
    def _get_parameter_grids(self) -> Dict[str, Dict[str, List]]:
        """
        Get parameter grids for hyperparameter tuning.
        
        Returns:
            Dictionary of parameter grids for each model
        """
        param_grids = {
            'RandomForest': {
                'n_estimators': [50, 100, 200, 300],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', None]
            },
            'LogisticRegression': {
                'C': [0.001, 0.01, 0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga'],
                'max_iter': [1000, 2000, 5000]
            },
            'SVM': {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
                'kernel': ['rbf', 'linear', 'poly']
            }
        }
        
        # Add XGBoost if available
        try:
            from xgboost import XGBClassifier
            param_grids['XGBoost'] = {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0]
            }
        except ImportError:
            pass
        
        return param_grids
    
    def train_ensemble_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> Any:
        """
        Train an ensemble model combining all individual models.
        
        Args:
            X_train: Training features
            y_train: Training target
            
        Returns:
            Trained ensemble model
        """
        try:
            from sklearn.ensemble import VotingClassifier
            
            logger.info("Training ensemble model...")
            
            # Get trained models
            trained_models = self.models.trained_models
            
            if len(trained_models) < 2:
                logger.warning("Need at least 2 models for ensemble")
                return None
            
            # Create voting classifier
            estimators = [(name, model) for name, model in trained_models.items()]
            ensemble = VotingClassifier(
                estimators=estimators,
                voting='soft'  # Use predicted probabilities
            )
            
            # Train ensemble
            ensemble.fit(X_train, y_train)
            
            # Store in models
            self.models.trained_models['Ensemble'] = ensemble
            
            logger.info("✅ Ensemble model training completed")
            return ensemble
            
        except Exception as e:
            logger.error(f"❌ Ensemble model training failed: {e}")
            return None
    
    def save_models(self, output_dir: str) -> None:
        """
        Save trained models to disk.
        
        Args:
            output_dir: Directory to save models
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving models to {output_path}")
        
        # Save individual models
        for model_name, model in self.models.trained_models.items():
            model_file = output_path / f"{model_name.lower()}_model.pkl"
            joblib.dump(model, model_file)
            logger.info(f"Saved {model_name} to {model_file}")
        
        # Save training history
        if self.training_history:
            history_file = output_path / "training_history.json"
            save_results(self.training_history, history_file)
        
        # Save tuning results
        if self.tuning_results:
            tuning_file = output_path / "tuning_results.json"
            save_results(self.tuning_results, tuning_file)
        
        logger.info("Models saved successfully")
    
    def load_models(self, model_dir: str) -> None:
        """
        Load trained models from disk.
        
        Args:
            model_dir: Directory containing saved models
        """
        model_path = Path(model_dir)
        
        if not model_path.exists():
            logger.error(f"Model directory not found: {model_path}")
            return
        
        logger.info(f"Loading models from {model_path}")
        
        # Find all model files
        model_files = list(model_path.glob("*_model.pkl"))
        
        for model_file in model_files:
            try:
                model_name = model_file.stem.replace('_model', '').title()
                model = joblib.load(model_file)
                self.models.trained_models[model_name] = model
                logger.info(f"Loaded {model_name} from {model_file}")
            except Exception as e:
                logger.error(f"Failed to load {model_file}: {e}")
        
        logger.info(f"Loaded {len(self.models.trained_models)} models")
    
    def get_training_summary(self) -> str:
        """
        Get a formatted summary of training history.
        
        Returns:
            Formatted string summary
        """
        if not self.training_history:
            return "No training history available."
        
        summary = []
        summary.append("=" * 80)
        summary.append("MODEL TRAINING SUMMARY")
        summary.append("=" * 80)
        
        for model_name, history in self.training_history.items():
            summary.append(f"\n{model_name}:")
            summary.append("-" * 40)
            
            for i, entry in enumerate(history, 1):
                summary.append(f"  Training {i}:")
                summary.append(f"    Method: {entry['method']}")
                summary.append(f"    Timestamp: {entry['timestamp']}")
                
                if 'best_score' in entry:
                    summary.append(f"    Best Score: {entry['best_score']:.4f}")
                
                if 'best_params' in entry:
                    summary.append(f"    Best Params: {entry['best_params']}")
        
        summary.append("=" * 80)
        
        return "\n".join(summary)


def main():
    """Main function for testing the ModelTrainer."""
    try:
        from data_preparation.data_loader import DataLoader
        from data_preparation.data_splitter import DataSplitter
        
        # Load and split data
        loader = DataLoader()
        data = loader.load_data()
        
        splitter = DataSplitter()
        X_train, X_test, y_train, y_test = splitter.split_data(data)
        
        # Initialize trainer
        trainer = ModelTrainer()
        
        # Train with cross-validation
        cv_results = trainer.train_with_cross_validation(X_train, y_train)
        
        # Hyperparameter tuning
        tuning_results = trainer.hyperparameter_tuning_comprehensive(X_train, y_train)
        
        # Train ensemble
        ensemble = trainer.train_ensemble_model(X_train, y_train)
        
        # Evaluate models
        results = trainer.models.evaluate_all_models(X_test, y_test)
        
        # Print summary
        summary = trainer.models.get_model_summary()
        print(summary)
        
        # Save models and results
        trainer.save_models("results/ml_results")
        trainer.models.save_results("results/ml_results")
        
        print("\n🎉 Model training completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in main: {e}")
        raise


if __name__ == "__main__":
    main()
