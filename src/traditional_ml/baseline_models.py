"""
Baseline ML models module for LLM vs Traditional ML comparison project.

This module implements four traditional machine learning models:
- Random Forest Classifier
- Logistic Regression
- Support Vector Machine (SVM)
- XGBoost Classifier

All models are configured with appropriate parameters for handling
class imbalance and achieving optimal performance.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, GridSearchCV
import warnings

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    warnings.warn("XGBoost not available. Install with: pip install xgboost")

from utils.config import ML_MODEL_PARAMS, RANDOM_STATE, EVALUATION_METRICS
from utils.logger import get_logger
from utils.helpers import save_results, calculate_class_balance

logger = get_logger(__name__)

class TraditionalMLModels:
    """
    Traditional ML models implementation for health prediction.
    
    This class provides methods to train, evaluate, and compare four different
    machine learning models using the processed NHANES data.
    """
    
    def __init__(self):
        """Initialize the TraditionalMLModels class."""
        self.models: Dict[str, Any] = {}
        self.trained_models: Dict[str, Any] = {}
        self.model_results: Dict[str, Dict[str, Any]] = {}
        self.feature_importance: Dict[str, pd.DataFrame] = {}
        
        # Initialize models
        self._initialize_models()
        
        logger.info("TraditionalMLModels initialized with 4 models")
    
    def _initialize_models(self) -> None:
        """Initialize all ML models with their parameters."""
        logger.info("Initializing ML models...")
        
        # Random Forest
        self.models['RandomForest'] = RandomForestClassifier(
            **ML_MODEL_PARAMS['RandomForest']
        )
        
        # Logistic Regression
        self.models['LogisticRegression'] = LogisticRegression(
            **ML_MODEL_PARAMS['LogisticRegression']
        )
        
        # Support Vector Machine
        self.models['SVM'] = SVC(
            **ML_MODEL_PARAMS['SVM']
        )
        
        # XGBoost (if available)
        if XGBOOST_AVAILABLE:
            self.models['XGBoost'] = XGBClassifier(
                **ML_MODEL_PARAMS['XGBoost']
            )
        else:
            logger.warning("XGBoost not available - skipping")
        
        logger.info(f"Models initialized: {list(self.models.keys())}")
    
    def train_all_models(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Any]:
        """
        Train all ML models on the training data.
        
        Args:
            X_train: Training features
            y_train: Training target
            
        Returns:
            Dictionary of trained models
        """
        logger.info("Starting model training...")
        
        # Log training data info
        logger.info(f"Training data shape: {X_train.shape}")
        logger.info(f"Training target distribution: {y_train.value_counts().to_dict()}")
        
        # Train each model
        for model_name, model in self.models.items():
            try:
                logger.info(f"Training {model_name}...")
                
                # Train the model
                model.fit(X_train, y_train)
                self.trained_models[model_name] = model
                
                logger.info(f"✅ {model_name} training completed")
                
            except Exception as e:
                logger.error(f"❌ {model_name} training failed: {e}")
                continue
        
        logger.info(f"Model training completed. Trained models: {list(self.trained_models.keys())}")
        return self.trained_models
    
    def evaluate_all_models(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Dict[str, Any]]:
        """
        Evaluate all trained models on test data.
        
        Args:
            X_test: Test features
            y_test: Test target
            
        Returns:
            Dictionary containing evaluation results for each model
        """
        logger.info("Starting model evaluation...")
        
        if not self.trained_models:
            raise ValueError("No trained models found. Call train_all_models() first.")
        
        # Log test data info
        logger.info(f"Test data shape: {X_test.shape}")
        logger.info(f"Test target distribution: {y_test.value_counts().to_dict()}")
        
        # Evaluate each model
        for model_name, model in self.trained_models.items():
            try:
                logger.info(f"Evaluating {model_name}...")
                
                # Get predictions
                y_pred = model.predict(X_test)
                y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
                
                # Calculate metrics
                results = self._calculate_metrics(y_test, y_pred, y_proba)
                
                # Store results
                self.model_results[model_name] = results
                
                logger.info(f"✅ {model_name} evaluation completed")
                logger.info(f"   Accuracy: {results['accuracy']:.4f}")
                logger.info(f"   AUC: {results['auc']:.4f}")
                
            except Exception as e:
                logger.error(f"❌ {model_name} evaluation failed: {e}")
                continue
        
        logger.info(f"Model evaluation completed. Results for: {list(self.model_results.keys())}")
        return self.model_results
    
    def _calculate_metrics(self, y_true: pd.Series, y_pred: np.ndarray, y_proba: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Calculate comprehensive evaluation metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities (if available)
            
        Returns:
            Dictionary of calculated metrics
        """
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score,
            roc_auc_score, confusion_matrix, classification_report
        )
        
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, average='weighted')
        metrics['recall'] = recall_score(y_true, y_pred, average='weighted')
        metrics['f1_score'] = f1_score(y_true, y_pred, average='weighted')
        
        # AUC (if probabilities available)
        if y_proba is not None:
            try:
                metrics['auc'] = roc_auc_score(y_true, y_proba)
            except ValueError:
                metrics['auc'] = 0.0
        else:
            metrics['auc'] = None
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        
        # Classification report
        metrics['classification_report'] = classification_report(y_true, y_pred, output_dict=True)
        
        # Additional metrics
        metrics['true_negatives'] = int(cm[0, 0])
        metrics['false_positives'] = int(cm[0, 1])
        metrics['false_negatives'] = int(cm[1, 0])
        metrics['true_positives'] = int(cm[1, 1])
        
        return metrics
    
    def cross_validate_models(self, X: pd.DataFrame, y: pd.Series, cv_folds: int = 5) -> Dict[str, Dict[str, Any]]:
        """
        Perform cross-validation for all models.
        
        Args:
            X: Features
            y: Target
            cv_folds: Number of CV folds
            
        Returns:
            Dictionary containing CV results for each model
        """
        logger.info(f"Starting {cv_folds}-fold cross-validation...")
        
        cv_results = {}
        
        for model_name, model in self.models.items():
            try:
                logger.info(f"Cross-validating {model_name}...")
                
                # Perform cross-validation
                cv_scores = cross_val_score(
                    model, X, y, 
                    cv=cv_folds, 
                    scoring='accuracy',
                    n_jobs=-1
                )
                
                # Calculate CV metrics
                cv_results[model_name] = {
                    'cv_scores': cv_scores.tolist(),
                    'mean_cv_score': cv_scores.mean(),
                    'std_cv_score': cv_scores.std(),
                    'cv_folds': cv_folds
                }
                
                logger.info(f"✅ {model_name} CV completed: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
                
            except Exception as e:
                logger.error(f"❌ {model_name} CV failed: {e}")
                continue
        
        logger.info(f"Cross-validation completed for: {list(cv_results.keys())}")
        return cv_results
    
    def get_feature_importance(self, feature_names: List[str]) -> Dict[str, pd.DataFrame]:
        """
        Get feature importance for models that support it.
        
        Args:
            feature_names: List of feature names
            
        Returns:
            Dictionary containing feature importance for each model
        """
        logger.info("Extracting feature importance...")
        
        for model_name, model in self.trained_models.items():
            try:
                if hasattr(model, 'feature_importances_'):
                    # Tree-based models (RandomForest, XGBoost)
                    importance = model.feature_importances_
                    
                    # Create DataFrame
                    importance_df = pd.DataFrame({
                        'feature': feature_names,
                        'importance': importance
                    }).sort_values('importance', ascending=False)
                    
                    self.feature_importance[model_name] = importance_df
                    
                    logger.info(f"✅ {model_name} feature importance extracted")
                    
                elif hasattr(model, 'coef_'):
                    # Linear models (LogisticRegression)
                    coef = model.coef_[0]
                    
                    # Create DataFrame
                    importance_df = pd.DataFrame({
                        'feature': feature_names,
                        'coefficient': coef,
                        'abs_coefficient': np.abs(coef)
                    }).sort_values('abs_coefficient', ascending=False)
                    
                    self.feature_importance[model_name] = importance_df
                    
                    logger.info(f"✅ {model_name} coefficients extracted")
                    
                else:
                    logger.info(f"⚠️ {model_name} does not support feature importance")
                    
            except Exception as e:
                logger.error(f"❌ {model_name} feature importance extraction failed: {e}")
                continue
        
        logger.info(f"Feature importance extracted for: {list(self.feature_importance.keys())}")
        return self.feature_importance
    
    def hyperparameter_tuning(self, X_train: pd.DataFrame, y_train: pd.Series, 
                            model_name: str, param_grid: Dict[str, List]) -> Dict[str, Any]:
        """
        Perform hyperparameter tuning for a specific model.
        
        Args:
            X_train: Training features
            y_train: Training target
            model_name: Name of model to tune
            param_grid: Parameter grid for GridSearchCV
            
        Returns:
            Best parameters and score
        """
        logger.info(f"Starting hyperparameter tuning for {model_name}...")
        
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        # Create base model
        base_model = self.models[model_name]
        
        # Perform grid search
        grid_search = GridSearchCV(
            base_model, 
            param_grid, 
            cv=3,  # Use fewer folds for speed
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        # Update model with best parameters
        self.models[model_name] = grid_search.best_estimator_
        
        results = {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'cv_results': grid_search.cv_results_
        }
        
        logger.info(f"✅ {model_name} hyperparameter tuning completed")
        logger.info(f"   Best score: {grid_search.best_score_:.4f}")
        logger.info(f"   Best params: {grid_search.best_params_}")
        
        return results
    
    def get_model_summary(self) -> str:
        """
        Get a formatted summary of all model results.
        
        Returns:
            Formatted string summary
        """
        if not self.model_results:
            return "No model results available. Train and evaluate models first."
        
        summary = []
        summary.append("=" * 80)
        summary.append("TRADITIONAL ML MODELS SUMMARY")
        summary.append("=" * 80)
        
        # Create results table
        summary.append(f"{'Model':<20} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'AUC':<10}")
        summary.append("-" * 80)
        
        for model_name, results in self.model_results.items():
            auc_str = f"{results['auc']:.4f}" if results['auc'] is not None else "N/A"
            summary.append(
                f"{model_name:<20} "
                f"{results['accuracy']:<10.4f} "
                f"{results['precision']:<10.4f} "
                f"{results['recall']:<10.4f} "
                f"{results['f1_score']:<10.4f} "
                f"{auc_str:<10}"
            )
        
        summary.append("-" * 80)
        
        # Find best model
        best_model = max(self.model_results.items(), key=lambda x: x[1]['accuracy'])
        summary.append(f"Best Model (by Accuracy): {best_model[0]} ({best_model[1]['accuracy']:.4f})")
        
        summary.append("=" * 80)
        
        return "\n".join(summary)
    
    def save_results(self, output_dir: str) -> None:
        """
        Save all model results to files.
        
        Args:
            output_dir: Directory to save results
        """
        from pathlib import Path
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving model results to {output_path}")
        
        # Save model results
        if self.model_results:
            results_file = output_path / "ml_model_results.json"
            save_results(self.model_results, results_file)
        
        # Save feature importance
        if self.feature_importance:
            for model_name, importance_df in self.feature_importance.items():
                importance_file = output_path / f"{model_name.lower()}_feature_importance.csv"
                importance_df.to_csv(importance_file, index=False)
        
        # Save model summary
        summary_file = output_path / "model_summary.txt"
        with open(summary_file, 'w') as f:
            f.write(self.get_model_summary())
        
        logger.info("Model results saved successfully")


def main():
    """Main function for testing the TraditionalMLModels."""
    try:
        from data_preparation.data_loader import DataLoader
        from data_preparation.data_splitter import DataSplitter
        
        # Load and split data
        loader = DataLoader()
        data = loader.load_data()
        
        splitter = DataSplitter()
        X_train, X_test, y_train, y_test = splitter.split_data(data)
        
        # Initialize ML models
        ml_models = TraditionalMLModels()
        
        # Train models
        trained_models = ml_models.train_all_models(X_train, y_train)
        
        # Evaluate models
        results = ml_models.evaluate_all_models(X_test, y_test)
        
        # Get feature importance
        feature_names = X_train.columns.tolist()
        importance = ml_models.get_feature_importance(feature_names)
        
        # Print summary
        summary = ml_models.get_model_summary()
        print(summary)
        
        # Save results
        ml_models.save_results("results/ml_results")
        
        print("\n🎉 Traditional ML models implementation completed!")
        
    except Exception as e:
        logger.error(f"Error in main: {e}")
        raise


if __name__ == "__main__":
    main()
