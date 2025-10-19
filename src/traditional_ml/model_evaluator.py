"""
Model evaluator module for LLM vs Traditional ML comparison project.

This module provides comprehensive model evaluation functionality including
detailed metrics calculation, visualization, and statistical analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve,
    confusion_matrix, classification_report, matthews_corrcoef
)
from scipy import stats
import warnings

from .baseline_models import TraditionalMLModels
from utils.config import RANDOM_STATE, RESULTS_DIR
from utils.logger import get_logger
from utils.helpers import save_results, format_percentage

logger = get_logger(__name__)

class ModelEvaluator:
    """
    Comprehensive model evaluator for traditional ML models.
    
    This class provides detailed evaluation capabilities including metrics
    calculation, visualization, statistical analysis, and comparison.
    """
    
    def __init__(self, models: Optional[TraditionalMLModels] = None):
        """
        Initialize the ModelEvaluator.
        
        Args:
            models: TraditionalMLModels instance. If None, creates new instance.
        """
        self.models = models or TraditionalMLModels()
        self.evaluation_results: Dict[str, Dict[str, Any]] = {}
        self.comparison_results: Dict[str, Any] = {}
        
        # Set plotting style
        plt.style.use('seaborn-v0_8')
        
        logger.info("ModelEvaluator initialized")
    
    def evaluate_comprehensive(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Dict[str, Any]]:
        """
        Perform comprehensive evaluation of all models.
        
        Args:
            X_test: Test features
            y_test: Test target
            
        Returns:
            Dictionary containing comprehensive evaluation results
        """
        logger.info("Starting comprehensive model evaluation...")
        
        if not self.models.trained_models:
            raise ValueError("No trained models found. Train models first.")
        
        # Log test data info
        logger.info(f"Test data shape: {X_test.shape}")
        logger.info(f"Test target distribution: {y_test.value_counts().to_dict()}")
        
        comprehensive_results = {}
        
        for model_name, model in self.models.trained_models.items():
            try:
                logger.info(f"Evaluating {model_name} comprehensively...")
                
                # Get predictions
                y_pred = model.predict(X_test)
                y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
                
                # Calculate comprehensive metrics
                results = self._calculate_comprehensive_metrics(y_test, y_pred, y_proba)
                
                # Add model-specific information
                results['model_name'] = model_name
                results['model_type'] = type(model).__name__
                
                comprehensive_results[model_name] = results
                
                logger.info(f"✅ {model_name} comprehensive evaluation completed")
                
            except Exception as e:
                logger.error(f"❌ {model_name} evaluation failed: {e}")
                continue
        
        self.evaluation_results = comprehensive_results
        logger.info(f"Comprehensive evaluation completed for: {list(comprehensive_results.keys())}")
        return comprehensive_results
    
    def _calculate_comprehensive_metrics(self, y_true: pd.Series, y_pred: np.ndarray, 
                                       y_proba: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Calculate comprehensive evaluation metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities (if available)
            
        Returns:
            Dictionary of comprehensive metrics
        """
        metrics = {}
        
        # Basic classification metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, average='weighted')
        metrics['recall'] = recall_score(y_true, y_pred, average='weighted')
        metrics['f1_score'] = f1_score(y_true, y_pred, average='weighted')
        
        # Per-class metrics
        metrics['precision_per_class'] = precision_score(y_true, y_pred, average=None).tolist()
        metrics['recall_per_class'] = recall_score(y_true, y_pred, average=None).tolist()
        metrics['f1_per_class'] = f1_score(y_true, y_pred, average=None).tolist()
        
        # Additional metrics
        metrics['matthews_corrcoef'] = matthews_corrcoef(y_true, y_pred)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        metrics['confusion_matrix_normalized'] = (cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]).tolist()
        
        # Confusion matrix components
        tn, fp, fn, tp = cm.ravel()
        metrics['true_negatives'] = int(tn)
        metrics['false_positives'] = int(fp)
        metrics['false_negatives'] = int(fn)
        metrics['true_positives'] = int(tp)
        
        # Specificity and Sensitivity
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        # ROC and AUC metrics
        if y_proba is not None:
            try:
                metrics['auc_roc'] = roc_auc_score(y_true, y_proba)
                
                # ROC curve data
                fpr, tpr, roc_thresholds = roc_curve(y_true, y_proba)
                metrics['roc_curve'] = {
                    'fpr': fpr.tolist(),
                    'tpr': tpr.tolist(),
                    'thresholds': roc_thresholds.tolist()
                }
                
                # Precision-Recall curve
                precision, recall, pr_thresholds = precision_recall_curve(y_true, y_proba)
                metrics['precision_recall_curve'] = {
                    'precision': precision.tolist(),
                    'recall': recall.tolist(),
                    'thresholds': pr_thresholds.tolist()
                }
                
                # AUC-PR
                metrics['auc_pr'] = np.trapz(precision, recall)
                
            except ValueError as e:
                logger.warning(f"ROC/AUC calculation failed: {e}")
                metrics['auc_roc'] = None
                metrics['auc_pr'] = None
        
        # Classification report
        metrics['classification_report'] = classification_report(y_true, y_pred, output_dict=True)
        
        return metrics
    
    def compare_models(self) -> Dict[str, Any]:
        """
        Compare all models and identify the best performing model.
        
        Returns:
            Dictionary containing comparison results
        """
        if not self.evaluation_results:
            raise ValueError("No evaluation results found. Run evaluate_comprehensive() first.")
        
        logger.info("Comparing models...")
        
        comparison = {
            'model_rankings': {},
            'best_model': {},
            'statistical_tests': {},
            'summary_stats': {}
        }
        
        # Create comparison DataFrame
        comparison_data = []
        for model_name, results in self.evaluation_results.items():
            comparison_data.append({
                'Model': model_name,
                'Accuracy': results['accuracy'],
                'Precision': results['precision'],
                'Recall': results['recall'],
                'F1-Score': results['f1_score'],
                'AUC-ROC': results.get('auc_roc', None),
                'AUC-PR': results.get('auc_pr', None),
                'MCC': results['matthews_corrcoef']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Rank models by different metrics
        metrics_to_rank = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'MCC']
        for metric in metrics_to_rank:
            comparison['model_rankings'][metric] = comparison_df.nlargest(len(comparison_df), metric)[['Model', metric]].to_dict('records')
        
        # Overall best model (by accuracy)
        best_model_name = comparison_df.loc[comparison_df['Accuracy'].idxmax(), 'Model']
        comparison['best_model'] = {
            'name': best_model_name,
            'metrics': self.evaluation_results[best_model_name]
        }
        
        # Summary statistics
        comparison['summary_stats'] = {
            'mean_accuracy': comparison_df['Accuracy'].mean(),
            'std_accuracy': comparison_df['Accuracy'].std(),
            'best_accuracy': comparison_df['Accuracy'].max(),
            'worst_accuracy': comparison_df['Accuracy'].min(),
            'accuracy_range': comparison_df['Accuracy'].max() - comparison_df['Accuracy'].min()
        }
        
        # Statistical significance tests (if more than 1 model)
        if len(self.evaluation_results) > 1:
            comparison['statistical_tests'] = self._perform_statistical_tests()
        
        self.comparison_results = comparison
        logger.info("Model comparison completed")
        return comparison
    
    def _perform_statistical_tests(self) -> Dict[str, Any]:
        """
        Perform statistical significance tests between models.
        
        Returns:
            Dictionary containing statistical test results
        """
        logger.info("Performing statistical significance tests...")
        
        # Extract accuracy scores for all models
        model_names = list(self.evaluation_results.keys())
        accuracy_scores = [self.evaluation_results[name]['accuracy'] for name in model_names]
        
        statistical_tests = {}
        
        # Paired t-test between best and second-best models
        if len(model_names) >= 2:
            # Sort by accuracy
            sorted_models = sorted(zip(model_names, accuracy_scores), key=lambda x: x[1], reverse=True)
            best_model, best_score = sorted_models[0]
            second_best_model, second_best_score = sorted_models[1]
            
            # For demonstration, we'll use the accuracy scores directly
            # In practice, you'd want to use cross-validation scores
            t_stat, p_value = stats.ttest_rel([best_score], [second_best_score])
            
            statistical_tests['best_vs_second'] = {
                'best_model': best_model,
                'second_best_model': second_best_model,
                'best_score': best_score,
                'second_best_score': second_best_score,
                't_statistic': t_stat,
                'p_value': p_value,
                'significant': p_value < 0.05
            }
        
        return statistical_tests
    
    def create_evaluation_plots(self, output_dir: str) -> None:
        """
        Create comprehensive evaluation plots.
        
        Args:
            output_dir: Directory to save plots
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Creating evaluation plots in {output_path}")
        
        if not self.evaluation_results:
            logger.warning("No evaluation results found. Run evaluate_comprehensive() first.")
            return
        
        # 1. Model comparison bar chart
        self._plot_model_comparison(output_path)
        
        # 2. ROC curves
        self._plot_roc_curves(output_path)
        
        # 3. Precision-Recall curves
        self._plot_precision_recall_curves(output_path)
        
        # 4. Confusion matrices
        self._plot_confusion_matrices(output_path)
        
        # 5. Feature importance (if available)
        if self.models.feature_importance:
            self._plot_feature_importance(output_path)
        
        logger.info("Evaluation plots created successfully")
    
    def _plot_model_comparison(self, output_path: Path) -> None:
        """Create model comparison bar chart."""
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        
        # Prepare data
        comparison_data = []
        for model_name, results in self.evaluation_results.items():
            comparison_data.append({
                'Model': model_name,
                'Accuracy': results['accuracy'],
                'Precision': results['precision'],
                'Recall': results['recall'],
                'F1-Score': results['f1_score']
            })
        
        df = pd.DataFrame(comparison_data)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        x = np.arange(len(df))
        width = 0.2
        
        for i, metric in enumerate(metrics):
            ax.bar(x + i * width, df[metric], width, label=metric)
        
        ax.set_xlabel('Models')
        ax.set_ylabel('Score')
        ax.set_title('Model Performance Comparison')
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(df['Model'], rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / 'model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_roc_curves(self, output_path: Path) -> None:
        """Create ROC curves plot."""
        plt.figure(figsize=(10, 8))
        
        for model_name, results in self.evaluation_results.items():
            if 'roc_curve' in results:
                roc_data = results['roc_curve']
                plt.plot(roc_data['fpr'], roc_data['tpr'], 
                        label=f"{model_name} (AUC = {results['auc_roc']:.3f})")
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / 'roc_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_precision_recall_curves(self, output_path: Path) -> None:
        """Create Precision-Recall curves plot."""
        plt.figure(figsize=(10, 8))
        
        for model_name, results in self.evaluation_results.items():
            if 'precision_recall_curve' in results:
                pr_data = results['precision_recall_curve']
                plt.plot(pr_data['recall'], pr_data['precision'], 
                        label=f"{model_name} (AUC-PR = {results['auc_pr']:.3f})")
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / 'precision_recall_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_confusion_matrices(self, output_path: Path) -> None:
        """Create confusion matrices plot."""
        n_models = len(self.evaluation_results)
        cols = min(2, n_models)
        rows = (n_models + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
        if n_models == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, (model_name, results) in enumerate(self.evaluation_results.items()):
            row = i // cols
            col = i % cols
            
            cm = np.array(results['confusion_matrix'])
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       ax=axes[row, col] if rows > 1 else axes[col])
            axes[row, col].set_title(f'{model_name} Confusion Matrix')
            axes[row, col].set_xlabel('Predicted')
            axes[row, col].set_ylabel('Actual')
        
        # Hide empty subplots
        for i in range(n_models, rows * cols):
            row = i // cols
            col = i % cols
            axes[row, col].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(output_path / 'confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_feature_importance(self, output_path: Path) -> None:
        """Create feature importance plots."""
        n_models = len(self.models.feature_importance)
        cols = min(2, n_models)
        rows = (n_models + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
        if n_models == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, (model_name, importance_df) in enumerate(self.models.feature_importance.items()):
            row = i // cols
            col = i % cols
            
            # Get top 10 features
            top_features = importance_df.head(10)
            
            # Determine importance column
            if 'importance' in top_features.columns:
                importance_col = 'importance'
            elif 'abs_coefficient' in top_features.columns:
                importance_col = 'abs_coefficient'
            else:
                continue
            
            axes[row, col].barh(range(len(top_features)), top_features[importance_col])
            axes[row, col].set_yticks(range(len(top_features)))
            axes[row, col].set_yticklabels(top_features['feature'])
            axes[row, col].set_xlabel('Importance')
            axes[row, col].set_title(f'{model_name} Feature Importance')
            axes[row, col].invert_yaxis()
        
        # Hide empty subplots
        for i in range(n_models, rows * cols):
            row = i // cols
            col = i % cols
            axes[row, col].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(output_path / 'feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def get_evaluation_summary(self) -> str:
        """
        Get a comprehensive evaluation summary.
        
        Returns:
            Formatted string summary
        """
        if not self.evaluation_results:
            return "No evaluation results available. Run evaluate_comprehensive() first."
        
        summary = []
        summary.append("=" * 100)
        summary.append("COMPREHENSIVE MODEL EVALUATION SUMMARY")
        summary.append("=" * 100)
        
        # Model performance table
        summary.append(f"{'Model':<20} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'AUC-ROC':<10} {'MCC':<10}")
        summary.append("-" * 100)
        
        for model_name, results in self.evaluation_results.items():
            auc_str = f"{results.get('auc_roc', 0):.4f}" if results.get('auc_roc') is not None else "N/A"
            summary.append(
                f"{model_name:<20} "
                f"{results['accuracy']:<10.4f} "
                f"{results['precision']:<10.4f} "
                f"{results['recall']:<10.4f} "
                f"{results['f1_score']:<10.4f} "
                f"{auc_str:<10} "
                f"{results['matthews_corrcoef']:<10.4f}"
            )
        
        summary.append("-" * 100)
        
        # Best model
        if self.comparison_results:
            best_model = self.comparison_results['best_model']
            summary.append(f"\nBest Model: {best_model['name']}")
            summary.append(f"Best Accuracy: {best_model['metrics']['accuracy']:.4f}")
        
        # Summary statistics
        if self.comparison_results and 'summary_stats' in self.comparison_results:
            stats = self.comparison_results['summary_stats']
            summary.append(f"\nSummary Statistics:")
            summary.append(f"  Mean Accuracy: {stats['mean_accuracy']:.4f}")
            summary.append(f"  Std Accuracy: {stats['std_accuracy']:.4f}")
            summary.append(f"  Accuracy Range: {stats['accuracy_range']:.4f}")
        
        summary.append("=" * 100)
        
        return "\n".join(summary)
    
    def save_evaluation_results(self, output_dir: str) -> None:
        """
        Save all evaluation results to files.
        
        Args:
            output_dir: Directory to save results
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving evaluation results to {output_path}")
        
        # Save evaluation results
        if self.evaluation_results:
            results_file = output_path / "evaluation_results.json"
            save_results(self.evaluation_results, results_file)
        
        # Save comparison results
        if self.comparison_results:
            comparison_file = output_path / "comparison_results.json"
            save_results(self.comparison_results, comparison_file)
        
        # Save summary
        summary_file = output_path / "evaluation_summary.txt"
        with open(summary_file, 'w') as f:
            f.write(self.get_evaluation_summary())
        
        logger.info("Evaluation results saved successfully")


def main():
    """Main function for testing the ModelEvaluator."""
    try:
        from data_preparation.data_loader import DataLoader
        from data_preparation.data_splitter import DataSplitter
        
        # Load and split data
        loader = DataLoader()
        data = loader.load_data()
        
        splitter = DataSplitter()
        X_train, X_test, y_train, y_test = splitter.split_data(data)
        
        # Initialize models and trainer
        ml_models = TraditionalMLModels()
        ml_models.train_all_models(X_train, y_train)
        
        # Get feature importance
        feature_names = X_train.columns.tolist()
        ml_models.get_feature_importance(feature_names)
        
        # Initialize evaluator
        evaluator = ModelEvaluator(ml_models)
        
        # Comprehensive evaluation
        results = evaluator.evaluate_comprehensive(X_test, y_test)
        
        # Compare models
        comparison = evaluator.compare_models()
        
        # Create plots
        evaluator.create_evaluation_plots("results/visualizations")
        
        # Print summary
        summary = evaluator.get_evaluation_summary()
        print(summary)
        
        # Save results
        evaluator.save_evaluation_results("results/ml_results")
        
        print("\n🎉 Model evaluation completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in main: {e}")
        raise


if __name__ == "__main__":
    main()
