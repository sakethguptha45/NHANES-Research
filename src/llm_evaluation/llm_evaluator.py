import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import warnings

from utils.logger import get_logger
from utils.config import RESULTS_DIR, TARGET_COLUMN, RANDOM_STATE
from utils.helpers import save_results, load_results
from llm_evaluation.llm_client import OllamaClient

logger = get_logger(__name__)

class LLMEvaluator:
    """
    Evaluates Large Language Models (LLMs) for clinical prediction tasks
    using Ollama models (Llama 3.1 and Mistral).
    """
    
    def __init__(self, models: List[str] = None):
        """
        Initialize the LLM Evaluator.
        Args:
            models (List[str]): List of Ollama model names to evaluate.
        """
        if models is None:
            models = ["llama3.1:8b", "mistral:7b"]
        
        self.models = models
        self.llm_clients: Dict[str, OllamaClient] = {}
        self.evaluation_results: Dict[str, Any] = {}
        self.comparison_results: Dict[str, Any] = {}
        
        # Initialize clients for each model
        for model_name in models:
            self.llm_clients[model_name] = OllamaClient(model_name)
        
        logger.info(f"LLMEvaluator initialized with models: {models}")
    
    def test_all_connections(self) -> Dict[str, bool]:
        """
        Test connections to all LLM models.
        Returns:
            Dict[str, bool]: Connection status for each model.
        """
        logger.info("Testing connections to all LLM models...")
        connection_status = {}
        
        for model_name, client in self.llm_clients.items():
            logger.info(f"Testing connection to {model_name}...")
            status = client.test_connection()
            connection_status[model_name] = status
            
            if status:
                logger.info(f"✅ {model_name} connection successful")
            else:
                logger.warning(f"❌ {model_name} connection failed")
        
        logger.info(f"Connection test completed: {connection_status}")
        return connection_status
    
    def evaluate_model(self, model_name: str, X_test: pd.DataFrame, y_test: pd.Series, 
                      sample_size: int = None) -> Dict[str, Any]:
        """
        Evaluate a single LLM model on test data.
        Args:
            model_name (str): Name of the model to evaluate.
            X_test (pd.DataFrame): Test features.
            y_test (pd.Series): True labels for test set.
            sample_size (int): Number of samples to evaluate (None for all).
        Returns:
            Dict[str, Any]: Evaluation results for the model.
        """
        logger.info(f"Evaluating {model_name} on test data...")
        
        if model_name not in self.llm_clients:
            logger.error(f"Model {model_name} not found in available clients")
            return {}
        
        client = self.llm_clients[model_name]
        
        if not client.is_available:
            logger.error(f"Model {model_name} is not available")
            return {}
        
        # Determine sample size
        if sample_size is None:
            sample_size = len(X_test)
        else:
            sample_size = min(sample_size, len(X_test))
        
        logger.info(f"Evaluating on {sample_size} samples")
        
        # Sample the test data
        if sample_size < len(X_test):
            test_indices = np.random.choice(len(X_test), size=sample_size, replace=False)
            X_sample = X_test.iloc[test_indices]
            y_sample = y_test.iloc[test_indices]
        else:
            X_sample = X_test
            y_sample = y_test
        
        # Convert features to dictionaries for LLM input
        features_list = []
        for idx, row in X_sample.iterrows():
            features_dict = row.to_dict()
            features_list.append(features_dict)
        
        # Make predictions
        logger.info(f"Making predictions with {model_name}...")
        predictions = client.batch_predict(features_list, batch_size=5)
        
        # Filter out None predictions
        valid_indices = [i for i, pred in enumerate(predictions) if pred is not None]
        valid_predictions = [predictions[i] for i in valid_indices]
        valid_y_true = [y_sample.iloc[i] for i in valid_indices]
        
        if len(valid_predictions) == 0:
            logger.error(f"No valid predictions from {model_name}")
            return {}
        
        logger.info(f"Valid predictions: {len(valid_predictions)}/{len(predictions)}")
        
        # Calculate metrics
        accuracy = accuracy_score(valid_y_true, valid_predictions)
        precision = precision_score(valid_y_true, valid_predictions, zero_division=0)
        recall = recall_score(valid_y_true, valid_predictions, zero_division=0)
        f1 = f1_score(valid_y_true, valid_predictions, zero_division=0)
        mcc = matthews_corrcoef(valid_y_true, valid_predictions)
        
        # Confusion matrix
        cm = confusion_matrix(valid_y_true, valid_predictions)
        
        # Classification report
        class_report = classification_report(valid_y_true, valid_predictions, output_dict=True)
        
        results = {
            'model_name': model_name,
            'sample_size': len(valid_predictions),
            'total_samples': len(predictions),
            'success_rate': len(valid_predictions) / len(predictions),
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'mcc': mcc,
            'confusion_matrix': cm.tolist(),
            'classification_report': class_report,
            'predictions': valid_predictions,
            'true_labels': valid_y_true
        }
        
        logger.info(f"✅ {model_name} evaluation completed:")
        logger.info(f"  Accuracy: {accuracy:.4f}")
        logger.info(f"  Precision: {precision:.4f}")
        logger.info(f"  Recall: {recall:.4f}")
        logger.info(f"  F1-score: {f1:.4f}")
        logger.info(f"  MCC: {mcc:.4f}")
        
        return results
    
    def evaluate_all_models(self, X_test: pd.DataFrame, y_test: pd.Series, 
                           sample_size: int = None) -> Dict[str, Any]:
        """
        Evaluate all LLM models on test data.
        Args:
            X_test (pd.DataFrame): Test features.
            y_test (pd.Series): True labels for test set.
            sample_size (int): Number of samples to evaluate per model.
        Returns:
            Dict[str, Any]: Evaluation results for all models.
        """
        logger.info("Evaluating all LLM models...")
        
        all_results = {}
        
        for model_name in self.models:
            logger.info(f"Evaluating {model_name}...")
            results = self.evaluate_model(model_name, X_test, y_test, sample_size)
            
            if results:
                all_results[model_name] = results
            else:
                logger.warning(f"Failed to evaluate {model_name}")
        
        self.evaluation_results = all_results
        logger.info(f"All model evaluation completed: {list(all_results.keys())}")
        
        return all_results
    
    def compare_models(self) -> Dict[str, Any]:
        """
        Compare performance across all evaluated models.
        Returns:
            Dict[str, Any]: Comparison results.
        """
        logger.info("Comparing LLM models...")
        
        if not self.evaluation_results:
            logger.error("No evaluation results available for comparison")
            return {}
        
        # Create comparison summary
        comparison_data = []
        for model_name, results in self.evaluation_results.items():
            comparison_data.append({
                'Model': model_name,
                'Accuracy': results['accuracy'],
                'Precision': results['precision'],
                'Recall': results['recall'],
                'F1-Score': results['f1_score'],
                'MCC': results['mcc'],
                'Success_Rate': results['success_rate'],
                'Sample_Size': results['sample_size']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Find best model for each metric
        best_models = {}
        for metric in ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'MCC']:
            best_idx = comparison_df[metric].idxmax()
            best_models[metric] = {
                'model': comparison_df.loc[best_idx, 'Model'],
                'score': comparison_df.loc[best_idx, metric]
            }
        
        # Overall ranking (based on F1-score)
        comparison_df = comparison_df.sort_values('F1-Score', ascending=False)
        comparison_df['Rank'] = range(1, len(comparison_df) + 1)
        
        comparison_results = {
            'summary_table': comparison_df.to_dict('records'),
            'best_models': best_models,
            'overall_ranking': comparison_df[['Model', 'F1-Score', 'Rank']].to_dict('records')
        }
        
        self.comparison_results = comparison_results
        
        logger.info("Model comparison completed")
        logger.info(f"Best overall model: {best_models['F1-Score']['model']} (F1: {best_models['F1-Score']['score']:.4f})")
        
        return comparison_results
    
    def get_evaluation_summary(self) -> str:
        """
        Get a summary of evaluation results.
        Returns:
            str: Formatted summary string.
        """
        if not self.evaluation_results:
            return "No evaluation results available."
        
        summary = "LLM Model Evaluation Summary\n"
        summary += "=" * 50 + "\n\n"
        
        for model_name, results in self.evaluation_results.items():
            summary += f"Model: {model_name}\n"
            summary += f"  Sample Size: {results['sample_size']}\n"
            summary += f"  Success Rate: {results['success_rate']:.2%}\n"
            summary += f"  Accuracy: {results['accuracy']:.4f}\n"
            summary += f"  Precision: {results['precision']:.4f}\n"
            summary += f"  Recall: {results['recall']:.4f}\n"
            summary += f"  F1-Score: {results['f1_score']:.4f}\n"
            summary += f"  MCC: {results['mcc']:.4f}\n\n"
        
        if self.comparison_results:
            summary += "Best Models by Metric:\n"
            for metric, info in self.comparison_results['best_models'].items():
                summary += f"  {metric}: {info['model']} ({info['score']:.4f})\n"
        
        return summary
    
    def create_evaluation_plots(self, save_dir: Path) -> None:
        """
        Create visualization plots for LLM evaluation results.
        Args:
            save_dir (Path): Directory to save plots.
        """
        if not self.evaluation_results:
            logger.warning("No evaluation results available for plotting")
            return
        
        logger.info("Creating LLM evaluation plots...")
        
        # Create plots directory
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Model Performance Comparison
        self._plot_model_comparison(save_dir)
        
        # 2. Confusion Matrices
        self._plot_confusion_matrices(save_dir)
        
        # 3. Success Rate Analysis
        self._plot_success_rates(save_dir)
        
        logger.info(f"LLM evaluation plots saved to {save_dir}")
    
    def _plot_model_comparison(self, save_dir: Path) -> None:
        """Create model performance comparison plot."""
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'MCC']
        model_names = list(self.evaluation_results.keys())
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        x = np.arange(len(metrics))
        width = 0.8 / len(model_names)
        
        for i, model_name in enumerate(model_names):
            results = self.evaluation_results[model_name]
            values = [results['accuracy'], results['precision'], 
                     results['recall'], results['f1_score'], results['mcc']]
            
            ax.bar(x + i * width, values, width, label=model_name, alpha=0.8)
        
        ax.set_xlabel('Metrics')
        ax.set_ylabel('Score')
        ax.set_title('LLM Model Performance Comparison')
        ax.set_xticks(x + width * (len(model_names) - 1) / 2)
        ax.set_xticklabels(metrics)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_dir / 'llm_model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_confusion_matrices(self, save_dir: Path) -> None:
        """Create confusion matrix plots for all models."""
        n_models = len(self.evaluation_results)
        fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4))
        
        if n_models == 1:
            axes = [axes]
        
        for i, (model_name, results) in enumerate(self.evaluation_results.items()):
            cm = np.array(results['confusion_matrix'])
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i])
            axes[i].set_title(f'{model_name} Confusion Matrix')
            axes[i].set_xlabel('Predicted')
            axes[i].set_ylabel('Actual')
        
        plt.tight_layout()
        plt.savefig(save_dir / 'llm_confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_success_rates(self, save_dir: Path) -> None:
        """Create success rate analysis plot."""
        model_names = list(self.evaluation_results.keys())
        success_rates = [results['success_rate'] for results in self.evaluation_results.values()]
        sample_sizes = [results['sample_size'] for results in self.evaluation_results.values()]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Success rate bar plot
        bars = ax1.bar(model_names, success_rates, alpha=0.7, color='skyblue')
        ax1.set_ylabel('Success Rate')
        ax1.set_title('LLM Prediction Success Rates')
        ax1.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, rate in zip(bars, success_rates):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{rate:.2%}', ha='center', va='bottom')
        
        # Sample size bar plot
        bars2 = ax2.bar(model_names, sample_sizes, alpha=0.7, color='lightcoral')
        ax2.set_ylabel('Sample Size')
        ax2.set_title('LLM Evaluation Sample Sizes')
        
        # Add value labels on bars
        for bar, size in zip(bars2, sample_sizes):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(sample_sizes)*0.01,
                    f'{size}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(save_dir / 'llm_success_rates.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_evaluation_results(self, save_dir: Path) -> None:
        """
        Save evaluation results to files.
        Args:
            save_dir (Path): Directory to save results.
        """
        logger.info(f"Saving LLM evaluation results to {save_dir}")
        
        # Create directory
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save detailed results
        save_results(self.evaluation_results, save_dir / "llm_evaluation_results.json")
        
        # Save comparison results
        if self.comparison_results:
            save_results(self.comparison_results, save_dir / "llm_comparison_results.json")
        
        # Save summary
        summary = self.get_evaluation_summary()
        with open(save_dir / "llm_evaluation_summary.txt", 'w') as f:
            f.write(summary)
        
        logger.info("LLM evaluation results saved successfully")
