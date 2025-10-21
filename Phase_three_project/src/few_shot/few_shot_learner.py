"""
Few-Shot Learning implementation for Phase 3.

This module implements few-shot learning strategy for LLMs,
where we provide a few examples to improve prediction accuracy.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import requests
import json
import time

from utils.config import OLLAMA_BASE_URL, OLLAMA_MODELS, LLM_TEMPERATURE, LLM_MAX_TOKENS, LLM_TOP_P, LLM_BATCH_SIZE
from utils.helpers import (
    setup_logger, build_few_shot_prompt, calculate_performance_metrics,
    save_results, format_features_for_prompt, convert_features_to_text
)

logger = setup_logger(__name__)

class FewShotLLMClient:
    """
    LLM client specifically designed for few-shot learning.
    """
    
    def __init__(self, model_name: str):
        """
        Initialize the few-shot LLM client.
        
        Args:
            model_name: Name of the Ollama model to use
        """
        self.model_name = model_name
        self.base_url = OLLAMA_BASE_URL
        logger.info(f"FewShotLLMClient initialized with model: {self.model_name}")
    
    def _call_ollama_api(self, prompt: str) -> Optional[str]:
        """
        Call the Ollama API with the given prompt.
        
        Args:
            prompt: The prompt to send to the LLM
            
        Returns:
            LLM response or None if error
        """
        url = f"{self.base_url}/api/generate"
        headers = {"Content-Type": "application/json"}
        data = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": LLM_TEMPERATURE,
                "num_predict": LLM_MAX_TOKENS,
                "top_p": LLM_TOP_P,
            }
        }
        
        try:
            response = requests.post(url, headers=headers, data=json.dumps(data), timeout=60)
            response.raise_for_status()
            
            result = response.json()
            return result.get("response", "").strip()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error calling Ollama API: {e}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON response: {e}")
            return None
    
    def predict_with_few_shot(self, examples: List[Dict[str, Any]], test_features: Dict[str, Any]) -> Optional[int]:
        """
        Make a prediction using few-shot learning.
        
        Args:
            examples: List of few-shot examples
            test_features: Features for the test case
            
        Returns:
            Predicted label (0 or 1) or None if prediction fails
        """
        # Convert test features to text for better LLM understanding
        test_features_text = convert_features_to_text(test_features)
        
        # Build few-shot prompt with textual features
        prompt = build_few_shot_prompt(examples, test_features, test_features_text)
        
        # Get LLM response
        response = self._call_ollama_api(prompt)
        
        if response is None:
            return None
        
        # Parse response with improved parsing
        try:
            response_clean = response.strip()
            if response_clean in ['0', '1']:
                return int(response_clean)
            else:
                # Try to find number in response
                import re
                numbers = re.findall(r'\b[01]\b', response_clean)
                if numbers:
                    return int(numbers[0])
                else:
                    # More lenient parsing
                    txt = response_clean.lower()
                    if " 1" in txt or txt.endswith("1") or "poor" in txt or "bad" in txt:
                        return 1
                    if " 0" in txt or txt.endswith("0") or "good" in txt or "healthy" in txt:
                        return 0
                    
                    logger.warning(f"Could not parse LLM response: {response}")
                    return None
        except Exception as e:
            logger.error(f"Error parsing LLM response: {e}")
            return None
    
    def batch_predict_with_few_shot(self, examples: List[Dict[str, Any]], 
                                   test_cases: List[Dict[str, Any]]) -> List[Optional[int]]:
        """
        Make batch predictions using few-shot learning.
        
        Args:
            examples: List of few-shot examples
            test_cases: List of test cases
            
        Returns:
            List of predictions
        """
        predictions = []
        
        logger.info(f"Making few-shot predictions for {len(test_cases)} test cases")
        
        for i, test_case in enumerate(test_cases):
            if i % 10 == 0:
                logger.info(f"Processing test case {i+1}/{len(test_cases)}")
            
            prediction = self.predict_with_few_shot(examples, test_case)
            predictions.append(prediction)
            
            # Early stopping if success rate is too low
            if (i + 1) % 20 == 0:
                current_success = sum(1 for p in predictions if p is not None) / len(predictions)
                if current_success < 0.05:
                    logger.warning(f"Early stop: success rate {current_success:.1%} < 5%")
                    break
            
            # Small delay to avoid overwhelming the server
            time.sleep(0.1)
        
        success_rate = sum(1 for p in predictions if p is not None) / len(predictions)
        logger.info(f"Few-shot batch prediction completed. Success rate: {success_rate:.2%}")
        
        return predictions

class FewShotEvaluator:
    """
    Evaluator for few-shot learning strategy.
    """
    
    def __init__(self, model_names: List[str]):
        """
        Initialize the few-shot evaluator.
        
        Args:
            model_names: List of Ollama model names to evaluate
        """
        self.model_names = model_names
        self.clients = {name: FewShotLLMClient(name) for name in model_names}
        self.results: Dict[str, Dict[str, Any]] = {}
        
        logger.info(f"FewShotEvaluator initialized with models: {model_names}")
    
    def evaluate_models(self, examples: List[Dict[str, Any]], test_df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """
        Evaluate all models using few-shot learning.
        
        Args:
            examples: Few-shot examples
            test_df: Test DataFrame with features and labels
            
        Returns:
            Dictionary of evaluation results
        """
        logger.info("Starting few-shot evaluation...")
        
        # Prepare test cases
        test_cases = test_df.drop(columns=['Health_status']).to_dict(orient='records')
        true_labels = test_df['Health_status'].tolist()
        
        for model_name, client in self.clients.items():
            logger.info(f"Evaluating {model_name} with few-shot learning...")
            
            # Make predictions
            predictions = client.batch_predict_with_few_shot(examples, test_cases)
            
            # Filter out None predictions
            valid_predictions = [p for p in predictions if p is not None]
            valid_indices = [i for i, p in enumerate(predictions) if p is not None]
            valid_true_labels = [true_labels[i] for i in valid_indices]
            
            if not valid_predictions:
                logger.warning(f"No valid predictions for {model_name}")
                self.results[model_name] = {
                    'accuracy': 0.0,
                    'precision': 0.0,
                    'recall': 0.0,
                    'f1_score': 0.0,
                    'mcc': 0.0,
                    'confusion_matrix': [[0, 0], [0, 0]],
                    'sample_size': len(test_df),
                    'valid_predictions': 0,
                    'success_rate': 0.0
                }
                continue
            
            # Calculate metrics
            metrics = calculate_performance_metrics(valid_true_labels, valid_predictions)
            
            # Add additional information
            metrics.update({
                'sample_size': len(test_df),
                'valid_predictions': len(valid_predictions),
                'success_rate': len(valid_predictions) / len(test_df),
                'predictions': valid_predictions,
                'true_labels': valid_true_labels
            })
            
            self.results[model_name] = metrics
            
            logger.info(f"✅ {model_name} few-shot evaluation completed:")
            logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
            logger.info(f"  Precision: {metrics['precision']:.4f}")
            logger.info(f"  Recall: {metrics['recall']:.4f}")
            logger.info(f"  F1-Score: {metrics['f1_score']:.4f}")
            logger.info(f"  Success Rate: {metrics['success_rate']:.2%}")
        
        logger.info("Few-shot evaluation completed for all models")
        return self.results
    
    def get_best_model(self) -> Tuple[str, Dict[str, Any]]:
        """
        Get the best performing model based on F1-score.
        
        Returns:
            Tuple of (model_name, results)
        """
        if not self.results:
            return None, {}
        
        best_model = max(self.results.keys(), key=lambda k: self.results[k].get('f1_score', 0))
        return best_model, self.results[best_model]
    
    def save_results(self, save_dir: Path) -> None:
        """
        Save few-shot evaluation results.
        
        Args:
            save_dir: Directory to save results
        """
        logger.info(f"Saving few-shot results to {save_dir}")
        
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save detailed results
        save_results(self.results, save_dir / "few_shot_results.json")
        
        # Save summary
        summary_lines = [
            "FEW-SHOT LEARNING EVALUATION SUMMARY",
            "=" * 40,
            ""
        ]
        
        for model_name, metrics in self.results.items():
            summary_lines.append(f"Model: {model_name}")
            summary_lines.append(f"  Accuracy: {metrics.get('accuracy', 0):.4f}")
            summary_lines.append(f"  Precision: {metrics.get('precision', 0):.4f}")
            summary_lines.append(f"  Recall: {metrics.get('recall', 0):.4f}")
            summary_lines.append(f"  F1-Score: {metrics.get('f1_score', 0):.4f}")
            summary_lines.append(f"  Success Rate: {metrics.get('success_rate', 0):.2%}")
            summary_lines.append("")
        
        # Add best model
        best_model, best_metrics = self.get_best_model()
        if best_model:
            summary_lines.append(f"BEST MODEL: {best_model}")
            summary_lines.append(f"Best F1-Score: {best_metrics.get('f1_score', 0):.4f}")
        
        summary_text = "\n".join(summary_lines)
        
        with open(save_dir / "few_shot_summary.txt", 'w') as f:
            f.write(summary_text)
        
        logger.info("Few-shot results saved successfully")
