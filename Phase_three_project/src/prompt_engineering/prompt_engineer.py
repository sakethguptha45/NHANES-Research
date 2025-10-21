"""
Advanced Prompt Engineering implementation for Phase 3.

This module implements different prompt engineering strategies
to improve LLM performance on medical prediction tasks.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import requests
import json
import time

from utils.config import (
    OLLAMA_BASE_URL, OLLAMA_MODELS, LLM_TEMPERATURE, LLM_MAX_TOKENS, 
    LLM_TOP_P, LLM_BATCH_SIZE, PROMPT_TEMPLATES
)
from utils.helpers import (
    setup_logger, format_features_for_prompt, calculate_performance_metrics,
    save_results, convert_features_to_text
)

logger = setup_logger(__name__)

class PromptEngineeringClient:
    """
    LLM client for testing different prompt engineering strategies.
    """
    
    def __init__(self, model_name: str):
        """
        Initialize the prompt engineering client.
        
        Args:
            model_name: Name of the Ollama model to use
        """
        self.model_name = model_name
        self.base_url = OLLAMA_BASE_URL
        logger.info(f"PromptEngineeringClient initialized with model: {self.model_name}")
    
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
    
    def _parse_response(self, response: str) -> Optional[int]:
        """
        Parse LLM response to extract prediction.
        
        Args:
            response: Raw LLM response
            
        Returns:
            Parsed prediction (0 or 1) or None if parsing fails
        """
        if response is None:
            return None
        
        try:
            # Clean response
            response_clean = response.strip()
            
            # Direct match
            if response_clean in ['0', '1']:
                return int(response_clean)
            
            # Try to find number in response
            import re
            numbers = re.findall(r'\b[01]\b', response_clean)
            if numbers:
                return int(numbers[0])
            
            # More lenient parsing - check for keywords anywhere in response
            txt = response_clean.lower()
            if " 1" in txt or txt.endswith("1") or "poor" in txt or "bad" in txt or "unhealthy" in txt:
                return 1
            if " 0" in txt or txt.endswith("0") or "good" in txt or "healthy" in txt:
                return 0
            
            logger.warning(f"Could not parse LLM response: {response}")
            return None
            
        except Exception as e:
            logger.error(f"Error parsing LLM response: {e}")
            return None
    
    def predict_with_prompt_template(self, template_name: str, features: Dict[str, Any]) -> Optional[int]:
        """
        Make a prediction using a specific prompt template.
        
        Args:
            template_name: Name of the prompt template to use
            features: Patient features
            
        Returns:
            Predicted label (0 or 1) or None if prediction fails
        """
        if template_name not in PROMPT_TEMPLATES:
            logger.error(f"Unknown prompt template: {template_name}")
            return None
        
        # Format features for prompt
        features_str = format_features_for_prompt(features)
        
        # Handle textual profile template
        if template_name == "textual_profile":
            features_text = convert_features_to_text(features)
            prompt = PROMPT_TEMPLATES[template_name].format(features_text=features_text)
        else:
            # Build prompt using template
            prompt = PROMPT_TEMPLATES[template_name].format(features=features_str)
        
        # Get LLM response
        response = self._call_ollama_api(prompt)
        
        # Parse response
        prediction = self._parse_response(response)
        
        return prediction
    
    def batch_predict_with_template(self, template_name: str, test_cases: List[Dict[str, Any]]) -> List[Optional[int]]:
        """
        Make batch predictions using a specific prompt template.
        
        Args:
            template_name: Name of the prompt template to use
            test_cases: List of test cases
            
        Returns:
            List of predictions
        """
        predictions = []
        
        logger.info(f"Making predictions with {template_name} template for {len(test_cases)} test cases")
        
        for i, test_case in enumerate(test_cases):
            if i % 10 == 0:
                logger.info(f"Processing test case {i+1}/{len(test_cases)}")
            
            prediction = self.predict_with_prompt_template(template_name, test_case)
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
        logger.info(f"Batch prediction completed. Success rate: {success_rate:.2%}")
        
        return predictions

class PromptEngineeringEvaluator:
    """
    Evaluator for different prompt engineering strategies.
    """
    
    def __init__(self, model_names: List[str]):
        """
        Initialize the prompt engineering evaluator.
        
        Args:
            model_names: List of Ollama model names to evaluate
        """
        self.model_names = model_names
        self.clients = {name: PromptEngineeringClient(name) for name in model_names}
        self.results: Dict[str, Dict[str, Dict[str, Any]]] = {}
        
        logger.info(f"PromptEngineeringEvaluator initialized with models: {model_names}")
    
    def evaluate_all_templates(self, test_df: pd.DataFrame) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """
        Evaluate all prompt templates for all models.
        
        Args:
            test_df: Test DataFrame with features and labels
            
        Returns:
            Dictionary of evaluation results
        """
        logger.info("Starting prompt engineering evaluation...")
        
        # Prepare test cases
        test_cases = test_df.drop(columns=['Health_status']).to_dict(orient='records')
        true_labels = test_df['Health_status'].tolist()
        
        # Initialize results structure
        self.results = {model: {} for model in self.model_names}
        
        for model_name, client in self.clients.items():
            logger.info(f"Evaluating {model_name} with different prompt templates...")
            
            for template_name in PROMPT_TEMPLATES.keys():
                logger.info(f"  Testing template: {template_name}")
                
                # Make predictions
                predictions = client.batch_predict_with_template(template_name, test_cases)
                
                # Filter out None predictions
                valid_predictions = [p for p in predictions if p is not None]
                valid_indices = [i for i, p in enumerate(predictions) if p is not None]
                valid_true_labels = [true_labels[i] for i in valid_indices]
                
                if not valid_predictions:
                    logger.warning(f"No valid predictions for {model_name} with {template_name}")
                    self.results[model_name][template_name] = {
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
                    'template_name': template_name,
                    'predictions': valid_predictions,
                    'true_labels': valid_true_labels
                })
                
                self.results[model_name][template_name] = metrics
                
                logger.info(f"    ✅ {template_name}: F1={metrics['f1_score']:.4f}, Acc={metrics['accuracy']:.4f}")
        
        logger.info("Prompt engineering evaluation completed for all models and templates")
        return self.results
    
    def get_best_template_per_model(self) -> Dict[str, Tuple[str, Dict[str, Any]]]:
        """
        Get the best prompt template for each model based on F1-score.
        
        Returns:
            Dictionary mapping model names to (best_template, results)
        """
        best_templates = {}
        
        for model_name, template_results in self.results.items():
            if not template_results:
                continue
            
            best_template = max(template_results.keys(), 
                              key=lambda k: template_results[k].get('f1_score', 0))
            best_results = template_results[best_template]
            
            best_templates[model_name] = (best_template, best_results)
        
        return best_templates
    
    def get_overall_best_template(self) -> Tuple[str, str, Dict[str, Any]]:
        """
        Get the overall best template across all models.
        
        Returns:
            Tuple of (model_name, template_name, results)
        """
        best_overall = None
        best_f1 = 0
        
        for model_name, template_results in self.results.items():
            for template_name, results in template_results.items():
                f1_score = results.get('f1_score', 0)
                if f1_score > best_f1:
                    best_f1 = f1_score
                    best_overall = (model_name, template_name, results)
        
        return best_overall
    
    def save_results(self, save_dir: Path) -> None:
        """
        Save prompt engineering evaluation results.
        
        Args:
            save_dir: Directory to save results
        """
        logger.info(f"Saving prompt engineering results to {save_dir}")
        
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save detailed results
        save_results(self.results, save_dir / "prompt_engineering_results.json")
        
        # Save summary
        summary_lines = [
            "PROMPT ENGINEERING EVALUATION SUMMARY",
            "=" * 45,
            ""
        ]
        
        # Summary by model and template
        for model_name, template_results in self.results.items():
            summary_lines.append(f"Model: {model_name}")
            for template_name, metrics in template_results.items():
                summary_lines.append(f"  Template: {template_name}")
                summary_lines.append(f"    Accuracy: {metrics.get('accuracy', 0):.4f}")
                summary_lines.append(f"    Precision: {metrics.get('precision', 0):.4f}")
                summary_lines.append(f"    Recall: {metrics.get('recall', 0):.4f}")
                summary_lines.append(f"    F1-Score: {metrics.get('f1_score', 0):.4f}")
                summary_lines.append(f"    Success Rate: {metrics.get('success_rate', 0):.2%}")
                summary_lines.append("")
        
        # Best templates per model
        best_templates = self.get_best_template_per_model()
        summary_lines.append("BEST TEMPLATES PER MODEL:")
        summary_lines.append("-" * 30)
        for model_name, (template_name, results) in best_templates.items():
            summary_lines.append(f"{model_name}: {template_name} (F1: {results.get('f1_score', 0):.4f})")
        
        summary_lines.append("")
        
        # Overall best
        overall_best = self.get_overall_best_template()
        if overall_best:
            model_name, template_name, results = overall_best
            summary_lines.append(f"OVERALL BEST: {model_name} with {template_name}")
            summary_lines.append(f"Best F1-Score: {results.get('f1_score', 0):.4f}")
        
        summary_text = "\n".join(summary_lines)
        
        with open(save_dir / "prompt_engineering_summary.txt", 'w') as f:
            f.write(summary_text)
        
        logger.info("Prompt engineering results saved successfully")
