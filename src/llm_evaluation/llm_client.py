import subprocess
import json
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import time
import warnings
from pathlib import Path

from utils.logger import get_logger
from utils.config import RANDOM_STATE, TARGET_COLUMN, FEATURE_COLUMNS
from utils.helpers import save_results

logger = get_logger(__name__)

class OllamaClient:
    """
    Client for interacting with Ollama models (Llama 3.1 and Mistral)
    for clinical prediction tasks.
    """
    
    def __init__(self, model_name: str = "llama3.1:8b"):
        """
        Initialize the Ollama client.
        Args:
            model_name (str): Name of the Ollama model to use.
        """
        self.model_name = model_name
        self.is_available = self._check_ollama_availability()
        logger.info(f"OllamaClient initialized with model: {model_name}")
        logger.info(f"Ollama availability: {self.is_available}")
    
    def _check_ollama_availability(self) -> bool:
        """Check if Ollama is installed and the model is available."""
        try:
            # Check if Ollama is installed
            result = subprocess.run(
                ["ollama", "--version"], 
                capture_output=True, 
                text=True, 
                timeout=10
            )
            if result.returncode != 0:
                logger.warning("Ollama not found in PATH")
                return False
            
            # Check if the specific model is available
            result = subprocess.run(
                ["ollama", "list"], 
                capture_output=True, 
                text=True, 
                timeout=10
            )
            if result.returncode != 0:
                logger.warning("Failed to list Ollama models")
                return False
            
            available_models = result.stdout.lower()
            if self.model_name.lower() not in available_models:
                logger.warning(f"Model {self.model_name} not found in available models")
                return False
            
            logger.info(f"✅ Model {self.model_name} is available")
            return True
            
        except Exception as e:
            logger.error(f"Error checking Ollama availability: {e}")
            return False
    
    def _create_clinical_prompt(self, features: Dict[str, Any], target_name: str = "Health Status") -> str:
        """
        Create a clinical prediction prompt for the LLM.
        Args:
            features (Dict[str, Any]): Feature values for prediction.
            target_name (str): Name of the target variable.
        Returns:
            str: Formatted prompt for the LLM.
        """
        prompt = f"""You are a clinical prediction expert. Based on the following health and demographic features, predict the {target_name} (0 = Good Health, 1 = Poor Health).

Features:
"""
        
        for feature, value in features.items():
            if isinstance(value, (int, float)):
                prompt += f"- {feature}: {value:.2f}\n"
            else:
                prompt += f"- {feature}: {value}\n"
        
        prompt += f"""
Please analyze these features and provide your prediction for {target_name}.
Respond with ONLY a single number: 0 for Good Health or 1 for Poor Health.
Do not provide any explanation or additional text."""

        return prompt
    
    def predict(self, features: Dict[str, Any], max_retries: int = 3) -> Optional[int]:
        """
        Make a prediction using the Ollama model.
        Args:
            features (Dict[str, Any]): Feature values for prediction.
            max_retries (int): Maximum number of retry attempts.
        Returns:
            Optional[int]: Prediction (0 or 1) or None if failed.
        """
        if not self.is_available:
            logger.error("Ollama not available, cannot make prediction")
            return None
        
        prompt = self._create_clinical_prompt(features)
        
        for attempt in range(max_retries):
            try:
                logger.debug(f"Making prediction attempt {attempt + 1} with {self.model_name}")
                
                # Prepare the request
                request_data = {
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,  # Low temperature for consistent predictions
                        "top_p": 0.9,
                        "max_tokens": 10  # Limit response length
                    }
                }
                
                # Make the request to Ollama
                result = subprocess.run(
                    ["ollama", "run", self.model_name],
                    input=json.dumps(request_data),
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
                if result.returncode != 0:
                    logger.warning(f"Ollama request failed (attempt {attempt + 1}): {result.stderr}")
                    time.sleep(1)  # Wait before retry
                    continue
                
                # Parse the response
                response_text = result.stdout.strip()
                logger.debug(f"Raw response: {response_text}")
                
                # Extract prediction (look for 0 or 1)
                prediction = self._extract_prediction(response_text)
                if prediction is not None:
                    logger.debug(f"✅ Prediction successful: {prediction}")
                    return prediction
                else:
                    logger.warning(f"Could not extract prediction from response: {response_text}")
                
            except subprocess.TimeoutExpired:
                logger.warning(f"Ollama request timed out (attempt {attempt + 1})")
                time.sleep(2)
            except Exception as e:
                logger.error(f"Error making prediction (attempt {attempt + 1}): {e}")
                time.sleep(1)
        
        logger.error(f"Failed to make prediction after {max_retries} attempts")
        return None
    
    def _extract_prediction(self, response_text: str) -> Optional[int]:
        """
        Extract prediction (0 or 1) from LLM response.
        Args:
            response_text (str): Raw response from the LLM.
        Returns:
            Optional[int]: Extracted prediction or None if not found.
        """
        # Clean the response text
        response_text = response_text.strip().lower()
        
        # Look for explicit 0 or 1
        if "0" in response_text and "1" not in response_text:
            return 0
        elif "1" in response_text and "0" not in response_text:
            return 1
        
        # Look for keywords that might indicate prediction
        if any(word in response_text for word in ["good", "healthy", "excellent", "positive"]):
            return 0
        elif any(word in response_text for word in ["poor", "unhealthy", "bad", "negative"]):
            return 1
        
        # Try to extract any number
        import re
        numbers = re.findall(r'\d+', response_text)
        if numbers:
            num = int(numbers[0])
            if num in [0, 1]:
                return num
        
        return None
    
    def batch_predict(self, features_list: List[Dict[str, Any]], batch_size: int = 10) -> List[Optional[int]]:
        """
        Make predictions for a batch of feature sets.
        Args:
            features_list (List[Dict[str, Any]]): List of feature dictionaries.
            batch_size (int): Number of predictions to process in parallel.
        Returns:
            List[Optional[int]]: List of predictions.
        """
        logger.info(f"Making batch predictions for {len(features_list)} samples")
        predictions = []
        
        for i in range(0, len(features_list), batch_size):
            batch = features_list[i:i + batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(features_list) + batch_size - 1)//batch_size}")
            
            batch_predictions = []
            for features in batch:
                pred = self.predict(features)
                batch_predictions.append(pred)
                time.sleep(0.1)  # Small delay to avoid overwhelming Ollama
            
            predictions.extend(batch_predictions)
        
        logger.info(f"Batch prediction completed. Success rate: {sum(1 for p in predictions if p is not None)}/{len(predictions)}")
        return predictions
    
    def test_connection(self) -> bool:
        """
        Test the connection to Ollama and the model.
        Returns:
            bool: True if connection is successful.
        """
        logger.info("Testing Ollama connection...")
        
        if not self.is_available:
            logger.error("Ollama not available")
            return False
        
        # Test with a simple prompt
        test_features = {
            "Age": 45,
            "BMI": 25.5,
            "Physical_Activity": 1,
            "Education": 2
        }
        
        try:
            prediction = self.predict(test_features)
            if prediction is not None:
                logger.info("✅ Ollama connection test successful")
                return True
            else:
                logger.error("❌ Ollama connection test failed: No prediction returned")
                return False
        except Exception as e:
            logger.error(f"❌ Ollama connection test failed: {e}")
            return False
