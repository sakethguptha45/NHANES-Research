"""
RAG implementation using similarity-based retrieval.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import requests
import json
import time
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

from utils.config import (
    OLLAMA_BASE_URL, OLLAMA_MODELS, LLM_TEMPERATURE, LLM_MAX_TOKENS, 
    LLM_TOP_P, LLM_BATCH_SIZE, FEATURE_COLUMNS, TARGET_COLUMN,
    RAG_TOP_K, RAG_SIMILARITY_THRESHOLD, RAG_KNOWLEDGE_BASE_SIZE
)
from utils.helpers import setup_logger, convert_features_to_text

logger = setup_logger(__name__)

class RAGRetriever:
    """
    Retrieval component for RAG system.
    """
    
    def __init__(self, knowledge_base: pd.DataFrame):
        """
        Initialize RAG retriever with knowledge base.
        
        Args:
            knowledge_base: DataFrame containing cases for retrieval
        """
        self.knowledge_base = knowledge_base
        self.scaler = StandardScaler()
        
        # Prepare feature vectors for similarity computation
        self.feature_vectors = self._prepare_feature_vectors()
        
        logger.info(f"RAGRetriever initialized with {len(knowledge_base)} cases")
    
    def _prepare_feature_vectors(self) -> np.ndarray:
        """
        Prepare normalized feature vectors for similarity computation.
        
        Returns:
            Normalized feature vectors
        """
        # Extract feature columns
        features = self.knowledge_base[FEATURE_COLUMNS].values
        
        # Normalize features
        normalized_features = self.scaler.fit_transform(features)
        
        return normalized_features
    
    def retrieve_similar_cases(self, query_features: Dict[str, Any], top_k: int = RAG_TOP_K) -> List[Dict]:
        """
        Retrieve k most similar cases from knowledge base.
        
        Args:
            query_features: Query features
            top_k: Number of similar cases to retrieve
            
        Returns:
            List of similar cases with metadata
        """
        # Convert query to vector
        query_vector = np.array([query_features[col] for col in FEATURE_COLUMNS])
        query_vector_normalized = self.scaler.transform([query_vector])
        
        # Compute similarities
        similarities = cosine_similarity(query_vector_normalized, self.feature_vectors)[0]
        
        # Get top-k most similar cases
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        similar_cases = []
        for idx in top_indices:
            case = self.knowledge_base.iloc[idx]
            similarity_score = similarities[idx]
            
            # Only include cases above similarity threshold
            if similarity_score >= RAG_SIMILARITY_THRESHOLD:
                similar_cases.append({
                    'features': case[FEATURE_COLUMNS].to_dict(),
                    'features_text': convert_features_to_text(case[FEATURE_COLUMNS].to_dict()),
                    'label': int(case[TARGET_COLUMN]),
                    'similarity': similarity_score,
                    'index': idx
                })
        
        logger.info(f"Retrieved {len(similar_cases)} similar cases (threshold: {RAG_SIMILARITY_THRESHOLD})")
        return similar_cases
    
    def build_rag_prompt(self, similar_cases: List[Dict], test_features: Dict[str, Any]) -> str:
        """
        Build prompt with retrieved context.
        
        Args:
            similar_cases: Retrieved similar cases
            test_features: Test case features
            
        Returns:
            Complete RAG prompt
        """
        prompt_parts = [
            "Research classification using similar examples:",
            ""
        ]
        
        # Add similar cases as examples
        for i, case in enumerate(similar_cases, 1):
            prompt_parts.append(f"{i}. {case['features_text']} → {case['label']} (similarity: {case['similarity']:.3f})")
        
        # Add test case
        test_text = convert_features_to_text(test_features)
        prompt_parts.append(f"\nClassify: {test_text}")
        prompt_parts.append("Answer (0 or 1):")
        
        return "\n".join(prompt_parts)

class RAGLLMClient:
    """
    LLM client for RAG-based predictions.
    """
    
    def __init__(self, model_name: str):
        """
        Initialize RAG LLM client.
        
        Args:
            model_name: Name of the Ollama model to use
        """
        self.model_name = model_name
        self.base_url = OLLAMA_BASE_URL
        logger.info(f"RAGLLMClient initialized with model: {self.model_name}")
    
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
            
            # More lenient parsing
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
    
    def predict_with_rag(self, retriever: RAGRetriever, test_features: Dict[str, Any]) -> Optional[int]:
        """
        Make prediction using RAG.
        
        Args:
            retriever: RAG retriever instance
            test_features: Test case features
            
        Returns:
            Predicted label (0 or 1) or None if prediction fails
        """
        # Retrieve similar cases
        similar_cases = retriever.retrieve_similar_cases(test_features)
        
        if not similar_cases:
            logger.warning("No similar cases found for RAG prediction")
            return None
        
        # Build RAG prompt
        prompt = retriever.build_rag_prompt(similar_cases, test_features)
        
        # Get LLM response
        response = self._call_ollama_api(prompt)
        
        # Parse response
        prediction = self._parse_response(response)
        
        return prediction
    
    def batch_predict_with_rag(self, retriever: RAGRetriever, test_cases: List[Dict[str, Any]]) -> List[Optional[int]]:
        """
        Make batch predictions using RAG.
        
        Args:
            retriever: RAG retriever instance
            test_cases: List of test cases
            
        Returns:
            List of predictions
        """
        predictions = []
        
        logger.info(f"Making RAG predictions for {len(test_cases)} test cases")
        
        for i, test_case in enumerate(test_cases):
            if i % 10 == 0:
                logger.info(f"Processing test case {i+1}/{len(test_cases)}")
            
            prediction = self.predict_with_rag(retriever, test_case['features'])
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
        logger.info(f"RAG batch prediction completed. Success rate: {success_rate:.2%}")
        
        return predictions

class RAGEvaluator:
    """
    Evaluator for RAG strategy.
    """
    
    def __init__(self, model_names: List[str]):
        """
        Initialize RAG evaluator.
        
        Args:
            model_names: List of Ollama model names to evaluate
        """
        self.model_names = model_names
        self.clients: Dict[str, RAGLLMClient] = {name: RAGLLMClient(name) for name in model_names}
        self.results: Dict[str, Dict[str, Any]] = {}
        
        logger.info(f"RAGEvaluator initialized with models: {model_names}")
    
    def create_knowledge_base(self, df: pd.DataFrame, size: int = RAG_KNOWLEDGE_BASE_SIZE) -> pd.DataFrame:
        """
        Create knowledge base for RAG retrieval.
        
        Args:
            df: Source DataFrame
            size: Size of knowledge base
            
        Returns:
            Knowledge base DataFrame
        """
        if len(df) <= size:
            knowledge_base = df.copy()
        else:
            # Stratified sampling to maintain class balance
            knowledge_base = df.groupby(TARGET_COLUMN).apply(
                lambda x: x.sample(min(len(x), size // 2), random_state=42)
            ).reset_index(drop=True)
        
        logger.info(f"Created knowledge base with {len(knowledge_base)} cases")
        logger.info(f"Class distribution: {knowledge_base[TARGET_COLUMN].value_counts().to_dict()}")
        
        return knowledge_base
    
    def evaluate_all_models(self, test_df: pd.DataFrame, knowledge_base: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """
        Evaluate all models with RAG.
        
        Args:
            test_df: Test DataFrame
            knowledge_base: Knowledge base for retrieval
            
        Returns:
            Evaluation results for all models
        """
        logger.info(f"Evaluating {len(self.model_names)} models with RAG")
        
        # Create retriever
        retriever = RAGRetriever(knowledge_base)
        
        # Prepare test cases
        test_cases = []
        for _, row in test_df.iterrows():
            test_cases.append({
                'features': row[FEATURE_COLUMNS].to_dict(),
                'label': int(row[TARGET_COLUMN])
            })
        
        evaluation_results = {}
        
        for model_name, client in self.clients.items():
            logger.info(f"Evaluating model: {model_name}")
            
            # Make predictions
            predictions = client.batch_predict_with_rag(retriever, test_cases)
            
            # Calculate metrics
            true_labels = [case['label'] for case in test_cases]
            valid_predictions = [p for p in predictions if p is not None]
            valid_indices = [i for i, p in enumerate(predictions) if p is not None]
            valid_labels = [true_labels[i] for i in valid_indices]
            
            if len(valid_predictions) > 0:
                from utils.helpers import calculate_performance_metrics
                metrics = calculate_performance_metrics(valid_labels, valid_predictions)
                metrics['success_rate'] = len(valid_predictions) / len(predictions)
                metrics['total_samples'] = len(test_cases)
                metrics['valid_samples'] = len(valid_predictions)
            else:
                metrics = {
                    'accuracy': 0.0,
                    'precision': 0.0,
                    'recall': 0.0,
                    'f1_score': 0.0,
                    'success_rate': 0.0,
                    'total_samples': len(test_cases),
                    'valid_samples': 0
                }
            
            evaluation_results[model_name] = metrics
            logger.info(f"RAG evaluation completed for {model_name}: {metrics}")
        
        self.results = evaluation_results
        return evaluation_results
