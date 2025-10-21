"""
Utility functions for Phase 3: Advanced LLM Strategies

This module provides helper functions for few-shot learning, fine-tuning,
prompt engineering, and RAG implementations.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import json
import logging
from datetime import datetime

from .config import (
    FEATURE_COLUMNS, TARGET_COLUMN, ID_COLUMN, PROMPT_TEMPLATES,
    FEW_SHOT_EXAMPLES, RAG_TOP_K, RAG_SIMILARITY_THRESHOLD
)

def setup_logger(name: str) -> logging.Logger:
    """Set up logger for Phase 3 modules."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger

def format_features_for_prompt(features: Dict[str, Any]) -> str:
    """
    Format patient features into a readable string for LLM prompts.
    
    Args:
        features: Dictionary of patient features
        
    Returns:
        Formatted string of features
    """
    # Create readable feature descriptions
    feature_descriptions = {
        'Gender': 'Gender (0=Female, 1=Male)',
        'Age': 'Age (years)',
        'US_born': 'US Born (0=No, 1=Yes)',
        'HH_Income': 'Household Income (1-5 scale)',
        'HH_size': 'Household Size',
        'Edu_level': 'Education Level (1-5 scale)',
        'PA_mean': 'Physical Activity Mean',
        'PA_std': 'Physical Activity Std Dev',
        'WW_mean': 'Walking/Work Mean',
        'WW_std': 'Walking/Work Std Dev',
        'SW_mean': 'Sedentary Work Mean',
        'SW_std': 'Sedentary Work Std Dev',
        'NW_mean': 'Non-Work Mean',
        'NW_std': 'Non-Work Std Dev',
        'VWA': 'Vigorous Work Activity',
        'MWA': 'Moderate Work Activity',
        'W/B': 'Work/Balance Ratio',
        'VRC': 'Vigorous Recreation',
        'MRC': 'Moderate Recreation'
    }
    
    formatted_features = []
    for key, value in features.items():
        if key in feature_descriptions:
            formatted_features.append(f"{feature_descriptions[key]}: {value}")
    
    return ", ".join(formatted_features)

def convert_features_to_text(features: Dict[str, Any]) -> str:
    """Convert numerical features to textual description."""
    age = int(features.get('Age', 0))
    gender = "male" if int(features.get('Gender', 0)) == 1 else "female"
    
    pa_mean = features.get('PA_mean', 0)
    if pa_mean > 100:
        activity = "high physical activity"
    elif pa_mean > 50:
        activity = "moderate physical activity"
    else:
        activity = "low physical activity"
    
    income_map = {1: "very low", 2: "low", 3: "medium", 4: "high", 5: "very high"}
    income = income_map.get(int(features.get('HH_Income', 3)), "medium")
    
    edu_map = {1: "basic", 2: "high school", 3: "some college", 4: "bachelor's", 5: "graduate"}
    education = edu_map.get(int(features.get('Edu_level', 3)), "some college")
    
    return f"{age}-year-old {gender} with {activity}, {income} income, {education} education"

def create_few_shot_examples(df: pd.DataFrame, n_examples: int = FEW_SHOT_EXAMPLES) -> List[Dict[str, Any]]:
    """
    Create few-shot learning examples from the dataset.
    
    Args:
        df: DataFrame with patient data
        n_examples: Number of examples to create
        
    Returns:
        List of example dictionaries
    """
    logger = setup_logger(__name__)
    
    # Ensure balanced examples (equal good/poor health)
    good_health = df[df[TARGET_COLUMN] == 0].sample(n_examples // 2, random_state=42)
    poor_health = df[df[TARGET_COLUMN] == 1].sample(n_examples // 2, random_state=42)
    
    examples = []
    
    for _, row in pd.concat([good_health, poor_health]).iterrows():
        features = row[FEATURE_COLUMNS].to_dict()
        example = {
            'features': features,  # Store as dictionary, not formatted string
            'label': int(row[TARGET_COLUMN]),
            'label_text': 'Good Health' if row[TARGET_COLUMN] == 0 else 'Poor Health'
        }
        examples.append(example)
    
    logger.info(f"Created {len(examples)} few-shot examples")
    return examples

def build_few_shot_prompt(examples: List[Dict[str, Any]], test_features: Dict[str, Any], test_features_text: Optional[str] = None) -> str:
    """
    Build a few-shot learning prompt with examples.
    
    Args:
        examples: List of example dictionaries
        test_features: Features for the test case
        test_features_text: Optional textual description of test features
        
    Returns:
        Complete few-shot prompt
    """
    prompt_parts = [
        "Research classification task. Here are examples:\n"
    ]
    
    # Add examples with textual descriptions
    for i, example in enumerate(examples, 1):
        example_text = convert_features_to_text(example['features'])
        prompt_parts.append(
            f"Example {i}: {example_text} → {example['label']}"
        )
    
    # Add test case
    if test_features_text:
        prompt_parts.append(f"\nClassify: {test_features_text}")
    else:
        test_features_str = format_features_for_prompt(test_features)
        prompt_parts.append(f"\nClassify: {test_features_str}")
    
    prompt_parts.append("Answer (0 or 1):")
    
    return "\n".join(prompt_parts)

def create_rag_knowledge_base(df: pd.DataFrame, size: int = 1000) -> pd.DataFrame:
    """
    Create a knowledge base for RAG retrieval.
    
    Args:
        df: DataFrame with patient data
        size: Size of knowledge base
        
    Returns:
        Knowledge base DataFrame
    """
    logger = setup_logger(__name__)
    
    # Sample balanced knowledge base
    good_health = df[df[TARGET_COLUMN] == 0].sample(size // 2, random_state=42)
    poor_health = df[df[TARGET_COLUMN] == 1].sample(size // 2, random_state=42)
    
    knowledge_base = pd.concat([good_health, poor_health]).reset_index(drop=True)
    
    logger.info(f"Created RAG knowledge base with {len(knowledge_base)} cases")
    return knowledge_base

def find_similar_cases(query_features: Dict[str, Any], knowledge_base: pd.DataFrame, 
                      top_k: int = RAG_TOP_K) -> List[Dict[str, Any]]:
    """
    Find similar cases in the knowledge base using cosine similarity.
    
    Args:
        query_features: Query patient features
        knowledge_base: Knowledge base DataFrame
        top_k: Number of similar cases to return
        
    Returns:
        List of similar cases
    """
    from sklearn.metrics.pairwise import cosine_similarity
    
    # Convert features to vectors
    query_vector = np.array([query_features[col] for col in FEATURE_COLUMNS]).reshape(1, -1)
    kb_vectors = knowledge_base[FEATURE_COLUMNS].values
    
    # Calculate similarities
    similarities = cosine_similarity(query_vector, kb_vectors)[0]
    
    # Get top-k similar cases
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    
    similar_cases = []
    for idx in top_indices:
        case = knowledge_base.iloc[idx]
        similar_cases.append({
            'features': case[FEATURE_COLUMNS].to_dict(),
            'label': int(case[TARGET_COLUMN]),
            'similarity': similarities[idx],
            'formatted_features': format_features_for_prompt(case[FEATURE_COLUMNS].to_dict())
        })
    
    return similar_cases

def build_rag_prompt(similar_cases: List[Dict[str, Any]], test_features: Dict[str, Any]) -> str:
    """
    Build a RAG prompt using similar cases.
    
    Args:
        similar_cases: List of similar cases
        test_features: Features for the test case
        
    Returns:
        Complete RAG prompt
    """
    prompt_parts = [
        "You are a medical expert. Based on similar patient cases, predict the health status.\n"
    ]
    
    # Add similar cases
    prompt_parts.append("Similar cases:")
    for i, case in enumerate(similar_cases, 1):
        prompt_parts.append(
            f"Case {i}: {case['formatted_features']} → Health Status: {case['label']} (similarity: {case['similarity']:.3f})"
        )
    
    # Add test case
    test_features_str = format_features_for_prompt(test_features)
    prompt_parts.append(f"\nNow predict for this patient: {test_features_str}. Respond with only '0' or '1'.")
    
    return "\n".join(prompt_parts)

def save_results(results: Dict[str, Any], filepath: Path) -> None:
    """
    Save results dictionary to JSON file.
    
    Args:
        results: Dictionary containing results to save
        filepath: Path where to save the results
    """
    logger = setup_logger(__name__)
    
    try:
        # Ensure directory exists
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert numpy types to Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        # Convert results
        json_results = json.loads(json.dumps(results, default=convert_numpy))
        
        # Save to file
        with open(filepath, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        logger.info(f"Results saved to {filepath}")
        
    except Exception as e:
        logger.error(f"Failed to save results to {filepath}: {e}")

def load_results(filepath: Path) -> Dict[str, Any]:
    """
    Load results from JSON file.
    
    Args:
        filepath: Path to the results file
        
    Returns:
        Dictionary containing loaded results
    """
    logger = setup_logger(__name__)
    
    try:
        with open(filepath, 'r') as f:
            results = json.load(f)
        
        logger.info(f"Results loaded from {filepath}")
        return results
        
    except Exception as e:
        logger.error(f"Failed to load results from {filepath}: {e}")
        return {}

def calculate_performance_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
    """
    Calculate comprehensive performance metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        Dictionary of performance metrics
    """
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, matthews_corrcoef, confusion_matrix
    )
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0),
        'mcc': matthews_corrcoef(y_true, y_pred),
        'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
    }
    
    return metrics

def create_strategy_summary(results: Dict[str, Dict[str, Any]]) -> str:
    """
    Create a summary of all strategy results.
    
    Args:
        results: Dictionary containing results for each strategy
        
    Returns:
        Formatted summary string
    """
    summary_lines = [
        "PHASE 3: ADVANCED LLM STRATEGIES SUMMARY",
        "=" * 50,
        ""
    ]
    
    for strategy, metrics in results.items():
        summary_lines.append(f"Strategy: {strategy.upper()}")
        summary_lines.append(f"  Accuracy: {metrics.get('accuracy', 0):.4f}")
        summary_lines.append(f"  Precision: {metrics.get('precision', 0):.4f}")
        summary_lines.append(f"  Recall: {metrics.get('recall', 0):.4f}")
        summary_lines.append(f"  F1-Score: {metrics.get('f1_score', 0):.4f}")
        summary_lines.append(f"  MCC: {metrics.get('mcc', 0):.4f}")
        summary_lines.append("")
    
    # Find best strategy
    if results:
        best_strategy = max(results.keys(), key=lambda k: results[k].get('f1_score', 0))
        best_f1 = results[best_strategy].get('f1_score', 0)
        
        summary_lines.append(f"BEST STRATEGY: {best_strategy.upper()}")
        summary_lines.append(f"Best F1-Score: {best_f1:.4f}")
    
    return "\n".join(summary_lines)
