#!/usr/bin/env python3
"""
Complete Phase 3 evaluation with all strategies.
Uses small sample sizes for efficient testing.
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, Any, List
import json
import time

# Add src to path for imports
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

# Change to project root directory
os.chdir(project_root)

from data_preparation.data_loader import Phase3DataLoader
from few_shot.few_shot_learner import FewShotEvaluator
from prompt_engineering.prompt_engineer import PromptEngineeringEvaluator
from fine_tuning.lora_trainer import FineTuningEvaluator
from rag.rag_retriever import RAGEvaluator
from utils.logger import get_logger
from utils.config import (
    TOTAL_DATA_PATH, OLLAMA_MODELS, FEW_SHOT_SAMPLE_SIZE, 
    PROMPT_TEMPLATES, RAG_KNOWLEDGE_BASE_SIZE
)

logger = get_logger(__name__)

def load_and_prepare_data(sample_size: int = 30) -> tuple:
    """
    Load and prepare data for Phase 3 evaluation.
    
    Args:
        sample_size: Size of test sample
        
    Returns:
        Tuple of (cleaned_df, X_test, y_test, few_shot_examples, few_shot_test_data)
    """
    logger.info("Loading and preparing data for Phase 3 evaluation...")
    
    # Load data
    loader = Phase3DataLoader(data_path=TOTAL_DATA_PATH)
    df = loader.load_data()
    if df is None:
        raise ValueError("Failed to load data")
    
    cleaned_df = loader.clean_data()
    if cleaned_df is None:
        raise ValueError("Failed to clean data")
    
    # Split data
    X_train, X_test, y_train, y_test = loader.split_data(cleaned_df)
    
    # Prepare few-shot data
    few_shot_examples, few_shot_test_data = loader.prepare_few_shot_data(cleaned_df)
    
    # Sample test data if needed
    if len(X_test) > sample_size:
        # Stratified sampling for test set
        temp_df = pd.DataFrame(X_test.copy())
        temp_df['Health_status'] = y_test.copy()
        
        stratify_proportions = temp_df['Health_status'].value_counts(normalize=True)
        samples_per_class = (stratify_proportions * sample_size).round().astype(int)
        
        sampled_indices = []
        for class_label, num_samples in samples_per_class.items():
            class_indices = temp_df[temp_df['Health_status'] == class_label].index.tolist()
            num_samples = min(num_samples, len(class_indices))
            sampled_indices.extend(np.random.choice(class_indices, num_samples, replace=False))
        
        X_sample = X_test.loc[sampled_indices]
        y_sample = y_test.loc[sampled_indices]
    else:
        X_sample = X_test
        y_sample = y_test
    
    logger.info(f"Data prepared: {len(X_sample)} test samples, {len(few_shot_examples)} few-shot examples")
    return cleaned_df, X_sample, y_sample, few_shot_examples, few_shot_test_data

def test_improved_prompt_engineering(X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
    """
    Test improved prompt engineering with new templates.
    
    Args:
        X_test: Test features
        y_test: Test labels
        
    Returns:
        Prompt engineering results
    """
    logger.info("Testing improved prompt engineering...")
    
    results = {}
    
    for model_name in OLLAMA_MODELS:
        logger.info(f"Testing prompt engineering with {model_name}")
        
        try:
            evaluator = PromptEngineeringEvaluator([model_name])
            
            # Use small sample for testing
            sample_size = min(len(X_test), 20)
            X_sample = X_test.sample(sample_size, random_state=42)
            y_sample = y_test.loc[X_sample.index]
            
            # Create test DataFrame with labels
            test_df = X_sample.copy()
            test_df['Health_status'] = y_sample
            
            model_results = evaluator.evaluate_all_templates(test_df)
            results[model_name] = model_results[model_name]
            
            logger.info(f"Prompt engineering evaluation for {model_name} completed.")
            for template_name, metrics in model_results[model_name].items():
                logger.info(f"Template '{template_name}' Accuracy: {metrics.get('accuracy', 'N/A'):.4f}")
            
        except Exception as e:
            logger.error(f"Error testing prompt engineering for {model_name}: {e}")
            results[model_name] = {'error': str(e)}
    
    return results

def test_improved_few_shot(few_shot_examples: List[Dict], few_shot_test_data: pd.DataFrame) -> Dict[str, Any]:
    """
    Test improved few-shot learning with textual features.
    
    Args:
        few_shot_examples: Few-shot examples
        few_shot_test_data: Few-shot test data
        
    Returns:
        Few-shot learning results
    """
    logger.info("Testing improved few-shot learning...")
    
    results = {}
    
    for model_name in OLLAMA_MODELS:
        logger.info(f"Testing few-shot learning with {model_name}")
        
        try:
            learner = FewShotEvaluator([model_name])
            
            # Use small sample for testing
            sample_size = min(len(few_shot_test_data), 20)
            test_sample = few_shot_test_data.sample(sample_size, random_state=42)
            
            model_results = learner.evaluate_models(few_shot_examples, test_sample)
            results[model_name] = model_results[model_name]
            
            logger.info(f"Few-shot evaluation for {model_name} completed. Accuracy: {model_results[model_name].get('accuracy', 'N/A'):.4f}")
            
        except Exception as e:
            logger.error(f"Error testing few-shot learning for {model_name}: {e}")
            results[model_name] = {'error': str(e)}
    
    return results

def test_fine_tuning(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
    """
    Test LoRA-based fine-tuning (simplified version).
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        
    Returns:
        Fine-tuning results
    """
    logger.info("Testing LoRA-based fine-tuning...")
    
    # Note: This is a simplified test - actual fine-tuning would take much longer
    # For now, we'll simulate the results or skip if dependencies are not available
    
    try:
        from fine_tuning.lora_trainer import FineTuningEvaluator
        
        # Prepare training data
        train_data = []
        for idx, row in X_train.iterrows():
            train_data.append({
                'features': row.to_dict(),
                'label': int(y_train.loc[idx])
            })
        
        # Prepare test data
        test_data = []
        for idx, row in X_test.iterrows():
            test_data.append({
                'features': row.to_dict(),
                'label': int(y_test.loc[idx])
            })
        
        # Use small sample for testing
        train_sample = train_data[:50]  # Small training sample
        test_sample = test_data[:20]    # Small test sample
        
        evaluator = FineTuningEvaluator(OLLAMA_MODELS)
        
        # Train models (this would normally take much longer)
        logger.info("Training LoRA models (simplified test)...")
        training_results = evaluator.train_all_models(train_sample, test_sample)
        
        # Evaluate models
        evaluation_results = evaluator.evaluate_all_models(test_sample)
        
        return {
            'training_results': training_results,
            'evaluation_results': evaluation_results
        }
        
    except ImportError as e:
        logger.warning(f"Fine-tuning dependencies not available: {e}")
        return {'error': 'Fine-tuning dependencies not available'}
    except Exception as e:
        logger.error(f"Error testing fine-tuning: {e}")
        return {'error': str(e)}

def test_rag(X_test: pd.DataFrame, y_test: pd.Series, knowledge_base: pd.DataFrame) -> Dict[str, Any]:
    """
    Test RAG with similarity-based retrieval.
    
    Args:
        X_test: Test features
        y_test: Test labels
        knowledge_base: Knowledge base for retrieval
        
    Returns:
        RAG results
    """
    logger.info("Testing RAG with similarity-based retrieval...")
    
    try:
        evaluator = RAGEvaluator(OLLAMA_MODELS)
        
        # Use small sample for testing
        sample_size = min(len(X_test), 20)
        X_sample = X_test.sample(sample_size, random_state=42)
        y_sample = y_test.loc[X_sample.index]
        
        # Create test DataFrame
        test_df = X_sample.copy()
        test_df['Health_status'] = y_sample
        
        # Evaluate models
        results = evaluator.evaluate_all_models(test_df, knowledge_base)
        
        return results
        
    except Exception as e:
        logger.error(f"Error testing RAG: {e}")
        return {'error': str(e)}

def generate_comprehensive_report(all_results: Dict[str, Any]) -> str:
    """
    Generate comprehensive comparison report.
    
    Args:
        all_results: Results from all strategies
        
    Returns:
        Report text
    """
    report_parts = [
        "# Phase 3: Advanced LLM Strategies - Comprehensive Results",
        "",
        f"Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Summary",
        ""
    ]
    
    # Extract key metrics for each strategy
    strategies = ['prompt_engineering', 'few_shot', 'fine_tuning', 'rag']
    strategy_names = ['Prompt Engineering', 'Few-Shot Learning', 'Fine-Tuning (LoRA)', 'RAG']
    
    summary_table = []
    summary_table.append("| Strategy | Model | Accuracy | F1-Score | Success Rate |")
    summary_table.append("|----------|-------|----------|----------|--------------|")
    
    for strategy, strategy_name in zip(strategies, strategy_names):
        if strategy in all_results and 'error' not in all_results[strategy]:
            strategy_results = all_results[strategy]
            
            for model_name in OLLAMA_MODELS:
                if model_name in strategy_results and 'error' not in strategy_results[model_name]:
                    model_results = strategy_results[model_name]
                    
                    if isinstance(model_results, dict) and 'accuracy' in model_results:
                        accuracy = model_results.get('accuracy', 0.0)
                        f1_score = model_results.get('f1_score', 0.0)
                        success_rate = model_results.get('success_rate', 0.0)
                        
                        summary_table.append(f"| {strategy_name} | {model_name} | {accuracy:.3f} | {f1_score:.3f} | {success_rate:.3f} |")
    
    report_parts.extend(summary_table)
    report_parts.extend(["", "## Detailed Results", ""])
    
    # Add detailed results for each strategy
    for strategy, strategy_name in zip(strategies, strategy_names):
        if strategy in all_results:
            report_parts.append(f"### {strategy_name}")
            report_parts.append("")
            
            if 'error' in all_results[strategy]:
                report_parts.append(f"Error: {all_results[strategy]['error']}")
            else:
                strategy_results = all_results[strategy]
                
                for model_name in OLLAMA_MODELS:
                    if model_name in strategy_results:
                        report_parts.append(f"#### {model_name}")
                        
                        if 'error' in strategy_results[model_name]:
                            report_parts.append(f"Error: {strategy_results[model_name]['error']}")
                        else:
                            model_results = strategy_results[model_name]
                            
                            if isinstance(model_results, dict):
                                for metric, value in model_results.items():
                                    if isinstance(value, (int, float)):
                                        report_parts.append(f"- {metric}: {value:.4f}")
                                    else:
                                        report_parts.append(f"- {metric}: {value}")
                        
                        report_parts.append("")
            
            report_parts.append("")
    
    return "\n".join(report_parts)

def main():
    """
    Main function to run complete Phase 3 evaluation.
    """
    logger.info("Starting Phase 3: Advanced LLM Strategies - Complete Evaluation")
    logger.info("="*60)
    
    try:
        # Load and prepare data
        cleaned_df, X_test, y_test, few_shot_examples, few_shot_test_data = load_and_prepare_data()
        
        # Create knowledge base for RAG
        knowledge_base = cleaned_df.sample(min(len(cleaned_df), RAG_KNOWLEDGE_BASE_SIZE), random_state=42)
        
        all_results = {}
        
        # Test 1: Improved Prompt Engineering
        logger.info("\n" + "="*50)
        logger.info("Testing Improved Prompt Engineering")
        logger.info("="*50)
        all_results['prompt_engineering'] = test_improved_prompt_engineering(X_test, y_test)
        
        # Test 2: Improved Few-Shot Learning
        logger.info("\n" + "="*50)
        logger.info("Testing Improved Few-Shot Learning")
        logger.info("="*50)
        all_results['few_shot'] = test_improved_few_shot(few_shot_examples, few_shot_test_data)
        
        # Test 3: Fine-Tuning (LoRA)
        logger.info("\n" + "="*50)
        logger.info("Testing Fine-Tuning (LoRA)")
        logger.info("="*50)
        all_results['fine_tuning'] = test_fine_tuning(
            cleaned_df.drop(columns=['Health_status']), 
            cleaned_df['Health_status'], 
            X_test, 
            y_test
        )
        
        # Test 4: RAG
        logger.info("\n" + "="*50)
        logger.info("Testing RAG")
        logger.info("="*50)
        all_results['rag'] = test_rag(X_test, y_test, knowledge_base)
        
        # Generate comprehensive report
        logger.info("\n" + "="*50)
        logger.info("Generating Comprehensive Report")
        logger.info("="*50)
        
        report = generate_comprehensive_report(all_results)
        
        # Save results
        results_dir = Path("results/phase3_complete")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Save detailed results
        with open(results_dir / "detailed_results.json", "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        
        # Save report
        with open(results_dir / "comprehensive_report.txt", "w") as f:
            f.write(report)
        
        logger.info(f"Results saved to: {results_dir}")
        logger.info("\n" + "="*60)
        logger.info("Phase 3 Complete Evaluation Finished!")
        logger.info("="*60)
        
        # Print summary
        print("\n" + report)
        
    except Exception as e:
        logger.error(f"Error in main evaluation: {e}")
        raise

if __name__ == "__main__":
    main()
