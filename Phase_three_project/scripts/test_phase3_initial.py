#!/usr/bin/env python3
"""
Test script for Phase 3: Advanced LLM Strategies

This script tests the few-shot learning and prompt engineering implementations
to ensure everything works correctly before proceeding to fine-tuning and RAG.
"""

import sys
import os
from pathlib import Path
import pandas as pd

# Add src to path for imports
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

# Change to project root directory
os.chdir(project_root)

# Import modules
from data_preparation.data_loader import Phase3DataLoader
from few_shot.few_shot_learner import FewShotEvaluator
from prompt_engineering.prompt_engineer import PromptEngineeringEvaluator
from utils.config import TOTAL_DATA_PATH, OLLAMA_MODELS, RESULTS_DIR
from utils.helpers import setup_logger

logger = setup_logger(__name__)

def test_data_loading():
    """Test Phase 3 data loading and preparation."""
    logger.info("TESTING PHASE 3 DATA LOADING")
    try:
        # Initialize data loader
        loader = Phase3DataLoader(data_path=TOTAL_DATA_PATH)
        
        # Load data
        df = loader.load_data()
        assert df is not None, "DataFrame should not be None after loading"
        print(f"✅ Data loaded successfully. Shape: {df.shape}")
        
        # Clean data
        cleaned_df = loader.clean_data()
        print(f"✅ Data cleaning completed. Final shape: {cleaned_df.shape}")
        
        # Test data splitting
        X_train, X_test, y_train, y_test = loader.split_data(cleaned_df)
        print(f"✅ Data splitting completed:")
        print(f"   Training: {len(X_train)} samples")
        print(f"   Testing: {len(X_test)} samples")
        
        # Test few-shot data preparation
        few_shot_examples, few_shot_test = loader.prepare_few_shot_data(cleaned_df)
        print(f"✅ Few-shot data prepared: {len(few_shot_examples)} examples, {len(few_shot_test)} test samples")
        
        # Test prompt engineering data preparation
        prompt_test = loader.prepare_prompt_engineering_data(cleaned_df)
        print(f"✅ Prompt engineering data prepared: {len(prompt_test)} test samples")
        
        # Get data summary
        summary = loader.get_data_summary()
        print(f"✅ Data summary generated:")
        print(f"   Total samples: {summary.get('total_samples', 0)}")
        print(f"   Features: {summary.get('features', 0)}")
        print(f"   Target distribution: {summary.get('target_distribution', {})}")
        
        return cleaned_df, few_shot_examples, few_shot_test, prompt_test
        
    except Exception as e:
        print(f"❌ Data loading test failed: {e}")
        logger.error(f"Data loading test failed: {e}")
        return None, None, None, None

def test_few_shot_learning(few_shot_examples, few_shot_test):
    """Test few-shot learning implementation."""
    logger.info("TESTING FEW-SHOT LEARNING")
    if few_shot_examples is None or few_shot_test is None:
        print("❌ Skipping few-shot test: No data available")
        return None
    
    try:
        # Initialize evaluator
        evaluator = FewShotEvaluator(model_names=OLLAMA_MODELS)
        print("✅ Few-shot evaluator initialized")
        
        # Evaluate models
        results = evaluator.evaluate_models(few_shot_examples, few_shot_test)
        print(f"✅ Few-shot evaluation completed: {list(results.keys())}")
        
        # Get best model
        best_model, best_results = evaluator.get_best_model()
        if best_model:
            print(f"✅ Best few-shot model: {best_model}")
            print(f"   F1-Score: {best_results.get('f1_score', 0):.4f}")
            print(f"   Accuracy: {best_results.get('accuracy', 0):.4f}")
            print(f"   Success Rate: {best_results.get('success_rate', 0):.2%}")
        
        # Save results
        evaluator.save_results(RESULTS_DIR / "phase3" / "few_shot")
        print("✅ Few-shot results saved")
        
        return results
        
    except Exception as e:
        print(f"❌ Few-shot learning test failed: {e}")
        logger.error(f"Few-shot learning test failed: {e}")
        return None

def test_prompt_engineering(prompt_test):
    """Test prompt engineering implementation."""
    logger.info("TESTING PROMPT ENGINEERING")
    if prompt_test is None:
        print("❌ Skipping prompt engineering test: No data available")
        return None
    
    try:
        # Initialize evaluator
        evaluator = PromptEngineeringEvaluator(model_names=OLLAMA_MODELS)
        print("✅ Prompt engineering evaluator initialized")
        
        # Evaluate all templates
        results = evaluator.evaluate_all_templates(prompt_test)
        print(f"✅ Prompt engineering evaluation completed")
        
        # Get best templates per model
        best_templates = evaluator.get_best_template_per_model()
        print("✅ Best templates per model:")
        for model_name, (template_name, results) in best_templates.items():
            print(f"   {model_name}: {template_name} (F1: {results.get('f1_score', 0):.4f})")
        
        # Get overall best
        overall_best = evaluator.get_overall_best_template()
        if overall_best:
            model_name, template_name, results = overall_best
            print(f"✅ Overall best: {model_name} with {template_name}")
            print(f"   F1-Score: {results.get('f1_score', 0):.4f}")
        
        # Save results
        evaluator.save_results(RESULTS_DIR / "phase3" / "prompt_engineering")
        print("✅ Prompt engineering results saved")
        
        return results
        
    except Exception as e:
        print(f"❌ Prompt engineering test failed: {e}")
        logger.error(f"Prompt engineering test failed: {e}")
        return None

def main():
    """Main test function."""
    print("🚀 Starting Phase 3: Advanced LLM Strategies Tests")
    print("=" * 60)
    
    # Test 1: Data Loading
    print("\n1. Testing Data Loading and Preparation...")
    cleaned_df, few_shot_examples, few_shot_test, prompt_test = test_data_loading()
    
    if cleaned_df is None:
        print("❌ Data loading failed. Exiting.")
        return
    
    # Test 2: Few-Shot Learning
    print("\n2. Testing Few-Shot Learning...")
    few_shot_results = test_few_shot_learning(few_shot_examples, few_shot_test)
    
    # Test 3: Prompt Engineering
    print("\n3. Testing Prompt Engineering...")
    prompt_results = test_prompt_engineering(prompt_test)
    
    # Summary
    print("\n" + "=" * 60)
    print("🎉 Phase 3 Initial Tests Completed!")
    
    if few_shot_results:
        print(f"✅ Few-shot learning: {len(few_shot_results)} models tested")
    
    if prompt_results:
        print(f"✅ Prompt engineering: {len(prompt_results)} models tested")
    
    print("\nNext steps:")
    print("1. Implement fine-tuning (PEFT/LoRA)")
    print("2. Implement RAG (Retrieval-Augmented Generation)")
    print("3. Comprehensive comparison with Phase 2 results")
    print("4. Statistical analysis and final report")

if __name__ == "__main__":
    main()
