#!/usr/bin/env python3
"""
Debug script to fix Few-Shot and RAG errors.
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np

# Add src to path for imports
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

# Change to project root directory
os.chdir(project_root)

from data_preparation.data_loader import Phase3DataLoader
from few_shot.few_shot_learner import FewShotEvaluator
from rag.rag_retriever import RAGEvaluator
from utils.config import TOTAL_DATA_PATH, OLLAMA_MODELS

def debug_few_shot():
    """Debug few-shot learning error."""
    print("=== DEBUGGING FEW-SHOT LEARNING ===")
    
    try:
        # Load data
        loader = Phase3DataLoader(data_path=TOTAL_DATA_PATH)
        df = loader.load_data()
        cleaned_df = loader.clean_data()
        
        # Prepare few-shot data
        few_shot_examples, few_shot_test_data = loader.prepare_few_shot_data(cleaned_df)
        
        print(f"Few-shot examples: {len(few_shot_examples)}")
        print(f"Few-shot test data shape: {few_shot_test_data.shape}")
        print(f"Few-shot test data columns: {few_shot_test_data.columns.tolist()}")
        
        # Test with small sample
        test_sample = few_shot_test_data.head(5)
        print(f"Test sample shape: {test_sample.shape}")
        
        # Initialize evaluator
        evaluator = FewShotEvaluator(['llama3.1:8b'])
        print("Evaluator initialized successfully")
        
        # Try evaluation
        results = evaluator.evaluate_models(few_shot_examples, test_sample)
        print(f"Results type: {type(results)}")
        print(f"Results keys: {results.keys() if isinstance(results, dict) else 'Not a dict'}")
        
        if isinstance(results, dict) and 'llama3.1:8b' in results:
            model_result = results['llama3.1:8b']
            print(f"Model result type: {type(model_result)}")
            if isinstance(model_result, dict):
                print(f"Model result keys: {model_result.keys()}")
            else:
                print(f"Model result value: {model_result}")
        
    except Exception as e:
        print(f"Few-shot error: {e}")
        import traceback
        traceback.print_exc()

def debug_rag():
    """Debug RAG error."""
    print("\n=== DEBUGGING RAG ===")
    
    try:
        # Load data
        loader = Phase3DataLoader(data_path=TOTAL_DATA_PATH)
        df = loader.load_data()
        cleaned_df = loader.clean_data()
        
        # Create knowledge base
        knowledge_base = cleaned_df.head(100)
        print(f"Knowledge base shape: {knowledge_base.shape}")
        print(f"Knowledge base columns: {knowledge_base.columns.tolist()}")
        
        # Create test data
        test_df = cleaned_df.tail(10)
        print(f"Test data shape: {test_df.shape}")
        print(f"Test data columns: {test_df.columns.tolist()}")
        
        # Initialize evaluator
        evaluator = RAGEvaluator(['llama3.1:8b'])
        print("RAG evaluator initialized successfully")
        
        # Try evaluation
        results = evaluator.evaluate_all_models(test_df, knowledge_base)
        print(f"RAG results type: {type(results)}")
        print(f"RAG results keys: {results.keys() if isinstance(results, dict) else 'Not a dict'}")
        
    except Exception as e:
        print(f"RAG error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_few_shot()
    debug_rag()
