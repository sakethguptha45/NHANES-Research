#!/usr/bin/env python3
"""
Test script for LLM evaluation modules.

This script tests the functionality of LLM clients and evaluators
using Ollama models (Llama 3.1 and Mistral).
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
from data_preparation.data_loader import DataLoader
from data_preparation.data_splitter import DataSplitter
from llm_evaluation.llm_client import OllamaClient
from llm_evaluation.llm_evaluator import LLMEvaluator
from utils.logger import get_logger
from utils.config import TOTAL_DATA_PATH, PROCESSED_DATA_DIR, RESULTS_DIR, TARGET_COLUMN, FEATURE_COLUMNS
# from utils.helpers import create_directory  # Not needed - using Path.mkdir() directly

logger = get_logger(__name__)

def test_ollama_setup():
    """Test Ollama installation and model availability."""
    logger.info("TESTING OLLAMA SETUP")
    try:
        # Test Ollama installation
        import subprocess
        result = subprocess.run(["ollama", "--version"], capture_output=True, text=True)
        if result.returncode != 0:
            print("❌ Ollama not found. Please install from: https://ollama.ai/download")
            return False
        
        print("✅ Ollama is installed")
        
        # Check available models
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
        if result.returncode != 0:
            print("❌ Failed to list Ollama models")
            return False
        
        print("Available models:")
        print(result.stdout)
        
        # Check for required models
        required_models = ["llama3.1:8b", "mistral:7b"]
        available_models = result.stdout.lower()
        
        for model in required_models:
            if model.lower() in available_models:
                print(f"✅ {model} is available")
            else:
                print(f"❌ {model} not found. Run: ollama pull {model}")
        
        return True
        
    except Exception as e:
        print(f"❌ Ollama setup test failed: {e}")
        logger.error(f"Ollama setup test failed: {e}")
        return False

def test_llm_client():
    """Test LLM client functionality."""
    logger.info("TESTING LLM CLIENT")
    try:
        # Test Llama 3.1 client
        llama_client = OllamaClient("llama3.1:8b")
        print(f"✅ Llama 3.1 client initialized: {llama_client.is_available}")
        
        if llama_client.is_available:
            # Test connection
            connection_test = llama_client.test_connection()
            print(f"✅ Llama 3.1 connection test: {connection_test}")
            
            if connection_test:
                # Test single prediction
                test_features = {
                    "Age": 45,
                    "BMI": 25.5,
                    "Physical_Activity": 1,
                    "Education": 2,
                    "Income": 3
                }
                
                prediction = llama_client.predict(test_features)
                print(f"✅ Llama 3.1 prediction test: {prediction}")
        
        # Test Mistral client
        mistral_client = OllamaClient("mistral:7b")
        print(f"✅ Mistral client initialized: {mistral_client.is_available}")
        
        if mistral_client.is_available:
            # Test connection
            connection_test = mistral_client.test_connection()
            print(f"✅ Mistral connection test: {connection_test}")
        
        return llama_client, mistral_client
        
    except Exception as e:
        print(f"❌ LLM client test failed: {e}")
        logger.error(f"LLM client test failed: {e}")
        return None, None

def test_llm_evaluator(llama_client, mistral_client):
    """Test LLM evaluator functionality."""
    logger.info("TESTING LLM EVALUATOR")
    try:
        # Load and prepare test data
        loader = DataLoader(data_path=TOTAL_DATA_PATH)
        df = loader.load_data()
        cleaned_df = loader.clean_data()
        
        splitter = DataSplitter()
        X_train, X_test, y_train, y_test = splitter.split_data(cleaned_df)
        
        print(f"✅ Test data prepared: {X_test.shape[0]} samples")
        
        # Initialize evaluator
        evaluator = LLMEvaluator(["llama3.1:8b", "mistral:7b"])
        print("✅ LLM Evaluator initialized")
        
        # Test connections
        connection_status = evaluator.test_all_connections()
        print(f"✅ Connection status: {connection_status}")
        
        # Evaluate models on small sample for testing
        sample_size = 10  # Small sample for testing
        print(f"⚠️  Evaluating on small sample ({sample_size} samples) for testing")
        
        evaluation_results = evaluator.evaluate_all_models(X_test, y_test, sample_size)
        print(f"✅ Evaluation completed: {list(evaluation_results.keys())}")
        
        # Compare models
        comparison_results = evaluator.compare_models()
        print("✅ Model comparison completed")
        
        # Get summary
        summary = evaluator.get_evaluation_summary()
        print("\nLLM Evaluation Summary:")
        print(summary)
        
        # Save results
        evaluator.save_evaluation_results(RESULTS_DIR / "llm_results")
        evaluator.create_evaluation_plots(RESULTS_DIR / "visualizations")
        
        return evaluator
        
    except Exception as e:
        print(f"❌ LLM evaluator test failed: {e}")
        logger.error(f"LLM evaluator test failed: {e}")
        return None

def main():
    """Main test function."""
    print("🚀 Starting LLM Evaluation Tests")
    print("=" * 50)
    
    # Test 1: Ollama Setup
    print("\n1. Testing Ollama Setup...")
    ollama_ok = test_ollama_setup()
    
    if not ollama_ok:
        print("\n❌ Ollama setup failed. Please install Ollama and required models.")
        print("Installation instructions:")
        print("1. Install Ollama: https://ollama.ai/download")
        print("2. Pull models: ollama pull llama3.1:8b && ollama pull mistral:7b")
        return
    
    # Test 2: LLM Clients
    print("\n2. Testing LLM Clients...")
    llama_client, mistral_client = test_llm_client()
    
    if not llama_client or not mistral_client:
        print("\n❌ LLM client test failed. Check model availability.")
        return
    
    # Test 3: LLM Evaluator
    print("\n3. Testing LLM Evaluator...")
    evaluator = test_llm_evaluator(llama_client, mistral_client)
    
    if evaluator:
        print("\n🎉 All LLM evaluation tests completed successfully!")
        print("\nNext steps:")
        print("1. Run full evaluation on larger sample")
        print("2. Compare with traditional ML results")
        print("3. Generate final analysis report")
    else:
        print("\n❌ LLM evaluator test failed.")

if __name__ == "__main__":
    main()
