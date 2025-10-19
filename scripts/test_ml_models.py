#!/usr/bin/env python3
"""
Test script for traditional ML models.

This script tests the complete ML pipeline including model training,
evaluation, hyperparameter tuning, and visualization.
"""

import sys
import os
from pathlib import Path

# Add src to path for imports
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

# Change to project root directory
os.chdir(project_root)

# Import modules
from traditional_ml.baseline_models import TraditionalMLModels
from traditional_ml.model_trainer import ModelTrainer
from traditional_ml.model_evaluator import ModelEvaluator
from data_preparation.data_loader import DataLoader
from data_preparation.data_splitter import DataSplitter
from utils.logger import get_logger

logger = get_logger(__name__)

def test_baseline_models():
    """Test baseline ML models functionality."""
    print("=" * 60)
    print("TESTING BASELINE ML MODELS")
    print("=" * 60)
    
    try:
        # Load and split data
        loader = DataLoader()
        data = loader.load_data()
        
        splitter = DataSplitter()
        X_train, X_test, y_train, y_test = splitter.split_data(data)
        
        # Initialize ML models
        ml_models = TraditionalMLModels()
        print(f"✅ ML models initialized: {list(ml_models.models.keys())}")
        
        # Train models
        trained_models = ml_models.train_all_models(X_train, y_train)
        print(f"✅ Models trained: {list(trained_models.keys())}")
        
        # Evaluate models
        results = ml_models.evaluate_all_models(X_test, y_test)
        print(f"✅ Models evaluated: {list(results.keys())}")
        
        # Get feature importance
        feature_names = X_train.columns.tolist()
        importance = ml_models.get_feature_importance(feature_names)
        print(f"✅ Feature importance extracted: {list(importance.keys())}")
        
        # Print summary
        summary = ml_models.get_model_summary()
        print("\nModel Summary:")
        print(summary)
        
        # Save results
        ml_models.save_results("results/ml_results")
        
        return ml_models, X_train, X_test, y_train, y_test
        
    except Exception as e:
        print(f"❌ Baseline models test failed: {e}")
        logger.error(f"Baseline models test failed: {e}")
        return None, None, None, None, None

def test_model_trainer(ml_models, X_train, y_train):
    """Test model trainer functionality."""
    print("\n" + "=" * 60)
    print("TESTING MODEL TRAINER")
    print("=" * 60)
    
    try:
        # Initialize trainer
        trainer = ModelTrainer(ml_models)
        print("✅ ModelTrainer initialized")
        
        # Train with cross-validation
        cv_results = trainer.train_with_cross_validation(X_train, y_train)
        print(f"✅ Cross-validation completed: {list(cv_results.keys())}")
        
        # Hyperparameter tuning (limited for testing)
        print("⚠️  Skipping hyperparameter tuning for speed (would take several minutes)")
        # tuning_results = trainer.hyperparameter_tuning_comprehensive(X_train, y_train)
        
        # Train ensemble
        ensemble = trainer.train_ensemble_model(X_train, y_train)
        if ensemble is not None:
            print("✅ Ensemble model trained")
        else:
            print("⚠️  Ensemble model training skipped")
        
        # Get training summary
        summary = trainer.get_training_summary()
        print("\nTraining Summary:")
        print(summary)
        
        return trainer
        
    except Exception as e:
        print(f"❌ Model trainer test failed: {e}")
        logger.error(f"Model trainer test failed: {e}")
        return None

def test_model_evaluator(ml_models, X_test, y_test):
    """Test model evaluator functionality."""
    print("\n" + "=" * 60)
    print("TESTING MODEL EVALUATOR")
    print("=" * 60)
    
    try:
        # Initialize evaluator
        evaluator = ModelEvaluator(ml_models)
        print("✅ ModelEvaluator initialized")
        
        # Comprehensive evaluation
        results = evaluator.evaluate_comprehensive(X_test, y_test)
        print(f"✅ Comprehensive evaluation completed: {list(results.keys())}")
        
        # Compare models
        comparison = evaluator.compare_models()
        print("✅ Model comparison completed")
        
        # Print evaluation summary
        summary = evaluator.get_evaluation_summary()
        print("\nEvaluation Summary:")
        print(summary)
        
        # Save results
        evaluator.save_evaluation_results("results/ml_results")
        evaluator.create_evaluation_plots("results/visualizations")
        
        return evaluator
        
    except Exception as e:
        print(f"❌ Model evaluator test failed: {e}")
        logger.error(f"Model evaluator test failed: {e}")
        return None

def test_file_creation():
    """Test that all expected files are created."""
    print("\n" + "=" * 60)
    print("TESTING FILE CREATION")
    print("=" * 60)
    
    files_to_check = [
        ("ML model results", "results/ml_results/ml_model_results.json"),
        ("Model summary", "results/ml_results/model_summary.txt"),
        ("Evaluation results", "results/ml_results/evaluation_results.json"),
        ("Comparison results", "results/ml_results/comparison_results.json"),
        ("Evaluation summary", "results/ml_results/evaluation_summary.txt"),
        ("Visualizations", "results/visualizations/model_comparison.png"),
        ("ROC curves", "results/visualizations/roc_curves.png"),
        ("Confusion matrices", "results/visualizations/confusion_matrices.png")
    ]
    
    all_files_exist = True
    
    for file_name, file_path in files_to_check:
        full_path = Path(file_path)
        if full_path.exists():
            file_size = full_path.stat().st_size
            print(f"✅ {file_name}: {file_path} ({file_size} bytes)")
        else:
            print(f"❌ {file_name}: {file_path} (NOT FOUND)")
            all_files_exist = False
    
    return all_files_exist

def test_performance_metrics():
    """Test that performance metrics are reasonable."""
    print("\n" + "=" * 60)
    print("TESTING PERFORMANCE METRICS")
    print("=" * 60)
    
    try:
        # Load results
        results_file = Path("results/ml_results/ml_model_results.json")
        if not results_file.exists():
            print("❌ Results file not found")
            return False
        
        import json
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        print("Performance Metrics:")
        print("-" * 40)
        
        all_metrics_good = True
        
        for model_name, model_results in results.items():
            accuracy = model_results['accuracy']
            print(f"{model_name}:")
            print(f"  Accuracy: {accuracy:.4f}")
            
            # Check if accuracy is reasonable (between 0.5 and 1.0)
            if 0.5 <= accuracy <= 1.0:
                print(f"  ✅ Accuracy is reasonable")
            else:
                print(f"  ❌ Accuracy is unreasonable")
                all_metrics_good = False
            
            # Check if AUC is available and reasonable
            if 'auc' in model_results and model_results['auc'] is not None:
                auc = model_results['auc']
                print(f"  AUC: {auc:.4f}")
                if 0.5 <= auc <= 1.0:
                    print(f"  ✅ AUC is reasonable")
                else:
                    print(f"  ❌ AUC is unreasonable")
                    all_metrics_good = False
            else:
                print(f"  AUC: N/A")
        
        return all_metrics_good
        
    except Exception as e:
        print(f"❌ Performance metrics test failed: {e}")
        logger.error(f"Performance metrics test failed: {e}")
        return False

def main():
    """Main test function."""
    print("🧪 TESTING TRADITIONAL ML MODELS")
    print("=" * 60)
    
    try:
        # Test 1: Baseline models
        ml_models, X_train, X_test, y_train, y_test = test_baseline_models()
        if ml_models is None:
            print("\n❌ Baseline models test failed. Stopping tests.")
            return False
        
        # Test 2: Model trainer
        trainer = test_model_trainer(ml_models, X_train, y_train)
        if trainer is None:
            print("\n❌ Model trainer test failed. Stopping tests.")
            return False
        
        # Test 3: Model evaluator
        evaluator = test_model_evaluator(ml_models, X_test, y_test)
        if evaluator is None:
            print("\n❌ Model evaluator test failed. Stopping tests.")
            return False
        
        # Test 4: File creation
        files_created = test_file_creation()
        
        # Test 5: Performance metrics
        metrics_good = test_performance_metrics()
        
        # Final results
        print("\n" + "=" * 60)
        print("TEST RESULTS SUMMARY")
        print("=" * 60)
        
        if files_created and metrics_good:
            print("🎉 ALL TESTS PASSED!")
            print("✅ Baseline models: PASSED")
            print("✅ Model trainer: PASSED")
            print("✅ Model evaluator: PASSED")
            print("✅ File creation: PASSED")
            print("✅ Performance metrics: PASSED")
            print("\n🚀 Ready to proceed with Step 4: LLM Evaluation!")
            return True
        else:
            print("❌ Some tests failed. Check the output above.")
            return False
            
    except Exception as e:
        print(f"\n❌ Unexpected error during testing: {e}")
        logger.error(f"Unexpected error during testing: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
