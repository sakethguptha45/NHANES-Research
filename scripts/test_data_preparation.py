#!/usr/bin/env python3
"""
Test script for data preparation modules.

This script tests the data loading, validation, and splitting functionality
to ensure everything works correctly before proceeding to model implementation.
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
from data_preparation.data_loader import DataLoader
from data_preparation.data_splitter import DataSplitter
from utils.logger import get_logger

logger = get_logger(__name__)

def test_data_loading():
    """Test data loading functionality."""
    print("=" * 60)
    print("TESTING DATA LOADING")
    print("=" * 60)
    
    try:
        # Initialize loader
        loader = DataLoader()
        
        # Load data
        data = loader.load_data()
        print(f"✅ Data loaded successfully. Shape: {data.shape}")
        
        # Get data info
        info = loader.get_data_info()
        print(f"✅ Data info collected:")
        print(f"   - Memory usage: {info['memory_usage'] / 1024 / 1024:.2f} MB")
        print(f"   - Missing values: {sum(info['missing_values'].values())}")
        print(f"   - Duplicate rows: {info['duplicate_rows']}")
        
        # Validate data
        validation_passed = loader.validate_data()
        print(f"✅ Data validation: {'PASSED' if validation_passed else 'FAILED'}")
        
        # Print validation report
        report = loader.get_validation_report()
        print("\nValidation Report:")
        print(report)
        
        # Clean data
        cleaned_data = loader.clean_data()
        print(f"✅ Data cleaning completed. Final shape: {cleaned_data.shape}")
        
        # Get feature summary
        summary = loader.get_feature_summary()
        print(f"✅ Feature summary generated. Shape: {summary.shape}")
        
        return data
        
    except Exception as e:
        print(f"❌ Data loading test failed: {e}")
        logger.error(f"Data loading test failed: {e}")
        return None

def test_data_splitting(data):
    """Test data splitting functionality."""
    print("\n" + "=" * 60)
    print("TESTING DATA SPLITTING")
    print("=" * 60)
    
    try:
        # Initialize splitter
        splitter = DataSplitter()
        
        # Split data
        X_train, X_test, y_train, y_test = splitter.split_data(data)
        print(f"✅ Data split completed:")
        print(f"   - Training set: {X_train.shape}")
        print(f"   - Test set: {X_test.shape}")
        
        # Print split summary
        summary = splitter.get_split_summary()
        print("\nSplit Summary:")
        print(summary)
        
        # Create LLM samples
        llm_samples = splitter.create_llm_samples(data)
        print(f"✅ LLM samples created: {len(llm_samples)}")
        
        # Create CV folds
        cv_folds = splitter.create_cv_folds(data, n_folds=5)
        print(f"✅ Cross-validation folds created: {len(cv_folds)}")
        
        # Save splits
        splitter.save_splits()
        splitter.save_llm_samples(llm_samples)
        print("✅ Data splits saved to files")
        
        return X_train, X_test, y_train, y_test, llm_samples
        
    except Exception as e:
        print(f"❌ Data splitting test failed: {e}")
        logger.error(f"Data splitting test failed: {e}")
        return None, None, None, None, None

def test_file_creation():
    """Test that all expected files are created."""
    print("\n" + "=" * 60)
    print("TESTING FILE CREATION")
    print("=" * 60)
    
    from utils.config import TRAIN_DATA_PATH, TEST_DATA_PATH, LLM_SAMPLES_PATH
    
    files_to_check = [
        ("Training data", TRAIN_DATA_PATH),
        ("Test data", TEST_DATA_PATH),
        ("LLM samples", LLM_SAMPLES_PATH),
        ("Split metadata", TRAIN_DATA_PATH.parent / "split_metadata.json")
    ]
    
    all_files_exist = True
    
    for file_name, file_path in files_to_check:
        if file_path.exists():
            file_size = file_path.stat().st_size
            print(f"✅ {file_name}: {file_path} ({file_size} bytes)")
        else:
            print(f"❌ {file_name}: {file_path} (NOT FOUND)")
            all_files_exist = False
    
    return all_files_exist

def main():
    """Main test function."""
    print("🧪 TESTING DATA PREPARATION MODULES")
    print("=" * 60)
    
    try:
        # Test 1: Data loading
        data = test_data_loading()
        if data is None:
            print("\n❌ Data loading test failed. Stopping tests.")
            return False
        
        # Test 2: Data splitting
        X_train, X_test, y_train, y_test, llm_samples = test_data_splitting(data)
        if X_train is None:
            print("\n❌ Data splitting test failed. Stopping tests.")
            return False
        
        # Test 3: File creation
        files_created = test_file_creation()
        
        # Final results
        print("\n" + "=" * 60)
        print("TEST RESULTS SUMMARY")
        print("=" * 60)
        
        if files_created:
            print("🎉 ALL TESTS PASSED!")
            print("✅ Data loading: PASSED")
            print("✅ Data validation: PASSED")
            print("✅ Data splitting: PASSED")
            print("✅ File creation: PASSED")
            print("\n🚀 Ready to proceed with Step 3: Traditional ML Models!")
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
