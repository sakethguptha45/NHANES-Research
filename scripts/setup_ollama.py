#!/usr/bin/env python3
"""
Ollama setup script for LLM vs Traditional ML comparison project.

This script handles the installation and setup of Ollama and required LLM models
for the project. It checks if Ollama is installed, downloads the necessary models,
and verifies the setup.
"""

import subprocess
import sys
import time
import requests
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from utils.config import LLM_MODELS, OLLAMA_BASE_URL
from utils.logger import get_logger

logger = get_logger(__name__)

def check_ollama_installed() -> bool:
    """
    Check if Ollama is installed on the system.
    
    Returns:
        True if Ollama is installed, False otherwise
    """
    try:
        result = subprocess.run(
            ["ollama", "--version"], 
            capture_output=True, 
            text=True, 
            check=True
        )
        logger.info(f"Ollama version: {result.stdout.strip()}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.error("Ollama is not installed or not in PATH")
        return False

def install_ollama() -> bool:
    """
    Provide instructions for installing Ollama.
    
    Returns:
        True if user confirms installation, False otherwise
    """
    print("\n" + "="*60)
    print("OLLAMA INSTALLATION REQUIRED")
    print("="*60)
    print("Ollama is not installed on your system.")
    print("\nTo install Ollama:")
    print("1. Visit: https://ollama.ai/download")
    print("2. Download the installer for your operating system")
    print("3. Run the installer")
    print("4. Restart your terminal")
    print("\nAfter installation, run this script again.")
    print("="*60)
    
    response = input("\nHave you installed Ollama? (y/n): ").lower().strip()
    return response in ['y', 'yes']

def check_ollama_running() -> bool:
    """
    Check if Ollama service is running.
    
    Returns:
        True if Ollama is running, False otherwise
    """
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/tags", timeout=5)
        if response.status_code == 200:
            logger.info("Ollama service is running")
            return True
        else:
            logger.error(f"Ollama service not responding: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        logger.error(f"Cannot connect to Ollama service: {e}")
        return False

def start_ollama_service() -> bool:
    """
    Attempt to start Ollama service.
    
    Returns:
        True if service started successfully, False otherwise
    """
    try:
        logger.info("Starting Ollama service...")
        subprocess.run(["ollama", "serve"], check=True, timeout=10)
        time.sleep(2)  # Give service time to start
        return check_ollama_running()
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        logger.error("Failed to start Ollama service")
        return False

def download_model(model_name: str) -> bool:
    """
    Download a specific Ollama model.
    
    Args:
        model_name: Name of the model to download
    
    Returns:
        True if download successful, False otherwise
    """
    try:
        logger.info(f"Downloading model: {model_name}")
        print(f"Downloading {model_name}... This may take several minutes.")
        
        result = subprocess.run(
            ["ollama", "pull", model_name],
            check=True,
            capture_output=True,
            text=True
        )
        
        logger.info(f"Successfully downloaded {model_name}")
        print(f"✅ {model_name} downloaded successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to download {model_name}: {e}")
        print(f"❌ Failed to download {model_name}")
        return False

def verify_model_availability() -> bool:
    """
    Verify that all required models are available.
    
    Returns:
        True if all models are available, False otherwise
    """
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/tags", timeout=5)
        if response.status_code != 200:
            logger.error("Failed to get model list from Ollama")
            return False
        
        available_models = [model['name'] for model in response.json().get('models', [])]
        required_models = list(LLM_MODELS.values())
        
        missing_models = set(required_models) - set(available_models)
        
        if missing_models:
            logger.warning(f"Missing models: {missing_models}")
            return False
        
        logger.info("All required models are available")
        print("✅ All required models are available")
        return True
        
    except Exception as e:
        logger.error(f"Failed to verify model availability: {e}")
        return False

def test_model_inference(model_name: str) -> bool:
    """
    Test model inference with a simple prompt.
    
    Args:
        model_name: Name of the model to test
    
    Returns:
        True if inference works, False otherwise
    """
    try:
        logger.info(f"Testing inference for {model_name}")
        
        test_prompt = "Hello, how are you?"
        
        response = requests.post(
            f"{OLLAMA_BASE_URL}/generate",
            json={
                "model": model_name,
                "prompt": test_prompt,
                "stream": False
            },
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            if 'response' in result:
                logger.info(f"Model {model_name} inference test successful")
                print(f"✅ {model_name} inference test passed")
                return True
        
        logger.error(f"Model {model_name} inference test failed")
        print(f"❌ {model_name} inference test failed")
        return False
        
    except Exception as e:
        logger.error(f"Model {model_name} inference test error: {e}")
        print(f"❌ {model_name} inference test error")
        return False

def setup_ollama() -> bool:
    """
    Complete Ollama setup process.
    
    Returns:
        True if setup successful, False otherwise
    """
    print("\n" + "="*60)
    print("OLLAMA SETUP FOR LLM VS TRADITIONAL ML PROJECT")
    print("="*60)
    
    # Step 1: Check if Ollama is installed
    if not check_ollama_installed():
        if not install_ollama():
            return False
        if not check_ollama_installed():
            return False
    
    # Step 2: Check if Ollama service is running
    if not check_ollama_running():
        print("\nOllama service is not running. Attempting to start...")
        if not start_ollama_service():
            print("Please start Ollama manually: ollama serve")
            return False
    
    # Step 3: Download required models
    print(f"\nDownloading required models: {list(LLM_MODELS.values())}")
    for model_name in LLM_MODELS.values():
        if not download_model(model_name):
            return False
    
    # Step 4: Verify model availability
    if not verify_model_availability():
        return False
    
    # Step 5: Test model inference
    print("\nTesting model inference...")
    for model_name in LLM_MODELS.values():
        if not test_model_inference(model_name):
            return False
    
    print("\n" + "="*60)
    print("🎉 OLLAMA SETUP COMPLETE!")
    print("="*60)
    print("All required models are installed and working.")
    print("You can now run the LLM evaluation experiments.")
    print("="*60)
    
    return True

def main():
    """Main function to run Ollama setup."""
    try:
        success = setup_ollama()
        if success:
            print("\nSetup completed successfully!")
            sys.exit(0)
        else:
            print("\nSetup failed. Please check the errors above.")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\nSetup interrupted by user.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error during setup: {e}")
        print(f"\nUnexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
