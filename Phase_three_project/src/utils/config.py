"""
Configuration file for Phase 3: Advanced LLM Strategies

This module contains all configuration parameters for implementing
few-shot learning, fine-tuning, prompt engineering, and RAG strategies.
"""

from pathlib import Path
from typing import Dict, List, Any

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
DOCS_DIR = PROJECT_ROOT / "docs"

# Data file paths
TOTAL_DATA_PATH = RAW_DATA_DIR / "total_data.csv"
TRAIN_DATA_PATH = PROCESSED_DATA_DIR / "train_data.csv"
TEST_DATA_PATH = PROCESSED_DATA_DIR / "test_data.csv"

# Data parameters
FEATURE_COLUMNS = [
    'PA_mean', 'PA_std', 'WW_mean', 'WW_std', 'SW_mean', 'SW_std', 
    'NW_mean', 'NW_std', 'VWA', 'MWA', 'W/B', 'VRC', 'MRC', 
    'Gender', 'Age', 'US_born', 'HH_Income', 'HH_size', 'Edu_level'
]
TARGET_COLUMN = 'Health_status'
ID_COLUMN = 'SEQN'

# Data splitting parameters
TEST_SIZE = 0.3
RANDOM_STATE = 42

# Ollama configuration
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODELS = ["llama3.1:8b", "mistral:7b"]

# LLM parameters
LLM_TEMPERATURE = 0.1  # Lower temperature for more consistent predictions
LLM_MAX_TOKENS = 10     # Short responses (0 or 1)
LLM_TOP_P = 0.9
LLM_BATCH_SIZE = 5      # Smaller batches for better quality

# Few-Shot Learning parameters
FEW_SHOT_EXAMPLES = 5   # Number of examples to include
FEW_SHOT_SAMPLE_SIZE = 30  # Sample size for few-shot evaluation

# Fine-tuning parameters
FINE_TUNING_EPOCHS = 3
FINE_TUNING_LEARNING_RATE = 2e-4
FINE_TUNING_BATCH_SIZE = 4
FINE_TUNING_MAX_LENGTH = 512

# PEFT/LoRA parameters
LORA_RANK = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.1
LORA_TARGET_MODULES = ["q_proj", "v_proj", "k_proj", "o_proj"]

# Prompt Engineering parameters
PROMPT_TEMPLATES = {
    "research_minimal": "Research classification task. Input features: {features}. Classify as 0 (good health) or 1 (poor health). Output only the digit (0 or 1).",
    
    "classification_direct": "Binary classification. Data: {features}. Label (0 or 1):",
    
    "educational_survey": "Educational exercise: classify survey response. Profile: {features}. Health status - respond with exactly 0 or 1:",
    
    "textual_profile": "Research: classify this patient profile as 0 (good) or 1 (poor): {features_text}. Answer with single digit."
}

# RAG parameters
RAG_TOP_K = 5  # Number of similar cases to retrieve
RAG_SIMILARITY_THRESHOLD = 0.7
RAG_KNOWLEDGE_BASE_SIZE = 1000  # Size of knowledge base for retrieval

# Evaluation parameters
EVALUATION_METRICS = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc', 'mcc']
CROSS_VALIDATION_FOLDS = 5
STATISTICAL_SIGNIFICANCE_LEVEL = 0.05

# Model comparison parameters
COMPARISON_MODELS = {
    'traditional_ml': ['RandomForest', 'LogisticRegression', 'SVM'],
    'llm_strategies': ['zero_shot', 'few_shot', 'prompt_engineering', 'fine_tuning', 'rag']
}

# Results saving parameters
SAVE_PREDICTIONS = True
SAVE_CONFUSION_MATRICES = True
SAVE_FEATURE_IMPORTANCE = True
SAVE_TRAINING_CURVES = True

# Logging configuration
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FILE = RESULTS_DIR / "phase3.log"

# Performance monitoring
ENABLE_PROFILING = True
PROFILE_OUTPUT_DIR = RESULTS_DIR / "profiling"
