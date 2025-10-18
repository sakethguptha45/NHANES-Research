"""
Configuration settings for LLM vs Traditional ML comparison project.

This module contains all configuration parameters used throughout the project,
including model parameters, data paths, and evaluation settings.
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
DOCS_DIR = PROJECT_ROOT / "docs"

# Data file paths
TOTAL_DATA_PATH = RAW_DATA_DIR / "total_data.csv"
TRAIN_DATA_PATH = PROCESSED_DATA_DIR / "train_data.csv"
TEST_DATA_PATH = PROCESSED_DATA_DIR / "test_data.csv"
LLM_SAMPLES_PATH = PROCESSED_DATA_DIR / "llm_samples.csv"

# Model configuration
RANDOM_STATE = 42
TEST_SIZE = 0.3
LLM_SAMPLE_SIZE = 200

# Traditional ML model parameters
ML_MODEL_PARAMS = {
    'RandomForest': {
        'n_estimators': 100,
        'class_weight': 'balanced',
        'random_state': RANDOM_STATE,
        'max_depth': None,
        'min_samples_split': 2,
        'min_samples_leaf': 1
    },
    'LogisticRegression': {
        'class_weight': 'balanced',
        'max_iter': 1000,
        'random_state': RANDOM_STATE,
        'solver': 'lbfgs'
    },
    'SVM': {
        'class_weight': 'balanced',
        'probability': True,
        'random_state': RANDOM_STATE,
        'kernel': 'rbf',
        'C': 1.0
    },
    'XGBoost': {
        'scale_pos_weight': 4,  # Handle class imbalance
        'random_state': RANDOM_STATE,
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1
    }
}

# LLM configuration
LLM_MODELS = {
    'Llama3.1': 'llama3.1:8b',
    'Mistral7B': 'mistral:7b'
}

OLLAMA_BASE_URL = "http://localhost:11434/api"
OLLAMA_TIMEOUT = 30
OLLAMA_TEMPERATURE = 0.1
OLLAMA_TOP_P = 0.9

# Evaluation metrics
EVALUATION_METRICS = [
    'accuracy',
    'precision',
    'recall',
    'f1_score',
    'auc_roc',
    'confusion_matrix'
]

# Statistical test parameters
BOOTSTRAP_SAMPLES = 1000
CONFIDENCE_LEVEL = 0.95
SIGNIFICANCE_LEVEL = 0.05

# Visualization settings
PLOT_STYLE = 'seaborn-v0_8'
FIGURE_SIZE = (12, 8)
DPI = 300

# Logging configuration
LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# Feature columns (excluding target and ID)
FEATURE_COLUMNS = [
    'PA_mean', 'PA_std',           # Physical activity
    'WW_mean', 'WW_std',           # Wake wear time
    'SW_mean', 'SW_std',           # Sleep wear time
    'NW_mean', 'NW_std',           # Non-wake time
    'VWA', 'MWA', 'W/B',           # Activity types
    'VRC', 'MRC',                   # Recreation
    'Gender', 'Age', 'US_born',     # Demographics
    'HH_Income', 'HH_size',         # Household
    'Edu_level'                     # Education
]

TARGET_COLUMN = 'Health_status'
ID_COLUMN = 'SEQN'

# Ensure directories exist
def ensure_directories():
    """Create necessary directories if they don't exist."""
    directories = [
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        RESULTS_DIR / "ml_results",
        RESULTS_DIR / "llm_results",
        RESULTS_DIR / "comparisons",
        RESULTS_DIR / "visualizations",
        DOCS_DIR
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

# Initialize directories on import
ensure_directories()
