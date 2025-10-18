"""
Logging utility for LLM vs Traditional ML comparison project.

This module provides centralized logging configuration for the entire project,
ensuring consistent log formatting and output across all modules.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime

from .config import LOG_LEVEL, LOG_FORMAT, PROJECT_ROOT

def setup_logger(name: str, level: str = LOG_LEVEL) -> logging.Logger:
    """
    Set up a logger with consistent formatting and output.
    
    Args:
        name: Name of the logger (typically __name__)
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    
    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    # Create formatter
    formatter = logging.Formatter(LOG_FORMAT)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))
    console_handler.setFormatter(formatter)
    
    # File handler (optional - for debugging)
    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"project_{timestamp}.log"
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)  # Always log everything to file
    file_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger

def get_logger(name: str) -> logging.Logger:
    """
    Get an existing logger or create a new one.
    
    Args:
        name: Name of the logger (typically __name__)
    
    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        return setup_logger(name)
    return logger

# Create a default logger for the project
project_logger = setup_logger("llm_vs_ml_project")
