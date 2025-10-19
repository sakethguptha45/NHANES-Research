"""
LLM Evaluation Module

This module provides functionality for evaluating Large Language Models (LLMs)
for clinical prediction tasks using Ollama models.

Modules:
- llm_client: Client for interacting with Ollama models
- llm_evaluator: Comprehensive evaluation and comparison of LLM models
"""

from .llm_client import OllamaClient
from .llm_evaluator import LLMEvaluator

__all__ = ['OllamaClient', 'LLMEvaluator']
