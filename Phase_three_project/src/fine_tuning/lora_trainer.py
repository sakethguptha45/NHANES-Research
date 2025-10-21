"""
LoRA-based fine-tuning for LLMs using PEFT library.
Simplified adapter approach for efficient training.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import json
import time
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
import warnings

from utils.config import (
    OLLAMA_MODELS, FINE_TUNING_EPOCHS, FINE_TUNING_LEARNING_RATE, 
    FINE_TUNING_BATCH_SIZE, LORA_RANK, LORA_ALPHA, LORA_DROPOUT, LORA_TARGET_MODULES
)
from utils.helpers import setup_logger, convert_features_to_text

logger = setup_logger(__name__)

class HealthDataset(Dataset):
    """
    Custom dataset for health prediction fine-tuning.
    """
    
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Tokenize text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class LoRATrainer:
    """
    LoRA-based fine-tuning trainer for health prediction.
    """
    
    def __init__(self, model_name: str, base_model_path: Optional[str] = None):
        """
        Initialize LoRA trainer.
        
        Args:
            model_name: Name of the model to fine-tune
            base_model_path: Optional path to base model (for local models)
        """
        self.model_name = model_name
        self.base_model_path = base_model_path
        self.tokenizer = None
        self.model = None
        self.lora_config = None
        
        # Configure LoRA parameters
        self.lora_config = LoraConfig(
            r=LORA_RANK,
            lora_alpha=LORA_ALPHA,
            target_modules=LORA_TARGET_MODULES,
            lora_dropout=LORA_DROPOUT,
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False
        )
        
        logger.info(f"LoRATrainer initialized for model: {model_name}")
    
    def _load_model_and_tokenizer(self):
        """
        Load the base model and tokenizer.
        """
        try:
            if self.base_model_path and Path(self.base_model_path).exists():
                # Load local model
                model_path = self.base_model_path
            else:
                # Use HuggingFace model (fallback)
                model_path = "microsoft/DialoGPT-medium"  # Smaller model for testing
            
            logger.info(f"Loading model from: {model_path}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            
            # Add padding token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            logger.info("Model and tokenizer loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def prepare_training_data(self, train_data: List[Dict]) -> Tuple[List[str], List[int]]:
        """
        Convert training data to text format for fine-tuning.
        
        Args:
            train_data: List of training samples
            
        Returns:
            Tuple of (texts, labels)
        """
        training_texts = []
        training_labels = []
        
        for sample in train_data:
            features_text = convert_features_to_text(sample['features'])
            label = sample['label']
            
            # Create training text in format: "Classify: {description} → {label}"
            text = f"Classify: {features_text} → {label}"
            training_texts.append(text)
            training_labels.append(label)
        
        logger.info(f"Prepared {len(training_texts)} training samples")
        return training_texts, training_labels
    
    def train(self, train_data: List[Dict], validation_data: Optional[List[Dict]] = None, 
              epochs: int = FINE_TUNING_EPOCHS) -> Dict[str, Any]:
        """
        Train model with LoRA adapters.
        
        Args:
            train_data: Training data
            validation_data: Optional validation data
            epochs: Number of training epochs
            
        Returns:
            Training results
        """
        logger.info(f"Starting LoRA fine-tuning for {epochs} epochs")
        logger.info(f"Training samples: {len(train_data)}")
        
        try:
            # Load model and tokenizer
            self._load_model_and_tokenizer()
            
            # Prepare training data
            train_texts, train_labels = self.prepare_training_data(train_data)
            
            # Create dataset
            train_dataset = HealthDataset(train_texts, train_labels, self.tokenizer)
            
            # Prepare validation data if provided
            val_dataset = None
            if validation_data:
                val_texts, val_labels = self.prepare_training_data(validation_data)
                val_dataset = HealthDataset(val_texts, val_labels, self.tokenizer)
            
            # Apply LoRA to model
            self.model = get_peft_model(self.model, self.lora_config)
            self.model.print_trainable_parameters()
            
            # Training arguments
            training_args = TrainingArguments(
                output_dir=f"./results/fine_tuning/{self.model_name}",
                num_train_epochs=epochs,
                per_device_train_batch_size=FINE_TUNING_BATCH_SIZE,
                per_device_eval_batch_size=FINE_TUNING_BATCH_SIZE,
                learning_rate=FINE_TUNING_LEARNING_RATE,
                logging_steps=10,
                save_steps=100,
                eval_steps=100 if val_dataset else None,
                evaluation_strategy="steps" if val_dataset else "no",
                save_strategy="steps",
                load_best_model_at_end=True if val_dataset else False,
                metric_for_best_model="eval_loss" if val_dataset else None,
                greater_is_better=False,
                warmup_steps=50,
                weight_decay=0.01,
                logging_dir=f"./logs/fine_tuning/{self.model_name}",
                report_to=None,  # Disable wandb/tensorboard
            )
            
            # Initialize trainer
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                tokenizer=self.tokenizer,
            )
            
            # Train the model
            logger.info("Starting training...")
            train_result = trainer.train()
            
            # Save the model
            trainer.save_model()
            self.tokenizer.save_pretrained(f"./results/fine_tuning/{self.model_name}")
            
            logger.info("Fine-tuning completed successfully")
            
            return {
                'train_loss': train_result.training_loss,
                'train_samples': len(train_data),
                'epochs': epochs,
                'model_path': f"./results/fine_tuning/{self.model_name}"
            }
            
        except Exception as e:
            logger.error(f"Error during fine-tuning: {e}")
            return {'error': str(e)}
    
    def predict(self, features: Dict[str, Any]) -> Optional[int]:
        """
        Make prediction using fine-tuned model.
        
        Args:
            features: Patient features
            
        Returns:
            Predicted label (0 or 1) or None if error
        """
        if self.model is None or self.tokenizer is None:
            logger.error("Model not loaded. Please train first.")
            return None
        
        try:
            # Convert features to text
            features_text = convert_features_to_text(features)
            prompt = f"Classify: {features_text} →"
            
            # Tokenize
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512
            )
            
            # Generate prediction
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=5,
                    temperature=0.1,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract prediction
            if "→ 0" in response:
                return 0
            elif "→ 1" in response:
                return 1
            else:
                logger.warning(f"Could not parse fine-tuned model response: {response}")
                return None
                
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            return None

class FineTuningEvaluator:
    """
    Evaluator for fine-tuned models.
    """
    
    def __init__(self, model_names: List[str]):
        """
        Initialize fine-tuning evaluator.
        
        Args:
            model_names: List of model names to evaluate
        """
        self.model_names = model_names
        self.trainers: Dict[str, LoRATrainer] = {}
        self.results: Dict[str, Dict[str, Any]] = {}
        
        logger.info(f"FineTuningEvaluator initialized with models: {model_names}")
    
    def train_all_models(self, train_data: List[Dict], validation_data: Optional[List[Dict]] = None) -> Dict[str, Dict[str, Any]]:
        """
        Train all models with LoRA fine-tuning.
        
        Args:
            train_data: Training data
            validation_data: Optional validation data
            
        Returns:
            Training results for all models
        """
        logger.info(f"Training {len(self.model_names)} models with LoRA fine-tuning")
        
        training_results = {}
        
        for model_name in self.model_names:
            logger.info(f"Training model: {model_name}")
            
            trainer = LoRATrainer(model_name)
            result = trainer.train(train_data, validation_data)
            
            self.trainers[model_name] = trainer
            training_results[model_name] = result
            
            logger.info(f"Training completed for {model_name}: {result}")
        
        return training_results
    
    def evaluate_all_models(self, test_data: List[Dict]) -> Dict[str, Dict[str, Any]]:
        """
        Evaluate all fine-tuned models.
        
        Args:
            test_data: Test data
            
        Returns:
            Evaluation results for all models
        """
        logger.info(f"Evaluating {len(self.trainers)} fine-tuned models")
        
        evaluation_results = {}
        
        for model_name, trainer in self.trainers.items():
            logger.info(f"Evaluating model: {model_name}")
            
            predictions = []
            for sample in test_data:
                prediction = trainer.predict(sample['features'])
                predictions.append(prediction)
            
            # Calculate metrics
            true_labels = [sample['label'] for sample in test_data]
            valid_predictions = [p for p in predictions if p is not None]
            valid_indices = [i for i, p in enumerate(predictions) if p is not None]
            valid_labels = [true_labels[i] for i in valid_indices]
            
            if len(valid_predictions) > 0:
                from utils.helpers import calculate_performance_metrics
                metrics = calculate_performance_metrics(valid_labels, valid_predictions)
                metrics['success_rate'] = len(valid_predictions) / len(predictions)
                metrics['total_samples'] = len(test_data)
                metrics['valid_samples'] = len(valid_predictions)
            else:
                metrics = {
                    'accuracy': 0.0,
                    'precision': 0.0,
                    'recall': 0.0,
                    'f1_score': 0.0,
                    'success_rate': 0.0,
                    'total_samples': len(test_data),
                    'valid_samples': 0
                }
            
            evaluation_results[model_name] = metrics
            logger.info(f"Evaluation completed for {model_name}: {metrics}")
        
        self.results = evaluation_results
        return evaluation_results
