#!/usr/bin/env python3
"""
Large-scale test focusing on the successful Mistral 7B + textual_profile template.
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, Any, List
import json
import time

# Add src to path for imports
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

# Change to project root directory
os.chdir(project_root)

from data_preparation.data_loader import Phase3DataLoader
from prompt_engineering.prompt_engineer import PromptEngineeringEvaluator
from utils.helpers import setup_logger
from utils.config import TOTAL_DATA_PATH, OLLAMA_MODELS

logger = setup_logger(__name__)

def run_large_scale_test(sample_size: int = 100) -> Dict[str, Any]:
    """
    Run large-scale test with the successful template.
    
    Args:
        sample_size: Size of test sample
        
    Returns:
        Test results
    """
    logger.info(f"Starting large-scale test with {sample_size} samples")
    
    # Load and prepare data
    loader = Phase3DataLoader(data_path=TOTAL_DATA_PATH)
    df = loader.load_data()
    cleaned_df = loader.clean_data()
    
    # Split data
    X_train, X_test, y_train, y_test = loader.split_data(cleaned_df)
    
    # Sample test data with stratification
    if len(X_test) > sample_size:
        temp_df = pd.DataFrame(X_test.copy())
        temp_df['Health_status'] = y_test.copy()
        
        stratify_proportions = temp_df['Health_status'].value_counts(normalize=True)
        samples_per_class = (stratify_proportions * sample_size).round().astype(int)
        
        sampled_indices = []
        for class_label, num_samples in samples_per_class.items():
            class_indices = temp_df[temp_df['Health_status'] == class_label].index.tolist()
            num_samples = min(num_samples, len(class_indices))
            sampled_indices.extend(np.random.choice(class_indices, num_samples, replace=False))
        
        X_sample = X_test.loc[sampled_indices]
        y_sample = y_test.loc[sampled_indices]
    else:
        X_sample = X_test
        y_sample = y_test
    
    logger.info(f"Test sample: {len(X_sample)} samples")
    logger.info(f"Class distribution: {y_sample.value_counts().to_dict()}")
    
    # Test only Mistral 7B with textual_profile template
    results = {}
    
    try:
        evaluator = PromptEngineeringEvaluator(['mistral:7b'])
        
        # Create test DataFrame with labels
        test_df = X_sample.copy()
        test_df['Health_status'] = y_sample
        
        # Test only the successful template
        template_results = evaluator.evaluate_all_templates(test_df)
        
        # Extract only textual_profile results
        mistral_results = template_results['mistral:7b']
        textual_profile_results = mistral_results['textual_profile']
        
        results['mistral_7b_textual_profile'] = textual_profile_results
        
        logger.info(f"Large-scale test completed!")
        logger.info(f"Accuracy: {textual_profile_results.get('accuracy', 0.0):.4f}")
        logger.info(f"F1-Score: {textual_profile_results.get('f1_score', 0.0):.4f}")
        logger.info(f"Success Rate: {textual_profile_results.get('success_rate', 0.0):.4f}")
        logger.info(f"Valid Predictions: {textual_profile_results.get('valid_samples', 0)}/{textual_profile_results.get('total_samples', 0)}")
        
        return results
        
    except Exception as e:
        logger.error(f"Error in large-scale test: {e}")
        return {'error': str(e)}

def run_comparison_with_phase2() -> Dict[str, Any]:
    """
    Compare Phase 3 results with Phase 2 traditional ML results.
    
    Returns:
        Comparison results
    """
    logger.info("Comparing Phase 3 results with Phase 2...")
    
    try:
        # Load Phase 2 results
        phase2_path = Path("../Phase_two_project/results")
        
        comparison_results = {
            'phase2_available': False,
            'phase3_results': {},
            'comparison': {}
        }
        
        # Check if Phase 2 results exist
        ml_results_file = phase2_path / "ml_model_results.json"
        if ml_results_file.exists():
            with open(ml_results_file, 'r') as f:
                phase2_results = json.load(f)
            comparison_results['phase2_available'] = True
            comparison_results['phase2_results'] = phase2_results
            
            logger.info("Phase 2 results loaded successfully")
            
            # Extract best Phase 2 model performance
            best_phase2_accuracy = 0.0
            best_phase2_model = None
            
            for model_name, model_results in phase2_results.items():
                if isinstance(model_results, dict) and 'accuracy' in model_results:
                    accuracy = model_results.get('accuracy', 0.0)
                    if accuracy > best_phase2_accuracy:
                        best_phase2_accuracy = accuracy
                        best_phase2_model = model_name
            
            comparison_results['best_phase2_model'] = best_phase2_model
            comparison_results['best_phase2_accuracy'] = best_phase2_accuracy
            
            logger.info(f"Best Phase 2 model: {best_phase2_model} with accuracy: {best_phase2_accuracy:.4f}")
            
        else:
            logger.warning("Phase 2 results not found")
        
        return comparison_results
        
    except Exception as e:
        logger.error(f"Error comparing with Phase 2: {e}")
        return {'error': str(e)}

def generate_comprehensive_report(large_scale_results: Dict[str, Any], comparison_results: Dict[str, Any]) -> str:
    """
    Generate comprehensive report with large-scale results and Phase 2 comparison.
    
    Args:
        large_scale_results: Results from large-scale test
        comparison_results: Results from Phase 2 comparison
        
    Returns:
        Report text
    """
    report_parts = [
        "# Phase 3: Large-Scale Test Results & Phase 2 Comparison",
        "",
        f"Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Executive Summary",
        ""
    ]
    
    # Large-scale results
    if 'mistral_7b_textual_profile' in large_scale_results:
        results = large_scale_results['mistral_7b_textual_profile']
        
        report_parts.extend([
            f"### Phase 3 Large-Scale Results (Mistral 7B + Textual Profile)",
            "",
            f"- **Accuracy**: {results.get('accuracy', 0.0):.4f}",
            f"- **F1-Score**: {results.get('f1_score', 0.0):.4f}",
            f"- **Precision**: {results.get('precision', 0.0):.4f}",
            f"- **Recall**: {results.get('recall', 0.0):.4f}",
            f"- **Success Rate**: {results.get('success_rate', 0.0):.4f}",
            f"- **Valid Predictions**: {results.get('valid_samples', 0)}/{results.get('total_samples', 0)}",
            f"- **Sample Size**: {results.get('total_samples', 0)}",
            ""
        ])
        
        # Confusion matrix
        if 'confusion_matrix' in results:
            cm = results['confusion_matrix']
            report_parts.extend([
                "### Confusion Matrix",
                "",
                f"| | Predicted 0 | Predicted 1 |",
                f"|--|-------------|-------------|",
                f"| Actual 0 | {cm[0][0]} | {cm[0][1]} |",
                f"| Actual 1 | {cm[1][0]} | {cm[1][1]} |",
                ""
            ])
    
    # Phase 2 comparison
    if comparison_results.get('phase2_available', False):
        report_parts.extend([
            "## Phase 2 Comparison",
            "",
            f"- **Best Phase 2 Model**: {comparison_results.get('best_phase2_model', 'N/A')}",
            f"- **Best Phase 2 Accuracy**: {comparison_results.get('best_phase2_accuracy', 0.0):.4f}",
            ""
        ])
        
        # Performance comparison
        if 'mistral_7b_textual_profile' in large_scale_results:
            phase3_accuracy = large_scale_results['mistral_7b_textual_profile'].get('accuracy', 0.0)
            phase2_accuracy = comparison_results.get('best_phase2_accuracy', 0.0)
            
            improvement = phase3_accuracy - phase2_accuracy
            
            report_parts.extend([
                "### Performance Comparison",
                "",
                f"- **Phase 3 Accuracy**: {phase3_accuracy:.4f}",
                f"- **Phase 2 Accuracy**: {phase2_accuracy:.4f}",
                f"- **Improvement**: {improvement:+.4f} ({improvement/phase2_accuracy*100:+.1f}%)",
                ""
            ])
            
            if improvement > 0:
                report_parts.append("🎉 **Phase 3 outperforms Phase 2!**")
            elif improvement < -0.05:
                report_parts.append("⚠️ **Phase 2 significantly outperforms Phase 3**")
            else:
                report_parts.append("📊 **Phase 3 and Phase 2 have similar performance**")
            
            report_parts.append("")
    
    # Key insights
    report_parts.extend([
        "## Key Insights",
        "",
        "1. **Textual Feature Conversion**: Converting numerical features to natural language descriptions significantly improved LLM performance",
        "",
        "2. **Model Selection**: Mistral 7B was more cooperative than Llama 3.1 8B for this task",
        "",
        "3. **Template Effectiveness**: The 'textual_profile' template was the only successful approach",
        "",
        "4. **Research Framing**: Framing the task as research rather than medical advice reduced LLM refusal rates",
        "",
        "5. **Success Rate**: Even with successful predictions, the success rate indicates room for improvement",
        ""
    ])
    
    return "\n".join(report_parts)

def main():
    """
    Main function to run large-scale test and comparison.
    """
    logger.info("Starting Phase 3 Large-Scale Test & Phase 2 Comparison")
    logger.info("="*60)
    
    try:
        # Run large-scale test
        logger.info("\n" + "="*50)
        logger.info("Running Large-Scale Test")
        logger.info("="*50)
        large_scale_results = run_large_scale_test(sample_size=100)
        
        # Run Phase 2 comparison
        logger.info("\n" + "="*50)
        logger.info("Comparing with Phase 2")
        logger.info("="*50)
        comparison_results = run_comparison_with_phase2()
        
        # Generate comprehensive report
        logger.info("\n" + "="*50)
        logger.info("Generating Comprehensive Report")
        logger.info("="*50)
        
        report = generate_comprehensive_report(large_scale_results, comparison_results)
        
        # Save results
        results_dir = Path("results/large_scale_test")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Save detailed results
        with open(results_dir / "large_scale_results.json", "w") as f:
            json.dump(large_scale_results, f, indent=2, default=str)
        
        with open(results_dir / "comparison_results.json", "w") as f:
            json.dump(comparison_results, f, indent=2, default=str)
        
        # Save report
        with open(results_dir / "comprehensive_report.txt", "w") as f:
            f.write(report)
        
        logger.info(f"Results saved to: {results_dir}")
        logger.info("\n" + "="*60)
        logger.info("Large-Scale Test & Comparison Completed!")
        logger.info("="*60)
        
        # Print summary
        print("\n" + report)
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise

if __name__ == "__main__":
    main()
