#!/usr/bin/env python3
"""
Run final comparison analysis between Phase 2 and Phase 3.
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

from comparison.final_analyzer import FinalAnalyzer
from utils.logger import get_logger

logger = get_logger(__name__)

def main():
    """
    Main function to run final comparison analysis.
    """
    logger.info("Starting Final Comparison Analysis")
    logger.info("="*50)
    
    try:
        analyzer = FinalAnalyzer()
        results = analyzer.run_complete_analysis()
        
        if 'error' in results:
            logger.error(f"Analysis failed: {results['error']}")
            return 1
        
        logger.info("Final comparison analysis completed successfully!")
        logger.info(f"Results saved to: {results['output_directory']}")
        
        # Print summary
        print("\n" + results['final_report'])
        
        return 0
        
    except Exception as e:
        logger.error(f"Error in final analysis: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
