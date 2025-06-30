#!/usr/bin/env python3
"""
MathCoRL - Step 3: Method Comparison

Compare three approaches on mathematical reasoning datasets:
1. Zero-shot FPP baseline
2. FPP + Random examples
3. FPP + Policy Network examples

Usage:
    python run_comparison.py --dataset FinQA --samples 10
    python run_comparison.py --dataset GSM8K --samples 20 --save-results
    python run_comparison.py --help
"""

import argparse
import os
import sys
import json
import random
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from comparison_study_generic import GenericComparisonStudy
from mint.config import load_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(
        description='Compare FPP methods on mathematical reasoning datasets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_comparison.py --dataset FinQA --samples 10
    python run_comparison.py --dataset GSM8K --samples 20 --save-results
    python run_comparison.py --dataset SVAMP --samples 15 --output-dir my_results
    python run_comparison.py --dataset TabMWP --samples 25 --verbose
    python run_comparison.py --dataset TAT-QA --samples 30 --seed 42

Method Comparison:
    1. Zero-shot FPP: Function Prototype Prompting without examples
    2. FPP + Random: FPP with randomly selected examples  
    3. FPP + Policy Network: FPP with policy-selected examples

Dataset Information:
    - GSM8K: Grade School Math (k=2 examples)
    - SVAMP: Simple Arithmetic (k=2 examples)
    - TabMWP: Tabular Math (k=2 examples) 
    - TAT-QA: Table-and-Text QA (k=3 examples)
    - FinQA: Financial QA (k=2 examples)
        """
    )
    
    # Required arguments
    parser.add_argument(
        '--dataset', '-d',
        type=str,
        required=True,
        choices=['GSM8K', 'SVAMP', 'TabMWP', 'TAT-QA', 'FinQA'],
        help='Dataset to run comparison on'
    )
    
    # Test parameters
    parser.add_argument(
        '--samples', '-s',
        type=int,
        default=10,
        help='Number of test samples to evaluate (default: 10)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    # Directories
    parser.add_argument(
        '--candidates-dir',
        type=str,
        default='candidates',
        help='Directory containing candidate files (default: candidates)'
    )
    
    parser.add_argument(
        '--models-dir',
        type=str,
        default='models',
        help='Directory containing trained models (default: models)'
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='results',
        help='Output directory for results (default: results)'
    )
    
    # Options
    parser.add_argument(
        '--save-results',
        action='store_true',
        help='Save detailed results to JSON file'
    )
    
    parser.add_argument(
        '--skip-policy',
        action='store_true',
        help='Skip Policy Network method (if model not available)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress detailed output during testing'
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.info("Verbose logging enabled")
    elif args.quiet:
        logging.getLogger().setLevel(logging.WARNING)
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    logger.info(f"Random seed set to: {args.seed}")
    
    # Load configuration
    config = load_config()
    
    # Display configuration
    logger.info("🏆 MathCoRL - Method Comparison")
    logger.info("=" * 50)
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Test samples: {args.samples}")
    logger.info(f"Random seed: {args.seed}")
    logger.info(f"Candidates dir: {args.candidates_dir}")
    logger.info(f"Models dir: {args.models_dir}")
    logger.info(f"Output dir: {args.output_dir}")
    logger.info(f"Save results: {args.save_results}")
    
    # Check required files
    candidates_file = os.path.join(args.candidates_dir, f"{args.dataset}.json")
    if not os.path.exists(candidates_file):
        logger.error(f"❌ Candidates file not found: {candidates_file}")
        logger.error("Please run generate_candidates.py first")
        return 1
    
    model_file = os.path.join(args.models_dir, f"{args.dataset}_policy_best.pt")
    if not os.path.exists(model_file) and not args.skip_policy:
        logger.warning(f"⚠️ Policy model not found: {model_file}")
        logger.warning("Policy Network method will be skipped")
        logger.warning("Use train_policy.py to train a model, or --skip-policy to suppress this warning")
    
    # Ensure output directory exists
    if args.save_results:
        os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        # Initialize comparison study
        logger.info("🔄 Initializing comparison study...")
        study = GenericComparisonStudy(args.dataset)
        
        # Check if Policy Network is available
        if study.policy_network is None and not args.skip_policy:
            logger.warning("⚠️ Policy Network not available - will skip policy method")
        
        # Run comparison
        logger.info(f"🚀 Starting comparison on {args.samples} {args.dataset} samples...")
        logger.info("=" * 50)
        
        results = study.run_comparison(n_samples=args.samples)
        
        # Display final summary
        logger.info("\n" + "=" * 60)
        logger.info("🏆 FINAL COMPARISON RESULTS")
        logger.info("=" * 60)
        logger.info(f"Dataset: {results['dataset']}")
        logger.info(f"Samples tested: {results['n_samples']}")
        logger.info(f"Best method: {results['best_method']} ({results['best_accuracy']:.1f}%)")
        logger.info("-" * 60)
        
        # Method-wise results
        methods_info = [
            ("Zero-shot FPP", results['accuracies']['zero_shot_fpp'], "🎯"),
            ("FPP + Random", results['accuracies']['fpp_random'], "🎲"),
            ("FPP + Policy Net", results['accuracies']['fpp_policy'], "🤖")
        ]
        
        for method_name, accuracy, emoji in methods_info:
            if accuracy is not None:
                status = "🏆" if method_name == results['best_method'] else "  "
                logger.info(f"{status} {emoji} {method_name:<18}: {accuracy:>6.1f}%")
            else:
                logger.info(f"   {emoji} {method_name:<18}: {'N/A':>6}")
        
        logger.info("=" * 60)
        
        # Calculate improvement
        if results['accuracies']['fpp_policy'] is not None:
            baseline = results['accuracies']['zero_shot_fpp']
            policy_acc = results['accuracies']['fpp_policy']
            improvement = policy_acc - baseline
            
            if improvement > 0:
                logger.info(f"📈 Policy Network improvement: +{improvement:.1f}% over zero-shot")
            elif improvement < 0:
                logger.info(f"📉 Policy Network performance: {improvement:.1f}% vs zero-shot")
            else:
                logger.info(f"📊 Policy Network matches zero-shot performance")
        
        # Save results if requested
        if args.save_results:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = os.path.join(
                args.output_dir, 
                f"{args.dataset}_comparison_{timestamp}.json"
            )
            
            # Add metadata
            results['metadata'] = {
                'timestamp': timestamp,
                'command_line_args': vars(args),
                'config': config,
                'random_seed': args.seed
            }
            
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            logger.info(f"💾 Detailed results saved to: {results_file}")
        
        # Return appropriate exit code
        if results['accuracies']['fpp_policy'] is not None:
            # All methods completed successfully
            return 0
        else:
            # Policy method failed/skipped
            return 2 if args.skip_policy else 1
        
    except Exception as e:
        logger.error(f"❌ Comparison failed: {e}")
        if args.verbose:
            import traceback
            logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    exit(main()) 