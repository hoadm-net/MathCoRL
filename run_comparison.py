#!/usr/bin/env python3
"""
MathCoRL - Step 3: Method Comparison

Compare four approaches on mathematical reasoning datasets:
1. Zero-shot FPP baseline
2. FPP + Random examples
3. FPP + Policy Network examples
4. FPP + KATE examples (kNN-based selection)

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
    4. FPP + KATE: FPP with kNN-based similarity selection

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
    
    parser.add_argument('--methods', type=str, default='zero-shot,random,policy,kate,cds',
                       help='Comma-separated list of methods to compare (default: zero-shot,random,policy,kate,cds). '
                            'Available: zero-shot, random, policy, kate, cds')
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.info("Verbose logging enabled")
    elif args.quiet:
        logging.getLogger().setLevel(logging.WARNING)
    
    # Parse methods selection
    available_methods = ['zero-shot', 'random', 'policy', 'kate', 'cds']
    selected_methods = [m.strip() for m in args.methods.split(',')]
    
    # Validate methods
    for method in selected_methods:
        if method not in available_methods:
            logger.error(f"❌ Invalid method: {method}")
            logger.error(f"Available methods: {', '.join(available_methods)}")
            return 1
    
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
    logger.info(f"Methods: {', '.join(selected_methods)}")
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
        
        results = study.run_comparison(n_samples=args.samples, methods=selected_methods)
        
        # Display final summary
        logger.info("\n" + "=" * 60)
        logger.info("🏆 FINAL COMPARISON RESULTS")
        logger.info("=" * 60)
        logger.info(f"Dataset: {results['dataset']}")
        logger.info(f"Samples tested: {results['n_samples']}")
        logger.info(f"Best method: {results['best_method']} ({results['best_accuracy']:.1f}%)")
        logger.info("-" * 60)
        
        # Method-wise results - only show selected methods
        all_methods_info = [
            ("zero-shot", "Zero-shot FPP", results['accuracies']['zero_shot_fpp'], "🎯"),
            ("random", "FPP + Random", results['accuracies']['fpp_random'], "🎲"),
            ("policy", "FPP + Policy Net", results['accuracies']['fpp_policy'], "🤖"),
            ("kate", "FPP + KATE", results['accuracies']['fpp_kate'], "🔍"),
            ("cds", "FPP + CDS", results['accuracies']['fpp_cds'], "📚")
        ]
        
        # Filter to only selected methods
        methods_info = [(method_key, method_name, accuracy, emoji) 
                       for method_key, method_name, accuracy, emoji in all_methods_info 
                       if method_key in selected_methods]
        
        for method_key, method_name, accuracy, emoji in methods_info:
            if accuracy is not None:
                status = "🏆" if method_name == results['best_method'] else "  "
                logger.info(f"{status} {emoji} {method_name:<18}: {accuracy:>6.1f}%")
            else:
                logger.info(f"   {emoji} {method_name:<18}: {'N/A':>6}")
        
        logger.info("=" * 60)
        
        # Calculate improvements (only if both baseline and comparison methods are selected)
        if 'zero-shot' in selected_methods:
            baseline = results['accuracies']['zero_shot_fpp']
            
            # Policy Network improvement
            if 'policy' in selected_methods and results['accuracies']['fpp_policy'] is not None:
                policy_acc = results['accuracies']['fpp_policy']
                policy_improvement = policy_acc - baseline
                
                if policy_improvement > 0:
                    logger.info(f"📈 Policy Network improvement: +{policy_improvement:.1f}% over zero-shot")
                elif policy_improvement < 0:
                    logger.info(f"📉 Policy Network performance: {policy_improvement:.1f}% vs zero-shot")
                else:
                    logger.info(f"📊 Policy Network matches zero-shot performance")
            
            # KATE improvement
            if 'kate' in selected_methods and results['accuracies']['fpp_kate'] is not None:
                kate_acc = results['accuracies']['fpp_kate']
                kate_improvement = kate_acc - baseline
                
                if kate_improvement > 0:
                    logger.info(f"📈 KATE improvement: +{kate_improvement:.1f}% over zero-shot")
                elif kate_improvement < 0:
                    logger.info(f"📉 KATE performance: {kate_improvement:.1f}% vs zero-shot")
                else:
                    logger.info(f"📊 KATE matches zero-shot performance")
        
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