#!/usr/bin/env python3
"""
Prototype library ablation study.

Compares minimal prototypes (core arithmetic only) vs full prototype library
to validate design choices and quantify impact on accuracy.

Configurations tested:
1. Minimal (4 functions): add, sub, mul, div
2. Core (6 functions): add, sub, mul, div, round, sum
3. Full (23 functions): all available prototypes

Usage:
    python scripts/prototype_ablation.py --dataset GSM8K --samples 50
    python scripts/prototype_ablation.py --all-datasets --config minimal,full
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Any, Set
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mint.utils import execute_code, clean_code
from mint.functions import get_execution_namespace

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Prototype configurations
PROTOTYPE_CONFIGS = {
    'minimal': {
        'name': 'Minimal (4 functions)',
        'functions': ['add', 'sub', 'mul', 'div'],
        'description': 'Core arithmetic only'
    },
    'core': {
        'name': 'Core (14 functions)',
        'functions': ['add', 'sub', 'mul', 'div', 'mod', 'pow', 'min', 'max', 'abs', 'floor', 'round', 'ceil', 'sum', 'mean'],
        'description': 'Core functions for general mathematical reasoning'
    },
    'basic': {
        'name': 'Basic (6 functions)',
        'functions': ['add', 'sub', 'mul', 'div', 'round', 'sum'],
        'description': 'Basic arithmetic + rounding'
    },
    'full': {
        'name': 'Full (23 functions)',
        'functions': [
            'add', 'sub', 'mul', 'div', 'mod', 'pow',
            'min', 'max', 'abs', 'round', 'sum',
            'mean', 'median', 'mode',
            'percentage', 'gcd', 'lcm',
            'ceil', 'floor', 'count',
            'equal', 'greater_than', 'less_than'
        ],
        'description': 'All available prototypes'
    }
}


def execute_with_namespace(code: str, allowed_functions: Set[str]) -> tuple:
    """
    Execute code with restricted function namespace.
    
    Args:
        code: Python code to execute
        allowed_functions: Set of allowed function names
        
    Returns:
        Tuple of (result, error_message)
    """
    try:
        # Get full namespace
        full_namespace = get_execution_namespace()
        
        # Filter to allowed functions only
        filtered_namespace = {
            name: func for name, func in full_namespace.items()
            if name in allowed_functions or not callable(func)
        }
        
        # Create execution namespace with builtins
        namespace = {
            '__builtins__': {
                'range': range,
                'len': len,
                'float': float,
                'int': int,
                'str': str,
                'list': list,
                'dict': dict,
                # Only include builtins that match allowed functions
                'max': max if 'max' in allowed_functions else None,
                'min': min if 'min' in allowed_functions else None,
                'abs': abs if 'abs' in allowed_functions else None,
                'round': round if 'round' in allowed_functions else None,
                'sum': sum if 'sum' in allowed_functions else None,
            }
        }
        
        # Remove None values
        namespace['__builtins__'] = {k: v for k, v in namespace['__builtins__'].items() if v is not None}
        
        # Add filtered functions
        namespace.update(filtered_namespace)
        
        # Execute code
        exec(code, namespace, namespace)
        
        # Get result
        if 'result' in namespace:
            return namespace['result'], ""
        elif 'answer' in namespace:
            return namespace['answer'], ""
        elif 'solution' in namespace:
            return namespace['solution'], ""
        else:
            return None, "Variable 'result', 'answer', or 'solution' not found"
            
    except Exception as e:
        return None, f"Execution error: {str(e)}"


def evaluate_candidate(candidate: Dict, allowed_functions: Set[str]) -> Dict[str, Any]:
    """
    Evaluate a single candidate with restricted function namespace.
    
    Args:
        candidate: Candidate dictionary with 'code' and 'answer' fields
        allowed_functions: Set of allowed function names
        
    Returns:
        Dictionary with evaluation results
    """
    code = candidate.get('code', '')
    ground_truth = candidate.get('answer', 0)
    
    # Clean code
    cleaned_code = clean_code(code)
    
    # Execute with restricted namespace
    result, error = execute_with_namespace(cleaned_code, allowed_functions)
    
    # Check if correct
    is_correct = False
    if result is not None and error == "":
        try:
            pred_float = float(result)
            gt_float = float(ground_truth)
            is_correct = abs(pred_float - gt_float) < 1e-6
        except (ValueError, TypeError):
            pass
    
    return {
        'correct': is_correct,
        'success': result is not None and error == "",
        'error': error,
        'result': result
    }


def run_ablation(
    candidates: List[Dict],
    config_name: str,
    max_samples: int = None
) -> Dict[str, Any]:
    """
    Run ablation for a specific prototype configuration.
    
    Args:
        candidates: List of candidate solutions
        config_name: Name of prototype configuration
        max_samples: Maximum number of samples to test (None for all)
        
    Returns:
        Dictionary with ablation results
    """
    config = PROTOTYPE_CONFIGS[config_name]
    allowed_functions = set(config['functions'])
    
    logger.info(f"Testing {config['name']}: {config['description']}")
    logger.info(f"Allowed functions: {', '.join(sorted(allowed_functions))}")
    
    # Limit samples if specified
    test_candidates = candidates[:max_samples] if max_samples else candidates
    
    # Evaluate all candidates
    results = []
    correct_count = 0
    success_count = 0
    
    for i, candidate in enumerate(test_candidates):
        eval_result = evaluate_candidate(candidate, allowed_functions)
        results.append(eval_result)
        
        if eval_result['correct']:
            correct_count += 1
        if eval_result['success']:
            success_count += 1
        
        if (i + 1) % 10 == 0:
            logger.info(f"  Processed {i + 1}/{len(test_candidates)} samples...")
    
    # Calculate metrics
    total = len(test_candidates)
    accuracy = correct_count / total if total > 0 else 0
    success_rate = success_count / total if total > 0 else 0
    
    return {
        'config_name': config_name,
        'config_description': config['name'],
        'num_functions': len(allowed_functions),
        'functions': sorted(allowed_functions),
        'total_samples': total,
        'correct': correct_count,
        'success': success_count,
        'failed': total - success_count,
        'accuracy': accuracy,
        'success_rate': success_rate,
        'results': results
    }


def compare_configs(
    dataset_name: str,
    config_names: List[str],
    max_samples: int = None
) -> Dict[str, Any]:
    """
    Compare multiple prototype configurations.
    
    Args:
        dataset_name: Name of dataset (e.g., 'GSM8K')
        config_names: List of configuration names to compare
        max_samples: Maximum samples to test per config
        
    Returns:
        Dictionary with comparison results
    """
    # Load candidates
    candidates_path = Path('candidates') / f'{dataset_name}.json'
    if not candidates_path.exists():
        raise FileNotFoundError(f"Candidates not found: {candidates_path}")
    
    with open(candidates_path, 'r') as f:
        candidates = json.load(f)
    
    logger.info(f"\n{'='*80}")
    logger.info(f"PROTOTYPE ABLATION STUDY - {dataset_name}")
    logger.info(f"{'='*80}")
    logger.info(f"Total candidates: {len(candidates)}")
    logger.info(f"Testing samples: {max_samples if max_samples else 'all'}")
    logger.info(f"Configurations: {', '.join(config_names)}")
    logger.info(f"{'='*80}\n")
    
    # Run ablation for each config
    ablation_results = []
    for config_name in config_names:
        result = run_ablation(candidates, config_name, max_samples)
        ablation_results.append(result)
        
        logger.info(f"\n{'='*80}")
        logger.info(f"Results for {result['config_description']}")
        logger.info(f"{'='*80}")
        logger.info(f"  Functions: {result['num_functions']}")
        logger.info(f"  Accuracy: {result['accuracy']:.1%} ({result['correct']}/{result['total_samples']})")
        logger.info(f"  Success rate: {result['success_rate']:.1%} ({result['success']}/{result['total_samples']})")
        logger.info(f"  Failed: {result['failed']}")
    
    # Generate comparison summary
    logger.info(f"\n{'='*80}")
    logger.info("COMPARISON SUMMARY")
    logger.info(f"{'='*80}")
    logger.info(f"{'Config':<20} {'Functions':<12} {'Accuracy':<12} {'Success Rate':<12}")
    logger.info(f"{'-'*80}")
    
    for result in ablation_results:
        logger.info(
            f"{result['config_description']:<20} "
            f"{result['num_functions']:<12} "
            f"{result['accuracy']:.1%}        "
            f"{result['success_rate']:.1%}"
        )
    
    # Calculate relative performance
    baseline = ablation_results[0]  # First config is baseline
    logger.info(f"\n{'='*80}")
    logger.info(f"RELATIVE PERFORMANCE (baseline: {baseline['config_description']})")
    logger.info(f"{'='*80}")
    
    for result in ablation_results[1:]:
        acc_diff = result['accuracy'] - baseline['accuracy']
        acc_change = (acc_diff / baseline['accuracy'] * 100) if baseline['accuracy'] > 0 else 0
        
        logger.info(f"{result['config_description']}:")
        logger.info(f"  Accuracy change: {acc_diff:+.1%} ({acc_change:+.1f}%)")
        logger.info(f"  Functions added: {result['num_functions'] - baseline['num_functions']}")
    
    return {
        'dataset': dataset_name,
        'timestamp': datetime.now().isoformat(),
        'max_samples': max_samples,
        'ablation_results': ablation_results,
        'baseline': baseline['config_name']
    }


def main():
    parser = argparse.ArgumentParser(description='Prototype library ablation study')
    
    parser.add_argument('--dataset', type=str, default='GSM8K',
                       help='Dataset to analyze (default: GSM8K)')
    parser.add_argument('--configs', type=str, default='minimal,core,full',
                       help='Comma-separated list of configs to test (default: minimal,core,full)')
    parser.add_argument('--samples', type=int, default=None,
                       help='Maximum samples to test (default: all)')
    parser.add_argument('--output-dir', type=str, default='results/prototype_ablation',
                       help='Output directory for results')
    parser.add_argument('--all-datasets', action='store_true',
                       help='Run ablation on all available datasets')
    
    args = parser.parse_args()
    
    # Parse config names
    config_names = [c.strip() for c in args.configs.split(',')]
    
    # Validate configs
    for config in config_names:
        if config not in PROTOTYPE_CONFIGS:
            logger.error(f"Unknown config: {config}")
            logger.error(f"Available configs: {', '.join(PROTOTYPE_CONFIGS.keys())}")
            sys.exit(1)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine datasets to process
    if args.all_datasets:
        datasets = [f.stem for f in Path('candidates').glob('*.json')]
    else:
        datasets = [args.dataset]
    
    # Run ablation for each dataset
    all_results = {}
    for dataset in datasets:
        try:
            results = compare_configs(dataset, config_names, args.samples)
            all_results[dataset] = results
            
            # Save results
            output_file = output_dir / f'{dataset}_ablation.json'
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"\n✅ Results saved to {output_file}")
            
        except FileNotFoundError as e:
            logger.warning(f"Skipping {dataset}: {e}")
        except Exception as e:
            logger.error(f"Error processing {dataset}: {e}")
    
    # Save combined results
    if len(all_results) > 1:
        combined_file = output_dir / f'all_datasets_ablation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(combined_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        logger.info(f"\n✅ Combined results saved to {combined_file}")
    
    logger.info(f"\n{'='*80}")
    logger.info("ABLATION STUDY COMPLETE")
    logger.info(f"{'='*80}")


if __name__ == '__main__':
    main()
