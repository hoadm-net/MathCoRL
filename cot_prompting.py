#!/usr/bin/env python3
"""
Chain-of-Thought (CoT) Prompting Dataset Testing Tool

This script provides comprehensive testing capabilities for mathematical datasets
using Chain-of-Thought prompting methodology. It supports multiple datasets with
dataset-specific tolerance functions and provides detailed accuracy metrics.

Usage:
    python cot_prompting.py DATASET [--limit N] [--verbose] [--output DIR] [--no-save]

Examples:
    python cot_prompting.py SVAMP --limit 50
    python cot_prompting.py GSM8K --limit 100 -v
    python cot_prompting.py FinQA --limit 100 --output cot_results
"""

import argparse
import json
import os
import sys
from typing import Dict, Any, List
from tqdm import tqdm
from mint.cot import ChainOfThoughtPrompting
from mint.utils import (
    load_svamp_test_data, load_gsm8k_test_data, load_tabmwp_test_data, 
    load_tatqa_test_data, load_finqa_test_data, FinQA_generate_candidates
)


def load_dataset(dataset_name: str) -> list:
    """Load dataset by name."""
    if dataset_name.upper() == 'SVAMP':
        return load_svamp_test_data('datasets/SVAMP/test.json')
    elif dataset_name.upper() == 'GSM8K':
        return load_gsm8k_test_data('datasets/GSM8K/test.jsonl')
    elif dataset_name.upper() == 'TABMWP':
        return load_tabmwp_test_data('datasets/TabMWP/test.json')
    elif dataset_name.upper() == 'TAT-QA' or dataset_name.upper() == 'TATQA':
        return load_tatqa_test_data('datasets/TAT-QA/test.json')
    elif dataset_name.upper() == 'FINQA' or dataset_name.upper() == 'FIN-QA':
        return load_finqa_test_data('datasets/FinQA/test.json')
    else:
        raise ValueError(f"Dataset {dataset_name} not supported. Available: SVAMP, GSM8K, TabMWP, TAT-QA, FinQA")


def solve_single_cot(question: str, context: str = "") -> Dict[str, Any]:
    """
    Solve a single problem using Chain-of-Thought prompting.
    
    Args:
        question: The question to solve
        context: Additional context for the problem
        
    Returns:
        Dictionary with result and reasoning
    """
    cot = ChainOfThoughtPrompting()
    result = cot.solve_silent(question, context)
    
    return {
        'result': result.get('result'),
        'reasoning': result.get('reasoning', ''),
        'method': 'Chain-of-Thought'
    }


def test_dataset(dataset_name, limit=None, verbose=False, output_dir="results", save_results=True):
    """
    Test a dataset using Chain-of-Thought prompting.
    
    Args:
        dataset_name: Name of the dataset to test
        limit: Maximum number of samples to test
        verbose: Whether to show detailed output
        output_dir: Directory to save results
        save_results: Whether to save results to file
        
    Returns:
        Dictionary with test summary
    """
    print(f"🚀 CoT Dataset Testing Tool")
    print(f"Dataset: {dataset_name}")
    
    if limit:
        print(f"Sample limit: {limit}")
    
    # Load dataset
    try:
        data = load_dataset(dataset_name)
        print(f"Loaded {len(data)} samples from {dataset_name}")
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return None
    
    # Apply limit
    if limit:
        data = data[:limit]
        print(f"Limited to first {limit} samples")
    
    # Get appropriate tolerance function
    tolerance_func = get_tolerance_function(dataset_name)
    
    # Test samples
    results = []
    correct_count = 0
    
    print(f"\n🧠 Testing with Chain-of-Thought prompting...")
    
    with tqdm(total=len(data), desc="Processing") as pbar:
        for i, sample in enumerate(data):
            question = sample['question']
            context = sample.get('context', '')
            ground_truth = sample['ground_truth']
            
            if verbose:
                print(f"\n📝 Sample {i+1}/{len(data)}")
                print(f"Question: {question}")
                if context:
                    print(f"Context: {context[:200]}...")
                print(f"Ground Truth: {ground_truth}")
            
            # Solve using CoT
            try:
                solution = solve_single_cot(question, context)
                result = solution['result']
                reasoning = solution['reasoning']
                
                # Check if answer is correct using tolerance function
                is_correct = tolerance_func(result, ground_truth)
                if is_correct:
                    correct_count += 1
                
                if verbose:
                    print(f"CoT Result: {result}")
                    print(f"Reasoning: {reasoning[:300]}...")
                    print(f"✅ Correct" if is_correct else "❌ Incorrect")
                
                results.append({
                    'question': question,
                    'context': context,
                    'ground_truth': ground_truth,
                    'result': result,
                    'reasoning': reasoning,
                    'correct': is_correct
                })
                
            except Exception as e:
                if verbose:
                    print(f"❌ Error: {e}")
                
                results.append({
                    'question': question,
                    'context': context,
                    'ground_truth': ground_truth,
                    'result': None,
                    'reasoning': f"Error: {str(e)}",
                    'correct': False
                })
            
            pbar.update(1)
    
    # Calculate metrics
    accuracy = (correct_count / len(data)) * 100 if data else 0
    
    summary = {
        'dataset': dataset_name,
        'method': 'Chain-of-Thought',
        'total_samples': len(data),
        'correct_predictions': correct_count,
        'accuracy': accuracy,
        'results': results
    }
    
    # Print summary
    print_summary(summary)
    
    # Save results
    if save_results:
        filepath = save_results_func(summary, output_dir)
        print(f"\n💾 Results saved to: {filepath}")
    
    return summary


def save_results_func(summary: Dict[str, Any], output_dir: str = 'results') -> str:
    """Save test results to JSON file."""
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Create filename
    dataset_name = summary['dataset'].lower()
    total = summary['total_samples']
    filename = f"{dataset_name}_cot_results_{total}samples.json"
    filepath = os.path.join(output_dir, filename)
    
    # Save results
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)
    
    return filepath


def print_summary(summary: Dict[str, Any]):
    """Print test summary."""
    print(f"\n{'='*50}")
    print(f"CoT Test Results for {summary['dataset']} Dataset")
    print(f"{'='*50}")
    print(f"Method: {summary['method']}")
    print(f"Total samples tested: {summary['total_samples']}")
    print(f"Correct predictions: {summary['correct_predictions']}")
    print(f"Accuracy: {summary['accuracy']:.2f}%")
    print(f"{'='*50}")


def is_close(a, b, tolerance=1e-3):
    """
    Check if two numbers are approximately equal within a tolerance.
    
    Args:
        a: First number
        b: Second number  
        tolerance: Maximum allowed difference (default: 1e-3)
        
    Returns:
        True if numbers are approximately equal, False otherwise
    """
    if a is None or b is None:
        return a == b
    
    # Handle the case where both are the same (including both being 0)
    if a == b:
        return True
        
    # Convert to float if possible
    try:
        a_float = float(a)
        b_float = float(b)
        
        # Use absolute tolerance for small numbers and relative tolerance for large numbers
        return abs(a_float - b_float) <= max(tolerance, tolerance * max(abs(a_float), abs(b_float)))
    except (ValueError, TypeError):
        # Fallback to exact comparison for non-numeric values
        return a == b


def is_close_tatqa(a, b, tolerance=0.05):
    """
    Special comparison function for TAT-QA dataset that handles rounding.
    TAT-QA often rounds results to 2 decimal places, so we need more flexible tolerance.
    
    Args:
        a: First number
        b: Second number  
        tolerance: Maximum allowed difference (default: 0.05 for ~2% tolerance)
        
    Returns:
        True if numbers are approximately equal considering TAT-QA rounding, False otherwise
    """
    if a is None or b is None:
        return a == b
    
    # Handle the case where both are the same
    if a == b:
        return True
        
    # Convert to float if possible
    try:
        a_float = float(a)
        b_float = float(b)
        
        # For TAT-QA, check if the difference is within tolerance
        # Also check if one could be a rounded version of the other
        diff = abs(a_float - b_float)
        
        # Basic tolerance check
        if diff <= tolerance:
            return True
            
        # Check if b could be a rounded version of a (common in TAT-QA)
        # Round a to different decimal places and see if it matches b
        for decimal_places in [0, 1, 2, 3]:
            if abs(round(a_float, decimal_places) - b_float) <= 0.001:
                return True
            if abs(a_float - round(b_float, decimal_places)) <= 0.001:
                return True
        
        # For percentage calculations, allow larger relative tolerance
        max_val = max(abs(a_float), abs(b_float))
        if max_val > 0:
            relative_diff = diff / max_val
            if relative_diff <= 0.02:  # 2% relative tolerance
                return True
                
        return False
        
    except (ValueError, TypeError):
        # Fallback to exact comparison for non-numeric values
        return a == b


def is_close_finqa(result, ground_truth):
    """
    Special comparison function for FinQA dataset that handles semantic equivalence.
    FinQA has very inconsistent rounding and percentage formats, so we use candidate
    generation to check if values are semantically equivalent (e.g., 0.92 ~ 92%).
    
    Args:
        result: The calculated result
        ground_truth: The expected answer
        
    Returns:
        True if values are semantically equivalent, False otherwise
    """
    if result is None or ground_truth is None:
        return result == ground_truth
    
    # Handle the case where both are the same
    if result == ground_truth:
        return True
        
    try:
        result_float = float(result)
        ground_truth_float = float(ground_truth)
        
        # Generate candidates for both result and ground truth
        result_candidates = FinQA_generate_candidates(result_float)
        ground_truth_candidates = FinQA_generate_candidates(ground_truth_float)
        
        # Check if any candidate from result matches any candidate from ground truth
        for r_candidate in result_candidates:
            for gt_candidate in ground_truth_candidates:
                # Use small tolerance for floating point comparison
                if abs(r_candidate - gt_candidate) <= 0.01:
                    return True
        
        # Also check direct candidate matching (result vs ground_truth candidates)
        for gt_candidate in ground_truth_candidates:
            if abs(result_float - gt_candidate) <= 0.01:
                return True
                
        # And vice versa (ground_truth vs result candidates)
        for r_candidate in result_candidates:
            if abs(ground_truth_float - r_candidate) <= 0.01:
                return True
                
        return False
        
    except (ValueError, TypeError):
        # Fallback to exact comparison for non-numeric values
        return result == ground_truth


def get_tolerance_function(dataset_name: str):
    """
    Get the appropriate tolerance function based on dataset.
    
    Args:
        dataset_name: Name of the dataset
        
    Returns:
        Tolerance function to use for comparison
    """
    if dataset_name.upper() in ['TAT-QA', 'TATQA']:
        return is_close_tatqa
    elif dataset_name.upper() in ['FINQA', 'FIN-QA']:
        return is_close_finqa
    else:
        return is_close


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Chain-of-Thought (CoT) Prompting Dataset Testing Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cot_prompting.py SVAMP                    # Test full SVAMP dataset
  python cot_prompting.py SVAMP --limit 100        # Test first 100 samples
  python cot_prompting.py SVAMP --limit 50 -v      # Test 50 samples with verbose output
  python cot_prompting.py SVAMP --output results   # Save to specific directory
        """
    )
    
    parser.add_argument(
        'dataset',
        help='Dataset name to test (e.g., SVAMP)'
    )
    
    parser.add_argument(
        '--limit', '-l',
        type=int,
        default=None,
        help='Maximum number of samples to test (default: test all samples)'
    )
    
    parser.add_argument(
        '--output', '-o',
        default='results',
        help='Output directory for results (default: results)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Show verbose output'
    )
    
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Do not save results to file'
    )
    
    args = parser.parse_args()
    
    try:
        # Test dataset
        summary = test_dataset(
            dataset_name=args.dataset,
            limit=args.limit,
            verbose=args.verbose,
            output_dir=args.output,
            save_results=not args.no_save
        )
        
    except ValueError as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print(f"\n⚠️  Testing interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 