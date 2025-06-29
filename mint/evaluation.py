"""
Evaluation utilities for mathematical problem solving.

This module provides dataset-specific tolerance functions and evaluation metrics
that are shared across different prompting methods (FPP, CoT, etc.).
"""

import logging
import math
from typing import Any, Callable, Union

logger = logging.getLogger(__name__)


def is_close(a: Any, b: Any, tolerance: float = 1e-3) -> bool:
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


def is_close_tatqa(a: Any, b: Any, tolerance: float = 0.05) -> bool:
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
    
    if a == b:
        return True
    
    try:
        a_float = float(a)
        b_float = float(b)
        
        # Handle percentage values (convert if one is percentage form)
        if abs(a_float) > 1 and abs(b_float) < 1:
            b_float *= 100
        elif abs(b_float) > 1 and abs(a_float) < 1:
            a_float *= 100
        
        # Use TAT-QA specific tolerance
        diff = abs(a_float - b_float)
        
        # For small numbers, use absolute tolerance
        if max(abs(a_float), abs(b_float)) < 1:
            return diff <= tolerance
        
        # For larger numbers, use relative tolerance
        relative_tolerance = tolerance * max(abs(a_float), abs(b_float))
        return diff <= max(tolerance, relative_tolerance)
        
    except (ValueError, TypeError):
        return False


def is_close_finqa(result: Any, ground_truth: Any) -> bool:
    """
    FinQA-specific evaluation function with candidate generation.
    
    FinQA uses semantic equivalence checking with multiple candidate formats
    (e.g., 0.92 ≈ 92%, 920 basis points, etc.)
    
    Args:
        result: Predicted result
        ground_truth: Ground truth value
        
    Returns:
        True if semantically equivalent, False otherwise
    """
    if result is None or ground_truth is None:
        return result == ground_truth
    
    try:
        # Import the candidate generation function
        from .utils import FinQA_generate_candidates
        
        # Convert to float
        result_float = float(result)
        ground_truth_float = float(ground_truth)
        
        # Direct comparison first
        if abs(result_float - ground_truth_float) < 1e-3:
            return True
        
        # Generate candidates for ground truth and check if result matches any
        candidates = FinQA_generate_candidates(ground_truth_float)
        
        for candidate in candidates:
            if abs(result_float - candidate) < 1e-3:
                return True
        
        # Generate candidates for result and check if ground truth matches any
        result_candidates = FinQA_generate_candidates(result_float)
        
        for candidate in result_candidates:
            if abs(candidate - ground_truth_float) < 1e-3:
                return True
        
        return False
        
    except (ValueError, TypeError, ImportError):
        # Fallback to string comparison
        return str(result).strip() == str(ground_truth).strip()


def get_tolerance_function(dataset_name: str) -> Callable[[Any, Any], bool]:
    """
    Get the appropriate tolerance function for a dataset.
    
    Args:
        dataset_name: Name of the dataset (case-insensitive)
        
    Returns:
        Tolerance function for the dataset
    """
    dataset_upper = dataset_name.upper()
    
    if dataset_upper in ['TAT-QA', 'TATQA']:
        return is_close_tatqa
    elif dataset_upper in ['FINQA', 'FIN-QA']:
        return is_close_finqa
    else:
        # Default tolerance for SVAMP, GSM8K, TabMWP
        return is_close


def calculate_accuracy(results: list) -> dict:
    """
    Calculate accuracy metrics from evaluation results.
    
    Args:
        results: List of result dictionaries with 'correct' key
        
    Returns:
        Dictionary with accuracy metrics
    """
    if not results:
        return {
            'total_samples': 0,
            'correct_predictions': 0,
            'accuracy': 0.0,
            'error_rate': 0.0
        }
    
    total_samples = len(results)
    correct_predictions = sum(1 for r in results if r.get('correct', False))
    accuracy = (correct_predictions / total_samples) * 100 if total_samples > 0 else 0.0
    
    return {
        'total_samples': total_samples,
        'correct_predictions': correct_predictions,
        'accuracy': accuracy,
        'error_rate': 100.0 - accuracy
    }


def evaluate_predictions(predictions: list, ground_truths: list, dataset_name: str) -> dict:
    """
    Evaluate a list of predictions against ground truths.
    
    Args:
        predictions: List of predicted values
        ground_truths: List of ground truth values
        dataset_name: Name of the dataset for appropriate tolerance function
        
    Returns:
        Dictionary with evaluation results
    """
    if len(predictions) != len(ground_truths):
        raise ValueError("Predictions and ground truths must have the same length")
    
    tolerance_func = get_tolerance_function(dataset_name)
    
    results = []
    for pred, truth in zip(predictions, ground_truths):
        is_correct = tolerance_func(pred, truth)
        results.append({
            'prediction': pred,
            'ground_truth': truth,
            'correct': is_correct
        })
    
    metrics = calculate_accuracy(results)
    metrics['results'] = results
    metrics['dataset'] = dataset_name
    
    return metrics 