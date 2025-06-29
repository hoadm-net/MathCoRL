"""
Unified testing framework for mathematical problem solving methods.

This module provides a common testing interface for different prompting methods
(FPP, CoT, etc.) with support for multiple datasets and evaluation metrics.
"""

import json
import os
import logging
from typing import Dict, Any, List, Optional, Union, Callable
from tqdm import tqdm
from datetime import datetime

from .evaluation import get_tolerance_function, calculate_accuracy
from .utils import (
    load_svamp_test_data, load_gsm8k_test_data, load_tabmwp_test_data,
    load_tatqa_test_data, load_finqa_test_data
)

logger = logging.getLogger(__name__)


class DatasetLoader:
    """Centralized dataset loading functionality."""
    
    SUPPORTED_DATASETS = ['SVAMP', 'GSM8K', 'TABMWP', 'TAT-QA', 'TATQA', 'FINQA', 'FIN-QA']
    
    @staticmethod
    def load_dataset(dataset_name: str) -> List[Dict[str, Any]]:
        """
        Load dataset by name.
        
        Args:
            dataset_name: Name of the dataset to load
            
        Returns:
            List of problem dictionaries
            
        Raises:
            ValueError: If dataset is not supported
        """
        dataset_upper = dataset_name.upper()
        
        if dataset_upper == 'SVAMP':
            return load_svamp_test_data('datasets/SVAMP/test.json')
        elif dataset_upper == 'GSM8K':
            return load_gsm8k_test_data('datasets/GSM8K/test.jsonl')
        elif dataset_upper == 'TABMWP':
            return load_tabmwp_test_data('datasets/TabMWP/test.json')
        elif dataset_upper in ['TAT-QA', 'TATQA']:
            return load_tatqa_test_data('datasets/TAT-QA/test.json')
        elif dataset_upper in ['FINQA', 'FIN-QA']:
            return load_finqa_test_data('datasets/FinQA/test.json')
        else:
            supported = ', '.join(DatasetLoader.SUPPORTED_DATASETS)
            raise ValueError(f"Dataset '{dataset_name}' not supported. Available: {supported}")
    
    @staticmethod
    def get_supported_datasets() -> List[str]:
        """Get list of supported dataset names."""
        return DatasetLoader.SUPPORTED_DATASETS.copy()


class TestRunner:
    """Unified testing framework for different prompting methods."""
    
    def __init__(self, method_name: str, solver_func: Callable):
        """
        Initialize test runner.
        
        Args:
            method_name: Name of the prompting method (e.g., 'FPP', 'CoT')
            solver_func: Function that solves individual problems
        """
        self.method_name = method_name
        self.solver_func = solver_func
    
    def test_dataset(self, 
                    dataset_name: str,
                    limit: Optional[int] = None,
                    verbose: bool = False,
                    output_dir: str = "results",
                    save_results: bool = True) -> Dict[str, Any]:
        """
        Test the solver on a specific dataset.
        
        Args:
            dataset_name: Name of the dataset to test
            limit: Maximum number of samples to test
            verbose: Whether to show detailed output
            output_dir: Directory to save results
            save_results: Whether to save results to file
            
        Returns:
            Dictionary with test results and metrics
        """
        print(f"🚀 {self.method_name} Dataset Testing")
        print(f"Dataset: {dataset_name}")
        
        if limit:
            print(f"Sample limit: {limit}")
        
        # Load dataset
        try:
            data = DatasetLoader.load_dataset(dataset_name)
            print(f"Loaded {len(data)} samples from {dataset_name}")
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            return {'error': str(e)}
        
        # Apply limit
        if limit:
            data = data[:limit]
            print(f"Limited to first {limit} samples")
        
        # Get appropriate tolerance function
        tolerance_func = get_tolerance_function(dataset_name)
        
        # Test samples
        results = []
        correct_count = 0
        
        print(f"\n🧠 Testing with {self.method_name}...")
        
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
                
                # Solve using the provided solver function
                try:
                    solution = self.solver_func(question, context)
                    result = solution.get('result') if isinstance(solution, dict) else solution
                    
                    # Check if answer is correct using tolerance function
                    is_correct = tolerance_func(result, ground_truth)
                    if is_correct:
                        correct_count += 1
                    
                    if verbose:
                        print(f"{self.method_name} Result: {result}")
                        if isinstance(solution, dict) and 'reasoning' in solution:
                            print(f"Reasoning: {solution['reasoning'][:300]}...")
                        print(f"✅ Correct" if is_correct else "❌ Incorrect")
                    
                    result_dict = {
                        'question': question,
                        'context': context,
                        'ground_truth': ground_truth,
                        'result': result,
                        'correct': is_correct
                    }
                    
                    # Add method-specific fields
                    if isinstance(solution, dict):
                        result_dict.update({k: v for k, v in solution.items() 
                                          if k not in ['question', 'context', 'result']})
                    
                    results.append(result_dict)
                    
                except Exception as e:
                    if verbose:
                        print(f"❌ Error: {e}")
                    
                    results.append({
                        'question': question,
                        'context': context,
                        'ground_truth': ground_truth,
                        'result': None,
                        'error': str(e),
                        'correct': False
                    })
                
                pbar.update(1)
        
        # Calculate metrics
        metrics = calculate_accuracy(results)
        
        summary = {
            'dataset': dataset_name,
            'method': self.method_name,
            'timestamp': datetime.now().isoformat(),
            **metrics,
            'results': results
        }
        
        # Print summary
        self._print_summary(summary)
        
        # Save results
        if save_results:
            filepath = self._save_results(summary, output_dir)
            print(f"\n💾 Results saved to: {filepath}")
        
        return summary
    
    def _print_summary(self, summary: Dict[str, Any]):
        """Print test summary."""
        print(f"\n{'='*50}")
        print(f"Test Results for {summary['dataset']} Dataset")
        print(f"Method: {summary['method']}")
        print(f"{'='*50}")
        print(f"Total samples tested: {summary['total_samples']}")
        print(f"Correct predictions: {summary['correct_predictions']}")
        print(f"Accuracy: {summary['accuracy']:.2f}%")
        if 'error_rate' in summary:
            print(f"Error rate: {summary['error_rate']:.2f}%")
        print(f"{'='*50}")
    
    def _save_results(self, summary: Dict[str, Any], output_dir: str = 'results') -> str:
        """Save test results to JSON file."""
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Create filename
        dataset_name = summary['dataset'].lower()
        method_name = summary['method'].lower()
        total = summary['total_samples']
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{dataset_name}_{method_name}_results_{total}samples_{timestamp}.json"
        filepath = os.path.join(output_dir, filename)
        
        # Save results
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=4, ensure_ascii=False)
        
        return filepath
    
    def compare_methods(self, 
                       other_runner: 'TestRunner',
                       dataset_name: str,
                       limit: Optional[int] = None) -> Dict[str, Any]:
        """
        Compare this method with another method on the same dataset.
        
        Args:
            other_runner: Another TestRunner instance
            dataset_name: Name of the dataset to test
            limit: Maximum number of samples to test
            
        Returns:
            Comparison results
        """
        print(f"🔄 Comparing {self.method_name} vs {other_runner.method_name}")
        
        # Run both methods
        results_1 = self.test_dataset(dataset_name, limit, save_results=False)
        results_2 = other_runner.test_dataset(dataset_name, limit, save_results=False)
        
        # Create comparison
        comparison = {
            'dataset': dataset_name,
            'methods': [self.method_name, other_runner.method_name],
            'comparison': {
                self.method_name: {
                    'accuracy': results_1['accuracy'],
                    'correct': results_1['correct_predictions'],
                    'total': results_1['total_samples']
                },
                other_runner.method_name: {
                    'accuracy': results_2['accuracy'],
                    'correct': results_2['correct_predictions'],
                    'total': results_2['total_samples']
                }
            },
            'winner': self.method_name if results_1['accuracy'] > results_2['accuracy'] 
                     else other_runner.method_name,
            'detailed_results': {
                self.method_name: results_1,
                other_runner.method_name: results_2
            }
        }
        
        # Print comparison
        print(f"\n📊 Comparison Results:")
        print(f"{self.method_name}: {results_1['accuracy']:.2f}% accuracy")
        print(f"{other_runner.method_name}: {results_2['accuracy']:.2f}% accuracy")
        print(f"Winner: {comparison['winner']} 🏆")
        
        return comparison


def create_fpp_solver():
    """Create FPP solver function."""
    def solve_fpp(question: str, context: str = "") -> Dict[str, Any]:
        from .core import FunctionPrototypePrompting
        fpp = FunctionPrototypePrompting()
        return fpp.solve_detailed(question, context)
    return solve_fpp


def create_cot_solver():
    """Create CoT solver function."""
    def solve_cot(question: str, context: str = "") -> Dict[str, Any]:
        from .cot import ChainOfThoughtPrompting
        cot = ChainOfThoughtPrompting()
        return cot.solve_silent(question, context)
    return solve_cot 