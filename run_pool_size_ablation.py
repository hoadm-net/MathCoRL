#!/usr/bin/env python3
"""
Pool Size Ablation Study

Tests policy network performance with different candidate pool sizes
to find optimal balance between diversity and computational cost.

Usage:
    python run_pool_size_ablation.py --dataset GSM8K --sizes 10 20 30 50
    python run_pool_size_ablation.py --dataset TAT-QA --sizes 10 30 75 100
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import random

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from mint.testing import DatasetLoader, create_fpp_solver, get_tolerance_function
from mint.icrl.policy_network import PolicyNetwork
from mint.icrl.evaluator import PolicyNetworkEvaluator
from mint.prompts import create_fpp_with_examples_prompt
from mint.utils import clean_code, execute_code
from mint.config import create_standardized_embedding
from openai import OpenAI
import torch

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


class PoolSizeAblation:
    """Ablation study for candidate pool size."""
    
    def __init__(self, dataset_name: str):
        """Initialize ablation study.
        
        Args:
            dataset_name: Dataset to test (GSM8K, TAT-QA)
        """
        self.dataset_name = dataset_name
        
        # Load candidates
        candidates_file = f'candidates/{dataset_name}.json'
        if not os.path.exists(candidates_file):
            raise FileNotFoundError(f"Candidates not found: {candidates_file}")
        
        with open(candidates_file) as f:
            self.all_candidates = json.load(f)
        
        logger.info(f"Loaded {len(self.all_candidates)} candidates for {dataset_name}")
        
        # Load policy network
        model_file = f'models/{dataset_name}_policy_best.pt'
        if not os.path.exists(model_file):
            raise FileNotFoundError(f"Policy model not found: {model_file}")
        
        self.policy_network = PolicyNetwork(emb_dim=1536, hidden_dim=768)
        checkpoint = torch.load(model_file, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            self.policy_network.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.policy_network.load_state_dict(checkpoint)
        self.policy_network.eval()
        
        logger.info(f"Loaded policy network from {model_file}")
        
        # Initialize evaluator
        api_key = os.getenv('OPENAI_API_KEY')
        openai_client = OpenAI(api_key=api_key)
        self.evaluator = PolicyNetworkEvaluator(openai_client, 'gpt-4.1-mini')
        
        # Load dataset
        self.test_data = DatasetLoader.load_dataset(dataset_name)
        self.tolerance_func = get_tolerance_function(dataset_name)
        
        # Get k from config
        from mint.config import get_dataset_config
        config = get_dataset_config(dataset_name)
        self.k = config.get('k', 2)
        
        logger.info(f"Dataset: {dataset_name}, k={self.k}, test samples: {len(self.test_data)}")
    
    def test_with_pool_size(self, pool_size: int, n_samples: int = 50) -> Dict[str, Any]:
        """Test policy network with specific pool size.
        
        Args:
            pool_size: Number of candidates in pool
            n_samples: Number of test samples to evaluate
            
        Returns:
            Results dictionary with accuracy, success rate, etc.
        """
        if pool_size > len(self.all_candidates):
            logger.warning(f"Pool size {pool_size} > available candidates {len(self.all_candidates)}")
            pool_size = len(self.all_candidates)
        
        # Sample test problems
        test_samples = random.sample(self.test_data, min(n_samples, len(self.test_data)))
        
        results = {
            'pool_size': pool_size,
            'n_samples': n_samples,
            'correct': 0,
            'total': 0,
            'failed': 0,
            'details': []
        }
        
        print(f"\n{'='*60}")
        print(f"Testing pool_size={pool_size} on {n_samples} samples")
        print(f"{'='*60}")
        
        for i, sample in enumerate(test_samples, 1):
            try:
                # Create embedding
                test_embedding = create_standardized_embedding(
                    context=sample.get('context', ''),
                    question=sample['question']
                )
                
                problem_dict = {
                    'question': sample['question'],
                    'context': sample.get('context', ''),
                    'answer': sample['ground_truth'],
                    'embedding': test_embedding
                }
                
                # Randomly sample pool_size candidates
                candidate_pool = random.sample(self.all_candidates, pool_size)
                
                # Select examples with policy
                selected_examples = self.evaluator.select_with_policy(
                    policy_net=self.policy_network,
                    problem=problem_dict,
                    candidate_pool=candidate_pool,
                    k=self.k
                )
                
                if len(selected_examples) < self.k:
                    results['failed'] += 1
                    results['total'] += 1
                    continue
                
                # Create FPP solver
                from mint.core import FunctionPrototypePrompting
                fpp_instance = FunctionPrototypePrompting()
                
                # Create prompt with selected examples
                prompt = create_fpp_with_examples_prompt(
                    question=sample['question'],
                    examples=selected_examples,
                    context=sample.get('context', '')
                )
                
                # Generate solution
                response = fpp_instance._call_llm(prompt)
                if not response:
                    results['failed'] += 1
                    results['total'] += 1
                    continue
                
                # Execute code
                cleaned_code = clean_code(response)
                result, error = execute_code(cleaned_code)
                
                if error or result is None:
                    results['failed'] += 1
                    results['total'] += 1
                    continue
                
                # Check correctness
                is_correct = self.tolerance_func(result, sample['ground_truth'])
                
                if is_correct:
                    results['correct'] += 1
                
                results['total'] += 1
                
                # Progress
                if i % 10 == 0:
                    acc = 100.0 * results['correct'] / results['total'] if results['total'] > 0 else 0
                    print(f"  Progress: {i}/{n_samples} | Accuracy: {acc:.1f}%")
                
            except Exception as e:
                logger.error(f"Sample {i} failed: {e}")
                results['failed'] += 1
                results['total'] += 1
                continue
        
        # Calculate final metrics
        results['accuracy'] = 100.0 * results['correct'] / results['total'] if results['total'] > 0 else 0.0
        results['success_rate'] = 100.0 * (results['total'] - results['failed']) / results['total'] if results['total'] > 0 else 0.0
        
        print(f"\nResults for pool_size={pool_size}:")
        print(f"  Accuracy: {results['accuracy']:.2f}%")
        print(f"  Success Rate: {results['success_rate']:.2f}%")
        print(f"  Correct: {results['correct']}/{results['total']}")
        
        return results
    
    def run_ablation(self, pool_sizes: List[int], n_samples: int = 50) -> Dict[str, Any]:
        """Run ablation study across multiple pool sizes.
        
        Args:
            pool_sizes: List of pool sizes to test
            n_samples: Number of test samples per size
            
        Returns:
            Complete ablation results
        """
        ablation_results = {
            'dataset': self.dataset_name,
            'k': self.k,
            'n_samples': n_samples,
            'timestamp': datetime.now().isoformat(),
            'pool_sizes': pool_sizes,
            'results': []
        }
        
        for pool_size in pool_sizes:
            result = self.test_with_pool_size(pool_size, n_samples)
            ablation_results['results'].append(result)
        
        return ablation_results


def main():
    parser = argparse.ArgumentParser(description='Pool Size Ablation Study')
    parser.add_argument('--dataset', required=True, choices=['GSM8K', 'TAT-QA'],
                       help='Dataset to test')
    parser.add_argument('--sizes', nargs='+', type=int, default=[10, 20, 30, 50, 75, 100],
                       help='Pool sizes to test')
    parser.add_argument('--samples', type=int, default=50,
                       help='Number of test samples per size')
    parser.add_argument('--output', type=str, default=None,
                       help='Output JSON file')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Set seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    print(f"🎯 Pool Size Ablation Study")
    print(f"Dataset: {args.dataset}")
    print(f"Pool sizes: {args.sizes}")
    print(f"Samples per size: {args.samples}")
    print(f"Seed: {args.seed}")
    
    # Run ablation
    study = PoolSizeAblation(args.dataset)
    results = study.run_ablation(args.sizes, args.samples)
    
    # Save results
    if args.output:
        output_file = args.output
    else:
        os.makedirs('results/ablation', exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f'results/ablation/{args.dataset}_pool_size_{timestamp}.json'
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to {output_file}")
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"SUMMARY - {args.dataset} Pool Size Ablation")
    print(f"{'='*60}")
    for result in results['results']:
        print(f"Pool Size {result['pool_size']:3d}: {result['accuracy']:6.2f}% accuracy ({result['correct']}/{result['total']})")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
