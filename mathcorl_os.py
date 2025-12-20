#!/usr/bin/env python3
"""
MathCoRL-OS - Open-Source Mathematical Reasoning

Unified interface for open-source models (DeepSeek-R1, Qwen2.5-Math) with:
- Zero-shot prompting
- Few-shot with random examples
- FPP + Policy Network (full pipeline)

Usage:
    # Test with different methods
    python mathcorl_os.py test --method zero_shot --model deepseek_r1_7b --dataset GSM8K --samples 50
    python mathcorl_os.py test --method few_shot --model qwen_math_7b --dataset GSM8K --samples 50
    python mathcorl_os.py test --method fpp_policy --model deepseek_r1_7b --dataset GSM8K --samples 50
    
    # Compare all methods
    python mathcorl_os.py compare --model deepseek_r1_7b --dataset GSM8K --samples 101
    
    # Single problem solving
    python mathcorl_os.py solve --method fpp_policy --model deepseek_r1_7b "What is 15 + 27?"
"""

import argparse
import json
import logging
import os
import sys
import random
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from mint.testing import DatasetLoader
from mint.utils import execute_code, clean_code
from mint.config import get_dataset_config, create_standardized_embedding
from mint.icrl.policy_network import PolicyNetwork
from mint.providers.huggingface_provider import DeepSeekR1Provider, QwenMathProvider
from mint.evaluation import get_tolerance_function
import torch

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


def convert_to_serializable(obj):
    """Convert non-serializable objects to JSON-serializable types."""
    if isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    else:
        return obj


class OpenSourceEvaluator:
    """Evaluate open-source models with different prompting strategies."""
    
    # Model registry
    MODELS = {
        'deepseek_r1_7b': ('deepseek', '7B'),
        'deepseek_r1_1.5b': ('deepseek', '1.5B'),
        'deepseek_r1_14b': ('deepseek', '14B'),
        'qwen_math_7b': ('qwen', '7B'),
        'qwen_math_72b': ('qwen', '72B')
    }
    
    def __init__(self, model_name: str, dataset_name: str):
        """Initialize evaluator.
        
        Args:
            model_name: Model identifier (e.g., 'deepseek_r1_7b')
            dataset_name: Dataset name (e.g., 'GSM8K')
        """
        self.model_name = model_name
        self.dataset_name = dataset_name
        
        # Validate model
        if model_name not in self.MODELS:
            raise ValueError(f"Unknown model: {model_name}. Available: {list(self.MODELS.keys())}")
        
        model_type, variant = self.MODELS[model_name]
        
        # Load provider
        print(f"🔄 Loading {model_name}...")
        if model_type == 'deepseek':
            self.provider = DeepSeekR1Provider(model_variant=variant, load_in_8bit=False)
        elif model_type == 'qwen':
            self.provider = QwenMathProvider(model_variant=variant, load_in_8bit=False)
        print(f"✅ Model loaded")
        
        # Load candidates
        candidates_file = f'candidates/{dataset_name}.json'
        if os.path.exists(candidates_file):
            with open(candidates_file) as f:
                self.candidates = json.load(f)
            print(f"✅ Loaded {len(self.candidates)} candidates")
        else:
            self.candidates = []
            print(f"⚠️ No candidates found")
        
        # Load policy network
        model_file = f'models/{dataset_name}_policy_best.pt'
        if os.path.exists(model_file):
            self.policy_network = PolicyNetwork(emb_dim=1536, hidden_dim=768)
            checkpoint = torch.load(model_file, map_location='cpu')
            if 'model_state_dict' in checkpoint:
                self.policy_network.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.policy_network.load_state_dict(checkpoint)
            self.policy_network.eval()
            print(f"✅ Loaded policy network")
        else:
            self.policy_network = None
            print(f"⚠️ Policy network not available")
        
        # Load dataset
        self.test_data = DatasetLoader.load_dataset(dataset_name)
        print(f"✅ Loaded {len(self.test_data)} test samples")
        
        # Config
        config = get_dataset_config(dataset_name)
        self.k = config.get('k', 2)
        self.tolerance_func = get_tolerance_function(dataset_name)
        print(f"✅ Tolerance function: {self.tolerance_func.__name__}")
    
    def create_zero_shot_prompt(self, question: str, context: str = "") -> str:
        """Create zero-shot prompt for code generation."""
        prompt = "You are a math expert. Solve the problem by writing Python code.\n\n"
        if context:
            prompt += f"Context: {context}\n\n"
        prompt += f"Question: {question}\n\n"
        prompt += "Write Python code to solve this problem. "
        prompt += "Store the final answer in a variable named 'result'.\n"
        prompt += "Provide ONLY the Python code inside markdown code block.\n"
        return prompt
    
    def create_few_shot_prompt(self, question: str, examples: List[Dict], context: str = "") -> str:
        """Create few-shot prompt with examples."""
        prompt = "You are a math expert. Solve problems by writing Python code.\n\n"
        
        # Add examples
        prompt += "Here are some examples:\n\n"
        for i, ex in enumerate(examples, 1):
            prompt += f"Example {i}:\n"
            if ex.get('context'):
                prompt += f"Context: {ex['context']}\n"
            prompt += f"Question: {ex['question']}\n"
            prompt += f"Code:\n```python\n{ex['code']}\n```\n"
            prompt += f"Answer: {ex['answer']}\n\n"
        
        # Add test problem
        prompt += "Now solve this problem following the same format:\n"
        if context:
            prompt += f"Context: {context}\n"
        prompt += f"Question: {question}\n\n"
        prompt += "Provide your solution as Python code in a markdown code block.\n"
        return prompt
    
    def create_fpp_prompt(self, question: str, examples: List[Dict], context: str = "") -> str:
        """Create FPP-style prompt with function prototypes and examples."""
        
        # Function prototypes (simplified version)
        prompt = """You are a math expert. Use these Python functions to solve problems:

def add(a, b): return a + b
def sub(a, b): return a - b  
def mul(a, b): return a * b
def div(a, b): return a / b if b != 0 else 0
def min_val(*args): return min(args)
def max_val(*args): return max(args)
def sum_vals(*args): return sum(args)
def mean(*args): return sum(args) / len(args) if args else 0
def round_val(x, n=0): return round(x, n)

"""
        
        # Add examples
        prompt += "Here are examples:\n\n"
        for i, ex in enumerate(examples, 1):
            prompt += f"Example {i}:\n"
            if ex.get('context'):
                prompt += f"Context: {ex['context']}\n"
            prompt += f"Question: {ex['question']}\n"
            prompt += f"Solution:\n```python\n{ex['code']}\n```\n"
            prompt += f"Answer: {ex['answer']}\n\n"
        
        # Add test problem
        prompt += "Now solve this problem using the functions above:\n"
        if context:
            prompt += f"Context: {context}\n"
        prompt += f"Question: {question}\n\n"
        prompt += "IMPORTANT: Provide ONLY the Python code solution in a markdown code block.\n"
        prompt += "Do NOT include explanations or reasoning. Just the code.\n"
        prompt += "Use the provided functions and store the final answer in 'result' variable.\n"
        return prompt
    
    def solve_problem(self, sample: Dict, method: str) -> Dict[str, Any]:
        """Solve a single problem.
        
        Args:
            sample: Problem dictionary
            method: 'zero_shot', 'few_shot', or 'fpp_policy'
            
        Returns:
            Dict with keys: is_correct, predicted, elapsed, model_output, code, error
        """
        start_time = time.time()
        
        try:
            question = sample['question']
            context = sample.get('context', '')
            ground_truth = sample['ground_truth']
            
            # Select examples and create prompt based on method
            if method == 'zero_shot':
                prompt = self.create_zero_shot_prompt(question, context)
                
            elif method == 'few_shot':
                if not self.candidates:
                    return False, None, time.time() - start_time
                examples = random.sample(self.candidates, min(self.k, len(self.candidates)))
                prompt = self.create_few_shot_prompt(question, examples, context)
                
            elif method == 'fpp_policy':
                if not self.policy_network or not self.candidates:
                    return False, None, time.time() - start_time
                
                # Policy selection
                test_embedding = create_standardized_embedding(context=context, question=question)
                with torch.no_grad():
                    problem_emb = torch.tensor(test_embedding, dtype=torch.float32).unsqueeze(0)
                    candidate_embs = torch.tensor([c['embedding'] for c in self.candidates], 
                                                 dtype=torch.float32)
                    probs = self.policy_network(problem_emb, candidate_embs)
                    selected_indices = torch.multinomial(probs, self.k, replacement=False)
                    examples = [self.candidates[i.item()] for i in selected_indices]
                
                prompt = self.create_fpp_prompt(question, examples, context)
            
            else:
                raise ValueError(f"Unknown method: {method}")
            
            # Generate with model (use more tokens for FPP)
            max_tokens = 1024 if method == 'fpp_policy' else 512
            response = self.provider.generate(prompt, max_new_tokens=max_tokens, temperature=0.0)
            
            # Extract code
            code = None
            if '```python' in response:
                code = response.split('```python')[1].split('```')[0].strip()
            elif '```' in response:
                code = response.split('```')[1].split('```')[0].strip()
            
            # If no code block found, return error
            if not code:
                return {
                    'is_correct': False,
                    'predicted': None,
                    'elapsed': time.time() - start_time,
                    'model_output': response,
                    'code': '',
                    'error': 'No code block found in model output'
                }
            
            # Execute
            result, error = execute_code(code)
            
            elapsed = time.time() - start_time
            
            # Check correctness
            is_correct = False
            if error is None and result is not None:
                if self.tolerance_func is None:
                    print(f"  ERROR: tolerance_func is None!")
                    is_correct = (result == ground_truth)
                else:
                    is_correct = self.tolerance_func(result, ground_truth)
                # Debug: Always print comparison details
                print(f"  DEBUG: result={result} ({type(result).__name__}), ground_truth={ground_truth} ({type(ground_truth).__name__}), is_correct={is_correct}, func={self.tolerance_func.__name__ if self.tolerance_func else 'None'}")
            
            return {
                'is_correct': is_correct,
                'predicted': result,
                'elapsed': elapsed,
                'model_output': response,
                'code': code,
                'error': error
            }
            
        except Exception as e:
            logger.error(f"Error solving problem: {e}")
            return {
                'is_correct': False,
                'predicted': None,
                'elapsed': time.time() - start_time,
                'model_output': '',
                'code': '',
                'error': str(e)
            }
    
    def test_method(self, method: str, n_samples: int = 100) -> Dict[str, Any]:
        """Test a single method.
        
        Args:
            method: 'zero_shot', 'few_shot', or 'fpp_policy'
            n_samples: Number of samples to test
            
        Returns:
            Results dictionary
        """
        test_samples = random.sample(self.test_data, min(n_samples, len(self.test_data)))
        
        results = {
            'method': method,
            'model': self.model_name,
            'dataset': self.dataset_name,
            'k': self.k if method != 'zero_shot' else 0,
            'n_samples': n_samples,
            'correct': 0,
            'total': 0,
            'failed': 0,
            'times': [],
            'details': []  # Store per-sample details
        }
        
        print(f"\n{'='*70}")
        print(f"Testing: {method.upper()}")
        print(f"{'='*70}")
        
        for i, sample in enumerate(test_samples, 1):
            solve_result = self.solve_problem(sample, method)
            
            print(f"\n  Sample {i}: is_correct={solve_result['is_correct']}, predicted={solve_result['predicted']}, ground_truth={sample['ground_truth']}")
            
            results['times'].append(solve_result['elapsed'])
            
            if solve_result['predicted'] is None:
                results['failed'] += 1
            elif solve_result['is_correct']:
                results['correct'] += 1
            
            results['total'] += 1
            
            # Store detailed result
            results['details'].append({
                'id': i,
                'question': sample['question'][:100] + '...' if len(sample['question']) > 100 else sample['question'],
                'ground_truth': sample['ground_truth'],
                'predicted': solve_result['predicted'],
                'is_correct': solve_result['is_correct'],
                'model_output': solve_result['model_output'],
                'code': solve_result['code'],
                'error': solve_result['error'],
                'elapsed': solve_result['elapsed']
            })
            
            # Progress
            if i % 20 == 0 or i == len(test_samples):
                acc = 100.0 * results['correct'] / results['total'] if results['total'] > 0 else 0
                avg_time = sum(results['times']) / len(results['times']) if results['times'] else 0
                print(f"  Progress: {i}/{n_samples} | Acc: {acc:.1f}% | Avg: {avg_time:.2f}s/problem")
        
        # Calculate final metrics
        results['accuracy'] = 100.0 * results['correct'] / results['total'] if results['total'] > 0 else 0.0
        results['avg_time'] = sum(results['times']) / len(results['times']) if results['times'] else 0.0
        results['success_rate'] = 100.0 * (results['total'] - results['failed']) / results['total'] if results['total'] > 0 else 0.0
        
        print(f"\n✅ Results:")
        print(f"  Accuracy: {results['accuracy']:.2f}%")
        print(f"  Correct: {results['correct']}/{results['total']}")
        print(f"  Failed: {results['failed']}")
        print(f"  Avg Time: {results['avg_time']:.2f}s")
        
        return results
    
    def compare_methods(self, n_samples: int = 100) -> Dict[str, Any]:
        """Compare all methods.
        
        Args:
            n_samples: Number of samples per method
            
        Returns:
            Comparison results
        """
        methods = ['zero_shot', 'few_shot', 'fpp_policy']
        results = {}
        
        for method in methods:
            if method == 'few_shot' and not self.candidates:
                print(f"⚠️ Skipping {method} - no candidates")
                continue
            if method == 'fpp_policy' and (not self.policy_network or not self.candidates):
                print(f"⚠️ Skipping {method} - policy network or candidates not available")
                continue
            
            results[method] = self.test_method(method, n_samples)
        
        # Summary
        print(f"\n{'='*70}")
        print(f"📊 COMPARISON SUMMARY")
        print(f"{'='*70}")
        print(f"Model: {self.model_name}")
        print(f"Dataset: {self.dataset_name}")
        print(f"Samples: {n_samples}")
        print(f"-"*70)
        
        for method, res in results.items():
            print(f"{method:15s}: {res['accuracy']:6.2f}% ({res['correct']:3d}/{res['total']:3d}) - {res['avg_time']:.2f}s/problem")
        
        print(f"{'='*70}")
        
        return {
            'model': self.model_name,
            'dataset': self.dataset_name,
            'n_samples': n_samples,
            'timestamp': datetime.now().isoformat(),
            'results': results
        }


def main():
    parser = argparse.ArgumentParser(
        description='MathCoRL-OS - Open-Source Mathematical Reasoning',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    subparsers = parser.add_subparsers(dest='command', required=True, help='Commands')
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Test a single method')
    test_parser.add_argument('--method', required=True, 
                            choices=['zero_shot', 'few_shot', 'fpp_policy'],
                            help='Method to test')
    test_parser.add_argument('--model', required=True, 
                            choices=list(OpenSourceEvaluator.MODELS.keys()),
                            help='Model to use')
    test_parser.add_argument('--dataset', required=True, choices=['GSM8K', 'TAT-QA'],
                            help='Dataset to test on')
    test_parser.add_argument('--samples', type=int, default=100,
                            help='Number of samples to test')
    test_parser.add_argument('--seed', type=int, default=42,
                            help='Random seed')
    test_parser.add_argument('--output', type=str, default=None,
                            help='Output JSON file')
    
    # Compare command
    compare_parser = subparsers.add_parser('compare', help='Compare all methods')
    compare_parser.add_argument('--model', required=True,
                               choices=list(OpenSourceEvaluator.MODELS.keys()),
                               help='Model to use')
    compare_parser.add_argument('--dataset', required=True, choices=['GSM8K', 'TAT-QA'],
                               help='Dataset to test on')
    compare_parser.add_argument('--samples', type=int, default=100,
                               help='Number of samples per method')
    compare_parser.add_argument('--seed', type=int, default=42,
                               help='Random seed')
    compare_parser.add_argument('--output', type=str, default=None,
                               help='Output JSON file')
    
    args = parser.parse_args()
    
    # Set seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    print(f"🎯 MathCoRL-OS - Open-Source Evaluation")
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Seed: {args.seed}")
    
    # Initialize evaluator
    evaluator = OpenSourceEvaluator(args.model, args.dataset)
    
    # Execute command
    if args.command == 'test':
        results = evaluator.test_method(args.method, args.samples)
        
        # Save results
        if args.output:
            output_file = args.output
        else:
            os.makedirs('results/opensource', exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = f'results/opensource/{args.dataset}_{args.model}_{args.method}_{timestamp}.json'
        
        with open(output_file, 'w') as f:
            json.dump(convert_to_serializable(results), f, indent=2)
        print(f"\n✅ Results saved to {output_file}")
    
    elif args.command == 'compare':
        results = evaluator.compare_methods(args.samples)
        
        # Save results
        if args.output:
            output_file = args.output
        else:
            os.makedirs('results/opensource', exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = f'results/opensource/{args.dataset}_{args.model}_comparison_{timestamp}.json'
        
        with open(output_file, 'w') as f:
            json.dump(convert_to_serializable(results), f, indent=2)
        print(f"\n✅ Results saved to {output_file}")


if __name__ == '__main__':
    main()
