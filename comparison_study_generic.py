#!/usr/bin/env python3
"""
Generic Comparison Study: FPP vs FPP+Random vs FPP+Policy Network

This script can work with any dataset by passing the dataset name as parameter.
Usage: python comparison_study_generic.py --dataset GSM8K --samples 10
"""

import os
import json
import random
import argparse
from typing import Dict, List, Tuple, Any
from pathlib import Path
import sys
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Official imports for consistency
from mint.testing import DatasetLoader, create_fpp_solver, get_tolerance_function
from mint.core import FunctionPrototypePrompting
from mint.prompts import create_fpp_with_examples_prompt
from mint.icrl.candidate_generator import CandidateGenerator
from mint.icrl.policy_network import PolicyNetwork
from mint.icrl.evaluator import PolicyNetworkEvaluator
from mint.utils import execute_code, clean_code
from mint.config import create_standardized_embedding, load_config
from openai import OpenAI


class GenericComparisonStudy:
    """Generic comparison study that works with any dataset."""
    
    # Dataset configurations
    DATASET_CONFIGS = {
        'GSM8K': {
            'display_name': 'GSM8K (Grade School Math)',
            'candidates_file': 'candidates/GSM8K.json',
            'model_file': 'models/GSM8K_policy_best.pt',
            'k': 2,
            'description': 'Elementary arithmetic word problems'
        },
        'SVAMP': {
            'display_name': 'SVAMP (Simple Arithmetic Word Problems)',
            'candidates_file': 'candidates/SVAMP.json', 
            'model_file': 'models/SVAMP_policy_best.pt',
            'k': 2,
            'description': 'Arithmetic word problems with simple variations'
        },
        'TabMWP': {
            'display_name': 'TabMWP (Tabular Math Word Problems)',
            'candidates_file': 'candidates/TabMWP.json',
            'model_file': 'models/TabMWP_policy_best.pt', 
            'k': 2,
            'description': 'Math problems involving tables and charts'
        },
        'TAT-QA': {
            'display_name': 'TAT-QA (Table-and-Text QA)',
            'candidates_file': 'candidates/TAT-QA.json',
            'model_file': 'models/TAT-QA_policy_best.pt',
            'k': 3,
            'description': 'Financial reasoning with tables and text'
        },
        'FinQA': {
            'display_name': 'FinQA (Financial QA)',
            'candidates_file': 'candidates/FinQA.json',
            'model_file': 'models/FinQA_policy_best.pt',
            'k': 2,
            'description': 'Financial reasoning and calculations'
        }
    }
    
    def __init__(self, dataset_name: str):
        """
        Initialize comparison study for specified dataset.
        
        Args:
            dataset_name: Name of dataset (GSM8K, SVAMP, TabMWP, TAT-QA, FinQA)
        """
        if dataset_name not in self.DATASET_CONFIGS:
            raise ValueError(f"Dataset {dataset_name} not supported. Available: {list(self.DATASET_CONFIGS.keys())}")
        
        self.dataset_name = dataset_name
        self.config = self.DATASET_CONFIGS[dataset_name]
        self.k = self.config['k']
        
        print(f"🎯 Initializing {self.config['display_name']}")
        print(f"📝 {self.config['description']}")
        print(f"🔧 k={self.k} examples per prompt")
        
        # Load candidates
        print(f"🔄 Loading candidates...")
        self.candidates = self._load_candidates()
        print(f"✅ Loaded {len(self.candidates)} candidates")
        
        # Initialize official FPP solver
        print(f"🔄 Initializing FPP solver...")
        self.fpp_solver = create_fpp_solver()
        self.fpp_instance = FunctionPrototypePrompting()
        print(f"✅ FPP solver ready")
        
        # Load Policy Network (if exists)
        print(f"🔄 Loading Policy Network...")
        self.policy_network = self._load_policy_network()
        if self.policy_network:
            print(f"✅ Policy Network loaded")
        else:
            print(f"⚠️ Policy Network not found - will skip policy tests")
        
        # Get tolerance function
        self.tolerance_func = get_tolerance_function(dataset_name)
        
        print(f"🎯 {dataset_name} comparison study ready!")

    def _load_candidates(self) -> List[Dict]:
        """Load candidate examples for the dataset."""
        candidates_path = self.config['candidates_file']
        
        if not os.path.exists(candidates_path):
            raise FileNotFoundError(f"Candidates file not found: {candidates_path}")
        
        try:
            with open(candidates_path, 'r', encoding='utf-8') as f:
                candidates = json.load(f)
            
            # Filter valid candidates
            valid_candidates = []
            for candidate in candidates:
                if candidate.get('code') and candidate.get('question') and candidate.get('embedding'):
                    valid_candidates.append({
                        'question': candidate['question'],
                        'code': candidate['code'],
                        'context': candidate.get('context', ''),
                        'embedding': candidate['embedding']
                    })
            
            return valid_candidates
            
        except Exception as e:
            raise Exception(f"Error loading candidates: {e}")

    def _load_policy_network(self) -> PolicyNetwork:
        """Load trained Policy Network model if exists."""
        model_path = self.config['model_file']
        
        if not os.path.exists(model_path):
            return None
        
        try:
            # Initialize Policy Network with standard config
            policy_net = PolicyNetwork(emb_dim=1536, hidden_dim=768)
            
            # Load model state
            import torch
            checkpoint = torch.load(model_path, map_location='cpu')
            
            # Handle different checkpoint formats
            if 'model_state_dict' in checkpoint:
                policy_net.load_state_dict(checkpoint['model_state_dict'])
            else:
                policy_net.load_state_dict(checkpoint)
            
            policy_net.eval()
            return policy_net
            
        except Exception as e:
            print(f"⚠️ Policy Network loading failed: {e}")
            return None

    def test_zero_shot_fpp(self, sample: Dict) -> Tuple[bool, Any]:
        """Test FPP zero-shot using official implementation."""
        try:
            result = self.fpp_solver(sample['question'], sample['context'])
            predicted = result.get('result') if isinstance(result, dict) else result
            
            if predicted is not None:
                is_correct = self.tolerance_func(predicted, sample['ground_truth'])
                return is_correct, predicted
            else:
                return False, None
                
        except Exception as e:
            print(f"❌ Zero-shot FPP error: {e}")
            return False, None

    def test_fpp_with_random_examples(self, sample: Dict) -> Tuple[bool, Any]:
        """Test FPP with randomly selected examples."""
        try:
            if len(self.candidates) < self.k:
                print(f"⚠️ Not enough candidates ({len(self.candidates)} < {self.k})")
                return False, None
            
            random_examples = random.sample(self.candidates, self.k)
            
            # Create prompt with examples
            prompt = create_fpp_with_examples_prompt(
                question=sample['question'],
                examples=random_examples,
                context=sample['context']
            )
            
            # Use official FPP instance
            response = self.fpp_instance._call_llm(prompt)
            if not response:
                return False, None
            
            # Clean and execute code
            from mint.utils import clean_code, execute_code
            cleaned_code = clean_code(response)
            result, error = execute_code(cleaned_code)
            
            if error:
                return False, None
            
            if result is not None:
                is_correct = self.tolerance_func(result, sample['ground_truth'])
                return is_correct, result
            else:
                return False, None
                
        except Exception as e:
            print(f"❌ Random FPP error: {e}")
            return False, None

    def test_fpp_with_policy_examples(self, sample: Dict) -> Tuple[bool, Any]:
        """Test FPP with Policy Network selected examples."""
        if not self.policy_network:
            return False, None
            
        try:
            # Create embedding for test sample using standardized function
            test_embedding = create_standardized_embedding(
                context=sample['context'], 
                question=sample['question'],
                openai_client=OpenAI(api_key=self.fpp_instance.api_key)
            )
            
            # Create problem dict
            problem_dict = {
                'question': sample['question'],
                'context': sample['context'],
                'answer': sample['ground_truth'],
                'embedding': test_embedding
            }
            
            # Use Policy Network to select examples
            from openai import OpenAI
            openai_client = OpenAI(api_key=self.fpp_instance.api_key)
            evaluator = PolicyNetworkEvaluator(openai_client, self.fpp_instance.model)
            selected_examples = evaluator.select_with_policy(
                policy_net=self.policy_network,
                problem=problem_dict,
                candidate_pool=self.candidates,
                k=self.k
            )
            
            if len(selected_examples) < self.k:
                return False, None
            
            # Create prompt with selected examples
            prompt = create_fpp_with_examples_prompt(
                question=sample['question'],
                examples=selected_examples,
                context=sample['context']
            )
            
            # Use official FPP instance
            response = self.fpp_instance._call_llm(prompt)
            if not response:
                return False, None
            
            # Clean and execute code
            from mint.utils import clean_code, execute_code
            cleaned_code = clean_code(response)
            result, error = execute_code(cleaned_code)
            
            if error:
                return False, None
            
            if result is not None:
                is_correct = self.tolerance_func(result, sample['ground_truth'])
                return is_correct, result
            else:
                return False, None
                
        except Exception as e:
            print(f"❌ Policy FPP error: {e}")
            return False, None

    def run_comparison(self, n_samples: int = 10) -> Dict[str, Any]:
        """Run comparison study on dataset samples."""
        print(f"\n🚀 Starting {self.dataset_name} Comparison Study")
        print(f"📊 Testing {n_samples} samples")
        print("=" * 50)
        
        # Load test data
        data = DatasetLoader.load_dataset(self.dataset_name)
        test_samples = data[:n_samples]
        
        print(f"✅ Loaded {len(test_samples)} test samples")
        
        # Results tracking
        results = {
            'zero_shot': {'correct': 0, 'total': 0, 'details': []},
            'random_examples': {'correct': 0, 'total': 0, 'details': []},
            'policy_examples': {'correct': 0, 'total': 0, 'details': []}
        }
        
        for i, sample in enumerate(test_samples):
            print(f"\n📝 Sample {i+1}/{len(test_samples)}")
            print(f"Question: {sample['question'][:80]}...")
            print(f"Ground Truth: {sample['ground_truth']}")
            
            # Test 1: Zero-shot FPP
            print("  🎯 Testing Zero-shot FPP...")
            success1, result1 = self.test_zero_shot_fpp(sample)
            results['zero_shot']['total'] += 1
            if success1:
                results['zero_shot']['correct'] += 1
            results['zero_shot']['details'].append({
                'sample_id': i,
                'correct': success1,
                'predicted': result1,
                'ground_truth': sample['ground_truth']
            })
            print(f"    Result: {result1} ({'✅' if success1 else '❌'})")
            
            # Test 2: FPP + Random examples
            print("  🎲 Testing FPP + Random examples...")
            success2, result2 = self.test_fpp_with_random_examples(sample)
            results['random_examples']['total'] += 1
            if success2:
                results['random_examples']['correct'] += 1
            results['random_examples']['details'].append({
                'sample_id': i,
                'correct': success2,
                'predicted': result2,
                'ground_truth': sample['ground_truth']
            })
            print(f"    Result: {result2} ({'✅' if success2 else '❌'})")
            
            # Test 3: FPP + Policy Network (if available)
            if self.policy_network:
                print("  🤖 Testing FPP + Policy Network...")
                success3, result3 = self.test_fpp_with_policy_examples(sample)
                results['policy_examples']['total'] += 1
                if success3:
                    results['policy_examples']['correct'] += 1
                results['policy_examples']['details'].append({
                    'sample_id': i,
                    'correct': success3,
                    'predicted': result3,
                    'ground_truth': sample['ground_truth']
                })
                print(f"    Result: {result3} ({'✅' if success3 else '❌'})")
            else:
                print("  ⚠️ Policy Network not available - skipping")
                results['policy_examples']['details'].append({
                    'sample_id': i,
                    'correct': False,
                    'predicted': None,
                    'ground_truth': sample['ground_truth']
                })
            
            print("-" * 30)
        
        # Calculate final accuracies
        zero_shot_acc = (results['zero_shot']['correct'] / results['zero_shot']['total']) * 100 if results['zero_shot']['total'] > 0 else 0
        random_acc = (results['random_examples']['correct'] / results['random_examples']['total']) * 100 if results['random_examples']['total'] > 0 else 0
        policy_acc = (results['policy_examples']['correct'] / results['policy_examples']['total']) * 100 if results['policy_examples']['total'] > 0 else 0
        
        print(f"\n🏆 FINAL RESULTS - {self.dataset_name}")
        print("=" * 50)
        print(f"Zero-shot FPP:      {zero_shot_acc:.1f}% ({results['zero_shot']['correct']}/{results['zero_shot']['total']})")
        print(f"FPP + Random:       {random_acc:.1f}% ({results['random_examples']['correct']}/{results['random_examples']['total']})")
        if self.policy_network:
            print(f"FPP + Policy Net:   {policy_acc:.1f}% ({results['policy_examples']['correct']}/{results['policy_examples']['total']})")
        else:
            print(f"FPP + Policy Net:   N/A (model not found)")
        
        # Determine winner
        methods = [('Zero-shot FPP', zero_shot_acc), ('FPP + Random', random_acc)]
        if self.policy_network:
            methods.append(('FPP + Policy Network', policy_acc))
        
        best_method = max(methods, key=lambda x: x[1])
        print(f"\n🥇 Best Method: {best_method[0]} ({best_method[1]:.1f}%)")
        
        summary = {
            'dataset': self.dataset_name,
            'n_samples': n_samples,
            'accuracies': {
                'zero_shot_fpp': zero_shot_acc,
                'fpp_random': random_acc,
                'fpp_policy': policy_acc if self.policy_network else None
            },
            'detailed_results': results,
            'best_method': best_method[0],
            'best_accuracy': best_method[1]
        }
        
        return summary


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description='Generic Comparison Study for Mathematical Reasoning')
    parser.add_argument('--dataset', '-d', required=True, 
                       choices=['GSM8K', 'SVAMP', 'TabMWP', 'TAT-QA', 'FinQA'],
                       help='Dataset to test on')
    parser.add_argument('--samples', '-s', type=int, default=10,
                       help='Number of test samples (default: 10)')
    parser.add_argument('--output-dir', '-o', default='results',
                       help='Output directory for results (default: results)')
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    random.seed(42)
    
    try:
        # Initialize comparison study
        study = GenericComparisonStudy(args.dataset)
        
        # Run comparison
        results = study.run_comparison(n_samples=args.samples)
        
        # Save results
        output_file = f"{args.output_dir}/{args.dataset.lower()}_comparison_{args.samples}samples.json"
        os.makedirs(args.output_dir, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Results saved to: {output_file}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 