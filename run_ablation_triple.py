#!/usr/bin/env python3
"""
MathCoRL - Triple Ablation Study: Original vs Enhanced vs ALL Functions

Compare performance between:
1. Original functions (function_prototypes.txt)
2. Financial-specific functions (function_prototypes_fin.txt)  
3. ALL functions combined (function_prototypes_all.txt)

Usage:
    python run_ablation_triple.py --dataset FinQA --samples 20
    python run_ablation_triple.py --help
"""

import argparse
import os
import sys
import json
import random
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Tuple
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from mint.core import FunctionPrototypePrompting
from mint.testing import DatasetLoader
from mint.evaluation import get_tolerance_function
from mint.utils import evaluate_result
from mint.tracking import get_tracker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TripleAblationStudy:
    """Triple ablation study for function prototypes comparison."""
    
    def __init__(self, model: str = None, temperature: float = 0.1):
        """Initialize triple ablation study."""
        self.model = model or os.getenv("DEFAULT_MODEL", "gpt-4o-mini")
        self.temperature = temperature
        self.tracker = get_tracker()
        
        # Function prototype files
        self.prototype_files = {
            'original': 'templates/function_prototypes.txt',
            'financial': 'templates/function_prototypes_fin.txt',
            'all': 'templates/function_prototypes_all.txt'
        }
        
        logger.info(f"TripleAblationStudy initialized with model: {self.model}")
    
    def load_function_prototypes(self, prototype_type: str) -> str:
        """Load function prototypes from file."""
        file_path = self.prototype_files[prototype_type]
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Function prototypes file not found: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def create_solver_with_prototypes(self, prototype_type: str) -> FunctionPrototypePrompting:
        """Create FPP solver with specific function prototypes."""
        # Load prototypes
        prototypes = self.load_function_prototypes(prototype_type)
        
        # Create solver
        solver = FunctionPrototypePrompting(
            model=self.model,
            temperature=self.temperature
        )
        
        # Store prototypes for use in prompts
        solver._custom_prototypes = prototypes
        solver._prototype_type = prototype_type
        
        return solver
    
    def evaluate_problem(self, solver: FunctionPrototypePrompting, problem: Dict, dataset: str) -> Dict[str, Any]:
        """Evaluate a single problem with given solver."""
        try:
            # Get correct answer and context
            if dataset == 'FinQA':
                correct_answer = float(problem.get('ground_truth', 0))
                question_text = problem.get('question', '')
                context = problem.get('context', '')
            else:
                raise ValueError(f"Unsupported dataset: {dataset}")
            
            # Check if this is enhanced functions (non-original)
            if hasattr(solver, '_prototype_type') and solver._prototype_type != 'original':
                # Use enhanced functions namespace
                from mint.utils import execute_code_with_namespace
                from mint import enhanced_functions
                
                # Create namespace with enhanced functions
                namespace = {}
                for name in dir(enhanced_functions):
                    if not name.startswith('_'):
                        namespace[name] = getattr(enhanced_functions, name)
                
                # Use default solve method with custom prototypes
                from mint import prompts
                original_load_function_prototypes = prompts.load_function_prototypes
                
                def custom_load_function_prototypes():
                    return solver._custom_prototypes
                
                prompts.load_function_prototypes = custom_load_function_prototypes
                
                try:
                    # Get the generated code
                    result = solver.solve_detailed(question_text, context)
                    
                    # Re-execute with enhanced namespace if we have code
                    if result['code']:
                        from mint.utils import clean_code
                        cleaned_code = clean_code(result['code'])
                        result_value, error = execute_code_with_namespace(cleaned_code, namespace)
                        
                        result = {
                            'result': result_value,
                            'success': not error,  # True if error is empty/None
                            'error': error or '',
                            'code': cleaned_code
                        }
                        
                finally:
                    # Restore original function
                    prompts.load_function_prototypes = original_load_function_prototypes
            else:
                # Use default solve method for original functions
                result = solver.solve_detailed(question_text, context)
            
            # Evaluate correctness
            is_correct = False
            
            if result['success'] and result['result'] is not None:
                if dataset == 'FinQA':
                    # Use FinQA specific evaluation
                    from mint.utils import FinQA_generate_candidates
                    candidates = FinQA_generate_candidates(result['result'])
                    is_correct = correct_answer in candidates
                else:
                    # Use standard tolerance function
                    tolerance_fn = get_tolerance_function(dataset)
                    is_correct = tolerance_fn(result['result'], correct_answer)
            
            return {
                'problem_id': problem.get('id', ''),
                'question': question_text,
                'correct_answer': correct_answer,
                'predicted_answer': result['result'],
                'is_correct': is_correct,
                'success': result['success'],
                'error': result.get('error', ''),
                'code': result.get('code', '')
            }
            
        except Exception as e:
            logger.error(f"Error evaluating problem: {e}")
            return {
                'problem_id': problem.get('id', ''),
                'question': problem.get('question', ''),
                'correct_answer': None,
                'predicted_answer': None,
                'is_correct': False,
                'success': False,
                'error': str(e),
                'code': ''
            }
    
    def run_triple_ablation(self, dataset: str, n_samples: int, seed: int = 42) -> Dict[str, Any]:
        """Run triple ablation study on specific dataset."""
        logger.info(f"🧪 Starting triple ablation study for {dataset}")
        
        # Set random seed
        random.seed(seed)
        
        # Load test data
        test_data = DatasetLoader.load_dataset(dataset)
        
        if not test_data:
            raise ValueError(f"Could not load test data for {dataset}")
        
        # Sample problems
        if n_samples < len(test_data):
            test_problems = random.sample(test_data, n_samples)
        else:
            test_problems = test_data
            logger.warning(f"Requested {n_samples} samples, but only {len(test_data)} available")
        
        logger.info(f"Testing on {len(test_problems)} problems")
        
        # Create solvers with different function prototypes
        original_solver = self.create_solver_with_prototypes('original')
        financial_solver = self.create_solver_with_prototypes('financial')
        all_solver = self.create_solver_with_prototypes('all')
        
        # Results storage
        results = {
            'original': [],
            'financial': [],
            'all': []
        }
        
        # Run evaluation with original functions
        logger.info("📊 Evaluating with ORIGINAL function prototypes...")
        for i, problem in enumerate(test_problems):
            logger.info(f"Original functions - Problem {i+1}/{len(test_problems)}")
            result = self.evaluate_problem(original_solver, problem, dataset)
            results['original'].append(result)
        
        # Run evaluation with financial functions
        logger.info("📊 Evaluating with FINANCIAL function prototypes...")
        for i, problem in enumerate(test_problems):
            logger.info(f"Financial functions - Problem {i+1}/{len(test_problems)}")
            result = self.evaluate_problem(financial_solver, problem, dataset)
            results['financial'].append(result)
        
        # Run evaluation with ALL functions
        logger.info("📊 Evaluating with ALL function prototypes...")
        for i, problem in enumerate(test_problems):
            logger.info(f"All functions - Problem {i+1}/{len(test_problems)}")
            result = self.evaluate_problem(all_solver, problem, dataset)
            results['all'].append(result)
        
        # Calculate metrics for each approach
        metrics = {}
        for approach_name, approach_results in results.items():
            total = len(approach_results)
            correct = sum(1 for r in approach_results if r['is_correct'])
            successful = sum(1 for r in approach_results if r['success'])
            
            metrics[approach_name] = {
                'accuracy': correct / total if total > 0 else 0,
                'success_rate': successful / total if total > 0 else 0,
                'correct_count': correct,
                'success_count': successful,
                'total_count': total
            }
        
        # Create final results
        final_results = {
            'dataset': dataset,
            'n_samples': len(test_problems),
            'seed': seed,
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics,
            'results': results
        }
        
        # Log summary
        logger.info("✅ Triple ablation study completed!")
        logger.info(f"📊 Results Summary:")
        logger.info(f"   Original Accuracy:  {metrics['original']['accuracy']:.3f}")
        logger.info(f"   Financial Accuracy: {metrics['financial']['accuracy']:.3f}")
        logger.info(f"   All Functions Accuracy: {metrics['all']['accuracy']:.3f}")
        logger.info(f"   Original Success Rate:  {metrics['original']['success_rate']:.3f}")
        logger.info(f"   Financial Success Rate: {metrics['financial']['success_rate']:.3f}")
        logger.info(f"   All Functions Success Rate: {metrics['all']['success_rate']:.3f}")
        
        return final_results
    
    def save_results(self, results: Dict[str, Any], output_dir: str = "results"):
        """Save results to JSON file."""
        # Ensure output directory exists
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Create filename
        dataset = results['dataset']
        n_samples = results['n_samples']
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"triple_ablation_{dataset}_{n_samples}samples_{timestamp}.json"
        filepath = Path(output_dir) / filename
        
        # Save results
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 Results saved to: {filepath}")
        return str(filepath)

def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="MathCoRL Triple Ablation Study",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--dataset',
        default='FinQA',
        choices=['FinQA'],
        help='Dataset to test (default: FinQA)'
    )
    
    parser.add_argument(
        '--samples',
        type=int,
        default=10,
        help='Number of samples to test (default: 10)'
    )
    
    parser.add_argument(
        '--model',
        default=None,
        help='Model to use (default: from env/config)'
    )
    
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.1,
        help='Temperature for LLM (default: 0.1)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )
    
    parser.add_argument(
        '--output-dir',
        default='results',
        help='Output directory for results (default: results)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Display configuration
    logger.info("🧪 MathCoRL - Triple Function Prototypes Ablation Study")
    logger.info("=" * 60)
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Samples: {args.samples}")
    logger.info(f"Model: {args.model or 'from config'}")
    logger.info(f"Temperature: {args.temperature}")
    logger.info(f"Random seed: {args.seed}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info("")
    
    try:
        # Create ablation study
        study = TripleAblationStudy(
            model=args.model,
            temperature=args.temperature
        )
        
        # Run triple ablation
        results = study.run_triple_ablation(
            dataset=args.dataset,
            n_samples=args.samples,
            seed=args.seed
        )
        
        # Save results
        study.save_results(results, args.output_dir)
        
        logger.info("")
        logger.info("=" * 60)
        logger.info("🏆 TRIPLE ABLATION STUDY SUMMARY")
        logger.info("=" * 60)
        
        metrics = results['metrics']
        logger.info(f"{args.dataset}:")
        logger.info(f"  Original Accuracy:     {metrics['original']['accuracy']:.3f}")
        logger.info(f"  Financial Accuracy:    {metrics['financial']['accuracy']:.3f}")  
        logger.info(f"  All Functions Accuracy: {metrics['all']['accuracy']:.3f}")
        logger.info("")
        logger.info(f"  Original Success Rate:     {metrics['original']['success_rate']:.3f}")
        logger.info(f"  Financial Success Rate:    {metrics['financial']['success_rate']:.3f}")
        logger.info(f"  All Functions Success Rate: {metrics['all']['success_rate']:.3f}")
        logger.info("")
        
        # Determine winner
        accuracies = {
            'Original': metrics['original']['accuracy'],
            'Financial': metrics['financial']['accuracy'],
            'All Functions': metrics['all']['accuracy']
        }
        
        winner = max(accuracies, key=accuracies.get)
        logger.info(f"🥇 Best Performance: {winner} ({accuracies[winner]:.3f} accuracy)")
        
        logger.info("")
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 