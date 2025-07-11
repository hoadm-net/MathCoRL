#!/usr/bin/env python3
"""
MathCoRL - Ablation Study: Function Prototypes Comparison

Compare performance between original function prototypes and domain-specific function prototypes:
1. FinQA: Original functions vs Financial-specific functions
2. TabMWP: Original functions vs Table-specific functions

Usage:
    python run_ablation_study.py --dataset FinQA --samples 50
    python run_ablation_study.py --dataset TabMWP --samples 30
    python run_ablation_study.py --both --samples 20
    python run_ablation_study.py --help
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


class AblationStudy:
    """Ablation study for function prototypes comparison."""
    
    def __init__(self, model: str = None, temperature: float = 0.1):
        """Initialize ablation study."""
        self.model = model or os.getenv("DEFAULT_MODEL", "gpt-4o-mini")
        self.temperature = temperature
        self.tracker = get_tracker()
        
        # Function prototype files
        self.prototype_files = {
            'original': 'templates/function_prototypes.txt',
            'financial': 'templates/function_prototypes_fin.txt',
            'table': 'templates/function_prototypes_tbl.txt',
            'all': 'templates/function_prototypes_all.txt'
        }
        
        # Dataset configurations
        self.dataset_configs = {
            'FinQA': {
                'test_file': 'datasets/FinQA/test.json',
                'original_functions': 'original',
                'enhanced_functions': 'financial'
            },
            'TabMWP': {
                'test_file': 'datasets/TabMWP/test.json',
                'original_functions': 'original',
                'enhanced_functions': 'table'
            }
        }
        
        logger.info(f"AblationStudy initialized with model: {self.model}")
    
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
        
        return solver
    
    def create_custom_fpp_prompt(self, question: str, prototypes: str, context: str = "") -> str:
        """Create FPP prompt with custom function prototypes."""
        from mint.prompts import load_template
        
        # Load FPP template
        fpp_template_content = load_template("fpp")
        
        # Use custom prototypes instead of default ones
        from langchain.prompts import PromptTemplate
        template = PromptTemplate(
            template=fpp_template_content,
            input_variables=["function_prototypes"]
        )
        
        # Format FPP template with custom prototypes
        prompt = template.format(function_prototypes=prototypes)
        
        # Add context if provided
        if context:
            prompt += f"\n\n# CONTEXT\n{context}\n"
        
        # Add problem
        prompt += f"\n\n# PROBLEM\n{question}\n"
        prompt += "\n# SOLUTION\nGenerate Python code step by step:"
        
        return prompt
    
    def evaluate_problem(self, solver: FunctionPrototypePrompting, problem: Dict, dataset: str) -> Dict[str, Any]:
        """Evaluate a single problem with given solver."""
        try:
            # Get correct answer and context
            if dataset == 'FinQA':
                correct_answer = float(problem.get('ground_truth', 0))
                question_text = problem.get('question', '')
                context = problem.get('context', '')
            elif dataset == 'TabMWP':
                correct_answer = float(problem.get('ground_truth', 0))
                question_text = problem.get('question', '')
                context = problem.get('context', '')
            else:
                raise ValueError(f"Unsupported dataset: {dataset}")
            
            # Create custom prompt with appropriate prototypes
            if hasattr(solver, '_custom_prototypes') and solver._custom_prototypes != self.load_function_prototypes('original'):
                custom_prompt = self.create_custom_fpp_prompt(
                    question_text, 
                    solver._custom_prototypes,
                    problem.get('table', '') if dataset == 'TabMWP' else ''
                )
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
                    # Get the generated code (this will use original namespace for execution)
                    result = solver.solve_detailed(question_text, context)
                    
                    # Always re-execute with enhanced namespace for enhanced functions
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
                # Use default solve method
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
    
    def run_dataset_ablation(self, dataset: str, n_samples: int, seed: int = 42) -> Dict[str, Any]:
        """Run ablation study on specific dataset."""
        logger.info(f"🧪 Starting ablation study for {dataset}")
        
        # Set random seed
        random.seed(seed)
        
        # Get dataset config
        config = self.dataset_configs[dataset]
        
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
        original_solver = self.create_solver_with_prototypes(config['original_functions'])
        enhanced_solver = self.create_solver_with_prototypes(config['enhanced_functions'])
        
        # Run evaluation with original functions
        logger.info("📊 Evaluating with ORIGINAL function prototypes...")
        original_results = []
        for i, problem in enumerate(test_problems):
            logger.info(f"Original functions - Problem {i+1}/{len(test_problems)}")
            result = self.evaluate_problem(original_solver, problem, dataset)
            original_results.append(result)
        
        # Run evaluation with enhanced functions
        logger.info("📊 Evaluating with ENHANCED function prototypes...")
        enhanced_results = []
        for i, problem in enumerate(test_problems):
            logger.info(f"Enhanced functions - Problem {i+1}/{len(test_problems)}")
            result = self.evaluate_problem(enhanced_solver, problem, dataset)
            enhanced_results.append(result)
        
        # Calculate metrics
        original_accuracy = sum(1 for r in original_results if r['is_correct']) / len(original_results)
        enhanced_accuracy = sum(1 for r in enhanced_results if r['is_correct']) / len(enhanced_results)
        
        original_success_rate = sum(1 for r in original_results if r['success']) / len(original_results)
        enhanced_success_rate = sum(1 for r in enhanced_results if r['success']) / len(enhanced_results)
        
        # Summary
        summary = {
            'dataset': dataset,
            'n_samples': len(test_problems),
            'seed': seed,
            'original_functions': config['original_functions'],
            'enhanced_functions': config['enhanced_functions'],
            'original_accuracy': original_accuracy,
            'enhanced_accuracy': enhanced_accuracy,
            'accuracy_improvement': enhanced_accuracy - original_accuracy,
            'original_success_rate': original_success_rate,
            'enhanced_success_rate': enhanced_success_rate,
            'success_improvement': enhanced_success_rate - original_success_rate,
            'original_results': original_results,
            'enhanced_results': enhanced_results
        }
        
        logger.info(f"✅ {dataset} Ablation Results:")
        logger.info(f"   Original Accuracy: {original_accuracy:.3f}")
        logger.info(f"   Enhanced Accuracy: {enhanced_accuracy:.3f}")
        logger.info(f"   Improvement: {enhanced_accuracy - original_accuracy:+.3f}")
        logger.info(f"   Original Success Rate: {original_success_rate:.3f}")
        logger.info(f"   Enhanced Success Rate: {enhanced_success_rate:.3f}")
        
        return summary
    
    def save_results(self, results: Dict[str, Any], output_dir: str = "results"):
        """Save ablation results to file."""
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if isinstance(results, list):
            filename = f"ablation_study_combined_{timestamp}.json"
        else:
            dataset = results.get('dataset', 'unknown')
            filename = f"ablation_study_{dataset}_{timestamp}.json"
        
        filepath = os.path.join(output_dir, filename)
        
        # Save results
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 Results saved to: {filepath}")
        return filepath


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(
        description='Run ablation study comparing function prototypes',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_ablation_study.py --dataset FinQA --samples 50
    python run_ablation_study.py --dataset TabMWP --samples 30
    python run_ablation_study.py --both --samples 20
    python run_ablation_study.py --dataset FinQA --samples 10 --save-results
    
Ablation Comparison:
    - FinQA: Original functions vs Financial-specific functions
    - TabMWP: Original functions vs Table-specific functions
        """
    )
    
    # Dataset selection
    parser.add_argument(
        '--dataset', '-d',
        type=str,
        choices=['FinQA', 'TabMWP'],
        help='Dataset to run ablation study on'
    )
    
    parser.add_argument(
        '--both',
        action='store_true',
        help='Run ablation study on both FinQA and TabMWP'
    )
    
    # Test parameters
    parser.add_argument(
        '--samples', '-s',
        type=int,
        default=20,
        help='Number of test samples to evaluate (default: 20)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    # Model parameters
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='OpenAI model to use (default: from config)'
    )
    
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.1,
        help='Sampling temperature (default: 0.1)'
    )
    
    # Output options
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='results',
        help='Output directory for results (default: results)'
    )
    
    parser.add_argument(
        '--save-results',
        action='store_true',
        help='Save detailed results to JSON file'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.both and not args.dataset:
        parser.error("Must specify either --dataset or --both")
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.info("Verbose logging enabled")
    
    # Display configuration
    logger.info("🧪 MathCoRL - Function Prototypes Ablation Study")
    logger.info("=" * 60)
    
    if args.both:
        datasets = ['FinQA', 'TabMWP']
        logger.info("Datasets: Both FinQA and TabMWP")
    else:
        datasets = [args.dataset]
        logger.info(f"Dataset: {args.dataset}")
    
    logger.info(f"Samples per dataset: {args.samples}")
    logger.info(f"Random seed: {args.seed}")
    logger.info(f"Model: {args.model or 'from config'}")
    logger.info(f"Temperature: {args.temperature}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Save results: {args.save_results}")
    
    try:
        # Initialize ablation study
        study = AblationStudy(
            model=args.model,
            temperature=args.temperature
        )
        
        # Run ablation for each dataset
        all_results = []
        
        for dataset in datasets:
            logger.info(f"\n🚀 Starting {dataset} ablation study...")
            
            result = study.run_dataset_ablation(
                dataset=dataset,
                n_samples=args.samples,
                seed=args.seed
            )
            
            all_results.append(result)
        
        # Display summary
        logger.info("\n" + "=" * 60)
        logger.info("🏆 ABLATION STUDY SUMMARY")
        logger.info("=" * 60)
        
        for result in all_results:
            dataset = result['dataset']
            original_acc = result['original_accuracy']
            enhanced_acc = result['enhanced_accuracy']
            improvement = result['accuracy_improvement']
            
            logger.info(f"{dataset}:")
            logger.info(f"  Original Accuracy:  {original_acc:.3f}")
            logger.info(f"  Enhanced Accuracy:  {enhanced_acc:.3f}")
            logger.info(f"  Improvement:        {improvement:+.3f}")
            logger.info(f"  Status: {'✅ IMPROVED' if improvement > 0 else '❌ DECREASED' if improvement < 0 else '➖ NO CHANGE'}")
            logger.info("")
        
        # Save results if requested
        if args.save_results:
            if len(all_results) == 1:
                study.save_results(all_results[0], args.output_dir)
            else:
                study.save_results(all_results, args.output_dir)
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Ablation study failed: {e}")
        if args.verbose:
            import traceback
            logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    exit(main()) 