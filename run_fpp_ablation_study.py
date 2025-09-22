#!/usr/bin/env python3
"""
MathCoRL - FPP Ablation Study for Table Completion

Implements ablation variants for Function Prototype Prompting (FPP):
- Policy Network (retrieval vs random policy)
- No Policy (manual prompts without policy)
- Reduced Library (core functions only)
- No Constraints (unconstrained function usage)

Usage:
    python run_fpp_ablation_study.py --dataset TAT-QA --samples 25
    python run_fpp_ablation_study.py --dataset FinQA --samples 25
    python run_fpp_ablation_study.py --help
"""

import argparse
import json
import random
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Local imports
from mint.core import FunctionPrototypePrompting
from mint.testing import DatasetLoader
from mint.tracking import get_tracker
from mint.namespace_manager import NamespaceManager
from mint.problem_evaluator import ProblemEvaluator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FPPAblationStudy:
    """FPP Ablation Study for table completion."""
    
    def __init__(self, model: str = None, temperature: float = 0.1):
        """Initialize FPP ablation study.
        
        Args:
            model: LLM model name
            temperature: LLM temperature setting
        """
        self.model = model or os.getenv("DEFAULT_MODEL", "gpt-4o-mini")
        self.temperature = temperature
        self.tracker = get_tracker()
        
        # Initialize components
        self.namespace_manager = NamespaceManager("mint.enhanced_functions")
        self.problem_evaluator = ProblemEvaluator(self.namespace_manager)
        
        logger.info(f"FPPAblationStudy initialized with model: {self.model}")
    
    def load_function_prototypes(self, variant_type: str) -> str:
        """Load function prototypes for specific variant.
        
        Args:
            variant_type: Type of variant (original, reduced, etc.)
            
        Returns:
            Function prototypes content
        """
        if variant_type == "reduced":
            # Core mathematical functions only
            return self._get_reduced_function_prototypes()
        elif variant_type == "no_constraints":
            # All available functions without constraints
            return self._get_unconstrained_function_prototypes()
        else:
            # Standard function prototypes
            file_path = "templates/function_prototypes_all.txt"
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
    
    def _get_reduced_function_prototypes(self) -> str:
        """Get reduced set of core mathematical functions."""
        return """# Core Mathematical Functions

def add(a: float, b: float) -> float:
    \"\"\"Add two numbers.\"\"\"
    
def subtract(a: float, b: float) -> float:
    \"\"\"Subtract b from a.\"\"\"
    
def multiply(a: float, b: float) -> float:
    \"\"\"Multiply two numbers.\"\"\"
    
def divide(a: float, b: float) -> float:
    \"\"\"Divide a by b.\"\"\"
    
def percentage(value: float, percent: float) -> float:
    \"\"\"Calculate percentage of a value.\"\"\"
    
def average(numbers: list) -> float:
    \"\"\"Calculate average of a list of numbers.\"\"\"
    
def sum_list(numbers: list) -> float:
    \"\"\"Sum all numbers in a list.\"\"\"
    
def max_value(numbers: list) -> float:
    \"\"\"Find maximum value in a list.\"\"\"
    
def min_value(numbers: list) -> float:
    \"\"\"Find minimum value in a list.\"\"\"
"""
    
    def _get_unconstrained_function_prototypes(self) -> str:
        """Get unconstrained function prototypes with expanded capabilities."""
        # Load all available prototypes
        try:
            with open("templates/function_prototypes_all.txt", 'r', encoding='utf-8') as f:
                base_prototypes = f.read()
        except FileNotFoundError:
            with open("templates/function_prototypes.txt", 'r', encoding='utf-8') as f:
                base_prototypes = f.read()
        
        # Add unconstrained note
        unconstrained_header = """# Unconstrained Function Usage
# All functions below can be used freely without limitations
# Multiple function calls and complex operations are encouraged

"""
        return unconstrained_header + base_prototypes
    
    def create_solver_variant(self, variant: str, dataset: str) -> FunctionPrototypePrompting:
        """Create FPP solver for specific variant.
        
        Args:
            variant: Variant type (baseline, policy_retrieval, policy_random, no_policy, reduced, no_constraints)
            dataset: Dataset name for context
            
        Returns:
            Configured solver instance
        """
        solver = FunctionPrototypePrompting(
            model=self.model,
            temperature=self.temperature
        )
        
        if variant == "baseline":
            # Standard MathCoRL FPP with retrieval policy
            prototypes = self.load_function_prototypes("standard")
            solver._custom_prototypes = prototypes
            solver._variant_type = "baseline"
            solver._use_policy = True
            solver._policy_type = "retrieval"
            
        elif variant == "policy_retrieval":
            # Retrieval-based policy network
            prototypes = self.load_function_prototypes("standard")
            solver._custom_prototypes = prototypes
            solver._variant_type = "policy_retrieval"
            solver._use_policy = True
            solver._policy_type = "retrieval"
            
        elif variant == "policy_random":
            # Random policy selection
            prototypes = self.load_function_prototypes("standard")
            solver._custom_prototypes = prototypes
            solver._variant_type = "policy_random"
            solver._use_policy = True
            solver._policy_type = "random"
            
        elif variant == "no_policy":
            # Manual prompts without policy
            prototypes = self.load_function_prototypes("standard")
            solver._custom_prototypes = prototypes
            solver._variant_type = "no_policy"
            solver._use_policy = False
            solver._manual_prompt = True
            
        elif variant == "reduced_library":
            # Reduced function library
            prototypes = self.load_function_prototypes("reduced")
            solver._custom_prototypes = prototypes
            solver._variant_type = "reduced_library"
            solver._use_policy = True
            solver._policy_type = "retrieval"
            
        elif variant == "no_constraints":
            # No constraints on function usage
            prototypes = self.load_function_prototypes("no_constraints")
            solver._custom_prototypes = prototypes
            solver._variant_type = "no_constraints"
            solver._use_policy = False
            solver._unconstrained = True
            
        else:
            raise ValueError(f"Unknown variant: {variant}")
        
        return solver
    
    def run_ablation_study(self, dataset: str, n_samples: int, seed: int = 42) -> Dict[str, Any]:
        """Run complete FPP ablation study.
        
        Args:
            dataset: Dataset name (TAT-QA or FinQA)
            n_samples: Number of samples to test
            seed: Random seed for reproducibility
            
        Returns:
            Complete results dictionary
        """
        logger.info(f"🧪 Starting FPP ablation study for {dataset}")
        
        # Set random seed
        random.seed(seed)
        
        # Load and sample test data
        test_problems = self._load_and_sample_data(dataset, n_samples)
        
        # Define ablation variants
        variants = [
            "baseline",
            "policy_retrieval", 
            "policy_random",
            "no_policy",
            "reduced_library",
            "no_constraints"
        ]
        
        # Run evaluations for each variant
        results = {}
        for variant in variants:
            logger.info(f"📊 Evaluating variant: {variant}")
            
            solver = self.create_solver_variant(variant, dataset)
            variant_results = []
            
            for i, problem in enumerate(test_problems, 1):
                logger.info(f"{variant} - Problem {i}/{len(test_problems)}")
                result = self.problem_evaluator.evaluate_problem(solver, problem, dataset)
                result['variant'] = variant
                variant_results.append(result)
            
            results[variant] = variant_results
        
        # Calculate metrics
        metrics = self._calculate_metrics(results)
        
        # Create final results
        final_results = {
            'study_type': 'FPP_ablation',
            'dataset': dataset,
            'n_samples': len(test_problems),
            'seed': seed,
            'timestamp': datetime.now().isoformat(),
            'model': self.model,
            'temperature': self.temperature,
            'variants': variants,
            'metrics': metrics,
            'detailed_results': results
        }
        
        self._log_summary(metrics, dataset)
        return final_results
    
    def _load_and_sample_data(self, dataset: str, n_samples: int) -> List[Dict]:
        """Load and sample test data."""
        # Map dataset names
        dataset_map = {
            'TAT-QA': 'TAT-QA',
            'FinQA': 'FinQA'
        }
        
        mapped_dataset = dataset_map.get(dataset, dataset)
        test_data = DatasetLoader.load_dataset(mapped_dataset)
        
        if not test_data:
            raise ValueError(f"Could not load test data for {dataset}")
        
        # Sample problems
        if n_samples < len(test_data):
            test_problems = random.sample(test_data, n_samples)
        else:
            test_problems = test_data
            logger.warning(f"Requested {n_samples} samples, but only {len(test_data)} available")
        
        logger.info(f"Testing on {len(test_problems)} problems")
        return test_problems
    
    def _calculate_metrics(self, results: Dict[str, List[Dict]]) -> Dict[str, Dict[str, float]]:
        """Calculate performance metrics for each variant."""
        metrics = {}
        
        for variant, variant_results in results.items():
            total = len(variant_results)
            correct = sum(1 for r in variant_results if r['is_correct'])
            successful = sum(1 for r in variant_results if r['success'])
            
            metrics[variant] = {
                'accuracy': correct / total if total > 0 else 0,
                'success_rate': successful / total if total > 0 else 0,
                'correct_count': correct,
                'success_count': successful,
                'total_count': total
            }
        
        return metrics
    
    def _log_summary(self, metrics: Dict[str, Dict[str, float]], dataset: str):
        """Log summary of results."""
        logger.info("")
        logger.info("=" * 70)
        logger.info(f"🏆 FPP ABLATION STUDY SUMMARY - {dataset}")
        logger.info("=" * 70)
        
        # Sort variants by accuracy
        sorted_variants = sorted(
            metrics.items(), 
            key=lambda x: x[1]['accuracy'], 
            reverse=True
        )
        
        for variant, metric in sorted_variants:
            acc = metric['accuracy']
            logger.info(f"{variant:20} Accuracy: {acc:.3f} ({metric['correct_count']}/{metric['total_count']})")
        
        logger.info("")
        
        # Find best variant
        best_variant, best_metrics = sorted_variants[0]
        logger.info(f"🥇 Best Performance: {best_variant} ({best_metrics['accuracy']:.3f} accuracy)")
        logger.info("")
    
    def save_results(self, results: Dict[str, Any], output_dir: str = "results"):
        """Save results to file."""
        # Create output directory
        output_path = Path(output_dir) / "fpp_ablation"
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dataset = results['dataset']
        filename = f"fpp_ablation_{dataset}_{timestamp}.json"
        
        file_path = output_path / filename
        
        # Save results
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 Results saved to: {file_path}")
        
        # Also save summary table
        self._save_summary_table(results, output_path, timestamp)
    
    def _save_summary_table(self, results: Dict[str, Any], output_path: Path, timestamp: str):
        """Save summary table for easy reading."""
        dataset = results['dataset']
        metrics = results['metrics']
        
        summary_file = output_path / f"fpp_ablation_summary_{dataset}_{timestamp}.txt"
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(f"FPP Ablation Study Summary - {dataset}\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Model: {results['model']}\n")
            f.write(f"Samples: {results['n_samples']}\n")
            f.write(f"Timestamp: {results['timestamp']}\n\n")
            
            f.write("Results:\n")
            f.write("-" * 30 + "\n")
            
            # Sort by accuracy
            sorted_variants = sorted(
                metrics.items(), 
                key=lambda x: x[1]['accuracy'], 
                reverse=True
            )
            
            for variant, metric in sorted_variants:
                acc = metric['accuracy']
                f.write(f"{variant:20} {acc:.3f}\n")
        
        logger.info(f"📋 Summary saved to: {summary_file}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="FPP Ablation Study for Table Completion",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["TAT-QA", "FinQA"],
        help="Dataset to evaluate on"
    )
    
    parser.add_argument(
        "--samples",
        type=int,
        default=25,
        help="Number of samples to test"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="LLM model to use (default from config)"
    )
    
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="LLM temperature"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Output directory for results"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Display configuration
    logger.info("🧪 MathCoRL - FPP Ablation Study for Table Completion")
    logger.info("=" * 70)
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Samples: {args.samples}")
    logger.info(f"Model: {args.model or 'from config'}")
    logger.info(f"Temperature: {args.temperature}")
    logger.info(f"Random seed: {args.seed}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info("")
    
    try:
        # Create ablation study
        study = FPPAblationStudy(
            model=args.model,
            temperature=args.temperature
        )
        
        # Run ablation study
        results = study.run_ablation_study(
            dataset=args.dataset,
            n_samples=args.samples,
            seed=args.seed
        )
        
        # Save results
        study.save_results(results, args.output_dir)
        
        # Display completion table format
        logger.info("")
        logger.info("=" * 70)
        logger.info("📊 TABLE COMPLETION FORMAT")
        logger.info("=" * 70)
        
        metrics = results['metrics']
        logger.info(f"\n{args.dataset} Results:")
        logger.info(f"MathCoRL (baseline):     {metrics['baseline']['accuracy']:.2f}")
        logger.info(f"Policy Network:          {metrics['policy_retrieval']['accuracy']:.2f}")
        logger.info(f"Policy Random:           {metrics['policy_random']['accuracy']:.2f}")
        logger.info(f"No Policy:               {metrics['no_policy']['accuracy']:.2f}")
        logger.info(f"Reduced Library:         {metrics['reduced_library']['accuracy']:.2f}")
        logger.info(f"No Constraints:          {metrics['no_constraints']['accuracy']:.2f}")
        
        logger.info("")
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())