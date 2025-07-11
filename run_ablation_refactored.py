#!/usr/bin/env python3
"""
MathCoRL - Refactored Triple Ablation Study

Refactored version using improved architecture:
- Configuration management
- Namespace management 
- Problem evaluation
- Function registry

Usage:
    python run_ablation_refactored.py --dataset FinQA --samples 20
    python run_ablation_refactored.py --help
"""

import argparse
import json
import random
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Local imports
from mint.core import FunctionPrototypePrompting
from mint.testing import DatasetLoader
from mint.tracking import get_tracker
from mint.ablation_config import FunctionPrototypeConfig, AblationStudyConfig
from mint.namespace_manager import NamespaceManager
from mint.problem_evaluator import ProblemEvaluator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RefactoredTripleAblationStudy:
    """Refactored triple ablation study with improved architecture."""
    
    def __init__(self, model: str = None, temperature: float = 0.1):
        """Initialize ablation study.
        
        Args:
            model: LLM model name
            temperature: LLM temperature setting
        """
        self.model = AblationStudyConfig.get_model(model)
        self.temperature = temperature
        self.tracker = get_tracker()
        
        # Initialize components
        self.namespace_manager = NamespaceManager(
            AblationStudyConfig.ENHANCED_FUNCTIONS_MODULE
        )
        self.problem_evaluator = ProblemEvaluator(self.namespace_manager)
        
        # Validate configuration
        self._validate_setup()
        
        logger.info(f"RefactoredTripleAblationStudy initialized with model: {self.model}")
    
    def _validate_setup(self):
        """Validate setup and configuration."""
        # Check prototype files
        missing_files = FunctionPrototypeConfig.validate_prototype_files()
        if missing_files:
            logger.warning(f"Missing prototype files: {missing_files}")
        
        # Check namespace loading
        namespace = self.namespace_manager.get_enhanced_namespace()
        logger.info(f"Loaded {len(namespace)} enhanced functions")
    
    def load_function_prototypes(self, prototype_type: str) -> str:
        """Load function prototypes from file.
        
        Args:
            prototype_type: Type of prototypes to load
            
        Returns:
            Prototype file content
        """
        file_path = FunctionPrototypeConfig.get_prototype_file_path(prototype_type)
        
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def create_solver_with_prototypes(self, prototype_type: str) -> FunctionPrototypePrompting:
        """Create FPP solver with specific function prototypes.
        
        Args:
            prototype_type: Type of prototypes to use
            
        Returns:
            Configured solver instance
        """
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
    
    def run_triple_ablation(self, dataset: str, n_samples: int, seed: int = 42) -> Dict[str, Any]:
        """Run triple ablation study on specific dataset.
        
        Args:
            dataset: Dataset name
            n_samples: Number of samples to test
            seed: Random seed for reproducibility
            
        Returns:
            Complete results dictionary
        """
        logger.info(f"🧪 Starting refactored triple ablation study for {dataset}")
        
        # Validate and set parameters
        n_samples = AblationStudyConfig.validate_samples(n_samples)
        random.seed(seed)
        
        # Load and sample test data
        test_problems = self._load_and_sample_data(dataset, n_samples)
        
        # Create solvers for each function type
        solvers = self._create_solvers()
        
        # Run evaluations
        results = self._run_evaluations(solvers, test_problems, dataset)
        
        # Calculate metrics
        metrics = self._calculate_metrics(results)
        
        # Create final results
        final_results = {
            'dataset': dataset,
            'n_samples': len(test_problems),
            'seed': seed,
            'timestamp': datetime.now().isoformat(),
            'model': self.model,
            'temperature': self.temperature,
            'metrics': metrics,
            'detailed_results': results
        }
        
        self._log_summary(metrics)
        return final_results
    
    def _load_and_sample_data(self, dataset: str, n_samples: int) -> List[Dict]:
        """Load and sample test data.
        
        Args:
            dataset: Dataset name
            n_samples: Number of samples
            
        Returns:
            List of sampled problems
        """
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
        return test_problems
    
    def _create_solvers(self) -> Dict[str, FunctionPrototypePrompting]:
        """Create solvers for each function type.
        
        Returns:
            Dictionary mapping function type to solver
        """
        return {
            'original': self.create_solver_with_prototypes('original'),
            'financial': self.create_solver_with_prototypes('financial'),
            'all': self.create_solver_with_prototypes('all')
        }
    
    def _run_evaluations(self, solvers: Dict[str, FunctionPrototypePrompting], 
                        test_problems: List[Dict], dataset: str) -> Dict[str, List[Dict]]:
        """Run evaluations for all solver types.
        
        Args:
            solvers: Dictionary of solvers
            test_problems: List of problems to solve
            dataset: Dataset name
            
        Returns:
            Dictionary mapping solver type to results
        """
        results = {}
        
        for solver_name, solver in solvers.items():
            logger.info(f"📊 Evaluating with {solver_name.upper()} function prototypes...")
            solver_results = []
            
            for i, problem in enumerate(test_problems, 1):
                logger.info(f"{solver_name.capitalize()} functions - Problem {i}/{len(test_problems)}")
                result = self.problem_evaluator.evaluate_problem(solver, problem, dataset)
                solver_results.append(result)
            
            results[solver_name] = solver_results
        
        return results
    
    def _calculate_metrics(self, results: Dict[str, List[Dict]]) -> Dict[str, Dict[str, float]]:
        """Calculate performance metrics for each solver type.
        
        Args:
            results: Results from evaluations
            
        Returns:
            Dictionary of metrics by solver type
        """
        metrics = {}
        
        for solver_name, solver_results in results.items():
            total = len(solver_results)
            correct = sum(1 for r in solver_results if r['is_correct'])
            successful = sum(1 for r in solver_results if r['success'])
            
            metrics[solver_name] = {
                'accuracy': correct / total if total > 0 else 0,
                'success_rate': successful / total if total > 0 else 0,
                'correct_count': correct,
                'success_count': successful,
                'total_count': total
            }
        
        return metrics
    
    def _log_summary(self, metrics: Dict[str, Dict[str, float]]):
        """Log summary of results.
        
        Args:
            metrics: Performance metrics
        """
        logger.info("✅ Refactored triple ablation study completed!")
        logger.info("📊 Results Summary:")
        
        for solver_name, solver_metrics in metrics.items():
            logger.info(f"   {solver_name.capitalize()} Accuracy:  {solver_metrics['accuracy']:.3f}")
        
        for solver_name, solver_metrics in metrics.items():
            logger.info(f"   {solver_name.capitalize()} Success Rate:  {solver_metrics['success_rate']:.3f}")
    
    def save_results(self, results: Dict[str, Any], output_dir: str = None) -> str:
        """Save results to JSON file.
        
        Args:
            results: Results dictionary
            output_dir: Output directory
            
        Returns:
            Path to saved file
        """
        output_dir = output_dir or AblationStudyConfig.RESULTS_DIR
        
        # Ensure output directory exists
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Create filename
        dataset = results['dataset']
        n_samples = results['n_samples']
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"refactored_triple_ablation_{dataset}_{n_samples}samples_{timestamp}.json"
        filepath = Path(output_dir) / filename
        
        # Save results
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 Results saved to: {filepath}")
        return str(filepath)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="MathCoRL Refactored Triple Ablation Study",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--dataset',
        default='FinQA',
        choices=['FinQA', 'TabMWP'],
        help='Dataset to test (default: FinQA)'
    )
    
    parser.add_argument(
        '--samples',
        type=int,
        default=AblationStudyConfig.DEFAULT_SAMPLES,
        help=f'Number of samples to test (default: {AblationStudyConfig.DEFAULT_SAMPLES})'
    )
    
    parser.add_argument(
        '--model',
        default=None,
        help='Model to use (default: from env/config)'
    )
    
    parser.add_argument(
        '--temperature',
        type=float,
        default=AblationStudyConfig.DEFAULT_TEMPERATURE,
        help=f'Temperature for LLM (default: {AblationStudyConfig.DEFAULT_TEMPERATURE})'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=AblationStudyConfig.DEFAULT_SEED,
        help=f'Random seed (default: {AblationStudyConfig.DEFAULT_SEED})'
    )
    
    parser.add_argument(
        '--output-dir',
        default=AblationStudyConfig.RESULTS_DIR,
        help=f'Output directory for results (default: {AblationStudyConfig.RESULTS_DIR})'
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
    logger.info("🧪 MathCoRL - Refactored Triple Function Prototypes Ablation Study")
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
        study = RefactoredTripleAblationStudy(
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
        
        # Display final summary
        logger.info("")
        logger.info("=" * 70)
        logger.info("🏆 REFACTORED TRIPLE ABLATION STUDY SUMMARY")
        logger.info("=" * 70)
        
        metrics = results['metrics']
        logger.info(f"{args.dataset}:")
        for solver_name in ['original', 'financial', 'all']:
            if solver_name in metrics:
                acc = metrics[solver_name]['accuracy']
                logger.info(f"  {solver_name.capitalize()} Accuracy:     {acc:.3f}")
        
        logger.info("")
        for solver_name in ['original', 'financial', 'all']:
            if solver_name in metrics:
                sr = metrics[solver_name]['success_rate']
                logger.info(f"  {solver_name.capitalize()} Success Rate: {sr:.3f}")
        
        logger.info("")
        
        # Determine winner
        accuracies = {
            name.capitalize(): metrics[name]['accuracy']
            for name in metrics.keys()
        }
        
        winner = max(accuracies, key=accuracies.get)
        logger.info(f"🥇 Best Performance: {winner} ({accuracies[winner]:.3f} accuracy)")
        
        logger.info("")
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main()) 