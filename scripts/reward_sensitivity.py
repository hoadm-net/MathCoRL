#!/usr/bin/env python3
"""
Reward Weight Sensitivity Analysis for MathCoRL Policy Network

Analyzes how different reward weight configurations affect policy performance.

Test Configurations:
1. Baseline: (λacc=0.6, λsim=0.3, λdiv=0.1) - Current default
2. Accuracy-focused: (λacc=0.9, λsim=0.05, λdiv=0.05) - Prioritize correctness
3. Diversity-focused: (λacc=0.4, λsim=0.5, λdiv=0.1) - Emphasize varied examples
4. Balanced: (λacc=0.5, λsim=0.25, λdiv=0.25) - Equal consideration
5. Length-penalized: (λacc=0.5, λsim=0.2, λdiv=0.3) - Prefer diverse, short examples

Usage:
    python scripts/reward_sensitivity.py --dataset GSM8K --samples 50
    python scripts/reward_sensitivity.py --dataset FinQA --samples 100 --epochs 3
    python scripts/reward_sensitivity.py --all-configs --plot
"""

import argparse
import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any
import numpy as np
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from mint.icrl.trainer import PolicyNetworkTrainer
from mint.icrl.evaluator import PolicyNetworkEvaluator
from mint.config import load_config
from mint.reproducibility import set_seed

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Define reward configurations to test
REWARD_CONFIGS = {
    'baseline': {
        'lambda_accuracy': 0.6,
        'lambda_similarity': 0.3,
        'lambda_diversity': 0.1,
        'description': 'Current default - balanced with accuracy priority'
    },
    'accuracy_focused': {
        'lambda_accuracy': 0.9,
        'lambda_similarity': 0.05,
        'lambda_diversity': 0.05,
        'description': 'Maximize correctness - minimal similarity/diversity'
    },
    'diversity_focused': {
        'lambda_accuracy': 0.4,
        'lambda_similarity': 0.5,
        'lambda_diversity': 0.1,
        'description': 'Emphasize semantic similarity - explore varied approaches'
    },
    'balanced': {
        'lambda_accuracy': 0.5,
        'lambda_similarity': 0.25,
        'lambda_diversity': 0.25,
        'description': 'Equal consideration of all objectives'
    },
    'length_penalized': {
        'lambda_accuracy': 0.5,
        'lambda_similarity': 0.2,
        'lambda_diversity': 0.3,
        'description': 'Prefer diverse examples - reduce redundancy'
    }
}


class RewardSensitivityAnalyzer:
    """Analyze sensitivity of policy performance to reward weights."""
    
    def __init__(self, dataset: str, results_dir: str = "results/reward_sensitivity", seed: int = 42):
        """Initialize analyzer.
        
        Args:
            dataset: Dataset name (GSM8K, FinQA, etc.)
            results_dir: Directory to save results
            seed: Random seed for reproducibility
        """
        self.dataset = dataset
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.seed = seed
        
        # Set seed for reproducibility
        set_seed(seed)
        
        logger.info(f"RewardSensitivityAnalyzer initialized for {dataset}")
        logger.info(f"Results will be saved to {self.results_dir}")
    
    def train_with_config(self, config_name: str, config: Dict[str, float],
                         epochs: int = 3, samples: int = None) -> Dict[str, Any]:
        """Train policy network with specific reward configuration.
        
        Args:
            config_name: Name of reward configuration
            config: Reward weight configuration
            epochs: Number of training epochs
            samples: Number of samples per epoch (None = all)
            
        Returns:
            Dictionary with training results and metrics
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"Training with configuration: {config_name}")
        logger.info(f"Weights: λacc={config['lambda_accuracy']:.2f}, "
                   f"λsim={config['lambda_similarity']:.2f}, "
                   f"λdiv={config['lambda_diversity']:.2f}")
        logger.info(f"Description: {config['description']}")
        logger.info(f"{'='*80}\n")
        
        # Reset seed before training
        set_seed(self.seed)
        
        # Initialize trainer
        trainer = PolicyNetworkTrainer(
            dataset_name=self.dataset,
            candidates_dir="candidates",
            models_dir=f"models/reward_sensitivity/{config_name}"
        )
        
        # Override reward calculation method with custom weights
        original_calculate_reward = trainer.calculate_reward
        
        def custom_calculate_reward(is_correct, problem_emb, example_embs):
            """Custom reward with configurable weights."""
            import torch.nn.functional as F
            
            # Accuracy component
            accuracy_reward = 1.0 if is_correct else 0.0
            
            # Similarity component
            similarity_reward = F.cosine_similarity(
                problem_emb,
                example_embs.mean(dim=0)
            ).item()
            
            # Diversity component
            if example_embs.size(0) >= 2:
                diversity_reward = 1.0 - F.cosine_similarity(
                    example_embs[0].unsqueeze(0),
                    example_embs[1].unsqueeze(0)
                ).item()
            else:
                diversity_reward = 0.0
            
            # Apply custom weights
            total_reward = (
                config['lambda_accuracy'] * accuracy_reward +
                config['lambda_similarity'] * similarity_reward +
                config['lambda_diversity'] * diversity_reward
            )
            
            return total_reward
        
        trainer.calculate_reward = custom_calculate_reward
        
        # Train for specified epochs
        training_history = []
        for epoch in range(1, epochs + 1):
            epoch_metrics = trainer.train_epoch(epoch, n_samples=samples)
            training_history.append(epoch_metrics)
            
            logger.info(f"Epoch {epoch}/{epochs} - "
                       f"Loss: {epoch_metrics['loss']:.4f}, "
                       f"Reward: {epoch_metrics['reward']:.4f}, "
                       f"Accuracy: {epoch_metrics['accuracy']:.2%}")
        
        # Evaluate trained policy
        logger.info(f"\nEvaluating {config_name} policy...")
        # Simple evaluation: use final epoch accuracy
        final_accuracy = training_history[-1]['accuracy']
        
        # Compile results
        results = {
            'config_name': config_name,
            'config': config,
            'dataset': self.dataset,
            'epochs': epochs,
            'samples_per_epoch': samples,
            'seed': self.seed,
            'training_history': training_history,
            'final_metrics': {
                'accuracy': final_accuracy,
                'reward': training_history[-1]['reward'],
                'final_loss': training_history[-1]['loss']
            },
            'timestamp': datetime.now().isoformat()
        }
        
        # Save individual configuration results
        result_file = self.results_dir / f"{self.dataset}_{config_name}.json"
        with open(result_file, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {result_file}")
        
        return results
    
    def run_all_configs(self, epochs: int = 3, samples: int = None) -> Dict[str, Dict[str, Any]]:
        """Run sensitivity analysis for all configurations.
        
        Args:
            epochs: Number of training epochs per configuration
            samples: Number of samples per epoch (None = all)
            
        Returns:
            Dictionary mapping config names to results
        """
        all_results = {}
        
        logger.info(f"\n{'#'*80}")
        logger.info(f"Starting Reward Sensitivity Analysis")
        logger.info(f"Dataset: {self.dataset}")
        logger.info(f"Configurations: {len(REWARD_CONFIGS)}")
        logger.info(f"Epochs per config: {epochs}")
        logger.info(f"Samples per epoch: {samples or 'all'}")
        logger.info(f"{'#'*80}\n")
        
        for config_name, config in REWARD_CONFIGS.items():
            try:
                results = self.train_with_config(config_name, config, epochs, samples)
                all_results[config_name] = results
            except Exception as e:
                logger.error(f"Error training with {config_name}: {e}")
                continue
        
        # Save combined results
        combined_file = self.results_dir / f"{self.dataset}_all_configs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(combined_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        logger.info(f"\nCombined results saved to {combined_file}")
        
        # Print summary
        self.print_summary(all_results)
        
        return all_results
    
    def print_summary(self, results: Dict[str, Dict[str, Any]]):
        """Print summary table of results."""
        logger.info(f"\n{'='*80}")
        logger.info("REWARD SENSITIVITY ANALYSIS SUMMARY")
        logger.info(f"{'='*80}")
        logger.info(f"{'Configuration':<20} {'λacc':<6} {'λsim':<6} {'λdiv':<6} {'Accuracy':<10} {'Reward':<12}")
        logger.info(f"{'-'*80}")
        
        for config_name, result in results.items():
            config = result['config']
            metrics = result['final_metrics']
            logger.info(
                f"{config_name:<20} "
                f"{config['lambda_accuracy']:<6.2f} "
                f"{config['lambda_similarity']:<6.2f} "
                f"{config['lambda_diversity']:<6.2f} "
                f"{metrics['accuracy']:<10.2%} "
                f"{metrics['reward']:<12.4f}"
            )
        
        logger.info(f"{'='*80}\n")


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(
        description='Reward Weight Sensitivity Analysis for Policy Network',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Test single configuration
    python scripts/reward_sensitivity.py --dataset GSM8K --config baseline --samples 50
    
    # Test all configurations
    python scripts/reward_sensitivity.py --dataset FinQA --all-configs --epochs 3
    
    # Full analysis with plotting
    python scripts/reward_sensitivity.py --dataset GSM8K --all-configs --samples 100 --plot

Available Configurations:
    - baseline: (0.6, 0.3, 0.1) - Current default
    - accuracy_focused: (0.9, 0.05, 0.05) - Maximize correctness
    - diversity_focused: (0.4, 0.5, 0.1) - Emphasize variety
    - balanced: (0.5, 0.25, 0.25) - Equal weights
    - length_penalized: (0.5, 0.2, 0.3) - Prefer diversity
        """
    )
    
    # Required arguments
    parser.add_argument(
        '--dataset', '-d',
        type=str,
        required=True,
        choices=['GSM8K', 'SVAMP', 'TabMWP', 'TAT-QA', 'FinQA'],
        help='Dataset to analyze'
    )
    
    # Configuration selection
    parser.add_argument(
        '--config', '-c',
        type=str,
        choices=list(REWARD_CONFIGS.keys()),
        help='Specific configuration to test (default: all)'
    )
    
    parser.add_argument(
        '--all-configs',
        action='store_true',
        help='Test all reward configurations'
    )
    
    # Training parameters
    parser.add_argument(
        '--epochs', '-e',
        type=int,
        default=3,
        help='Training epochs per configuration (default: 3)'
    )
    
    parser.add_argument(
        '--samples', '-s',
        type=int,
        default=None,
        help='Samples per epoch (default: all candidates)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    # Output options
    parser.add_argument(
        '--results-dir',
        type=str,
        default='results/reward_sensitivity',
        help='Directory to save results (default: results/reward_sensitivity)'
    )
    
    parser.add_argument(
        '--plot',
        action='store_true',
        help='Generate visualization plots (requires matplotlib)'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.all_configs and not args.config:
        parser.error("Must specify either --config or --all-configs")
    
    # Initialize analyzer
    analyzer = RewardSensitivityAnalyzer(
        dataset=args.dataset,
        results_dir=args.results_dir,
        seed=args.seed
    )
    
    # Run analysis
    if args.all_configs:
        results = analyzer.run_all_configs(epochs=args.epochs, samples=args.samples)
    else:
        config = REWARD_CONFIGS[args.config]
        results = {
            args.config: analyzer.train_with_config(
                args.config, config, epochs=args.epochs, samples=args.samples
            )
        }
    
    # Generate plots if requested
    if args.plot:
        try:
            from scripts.plot_sensitivity import plot_sensitivity_results
            plot_dir = Path(args.results_dir) / "plots"
            plot_dir.mkdir(exist_ok=True)
            plot_sensitivity_results(results, output_dir=plot_dir)
            logger.info(f"Plots saved to {plot_dir}")
        except ImportError:
            logger.warning("Matplotlib not available. Skipping plot generation.")
            logger.warning("Install with: pip install matplotlib seaborn")
    
    logger.info("\n✅ Reward sensitivity analysis complete!")


if __name__ == '__main__':
    main()
