#!/usr/bin/env python3
"""
Reward Sensitivity Experiment for MathCoRL

Test different reward configurations and measure:
- Accuracy (correctness)
- Token usage (efficiency)
- Training dynamics

Usage:
    python scripts/reward_sensitivity_experiment.py --dataset TAT-QA --samples 151
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mint.icrl.trainer import PolicyNetworkTrainer
from mint.icrl.config import RewardConfig
from mint.config import load_config, get_dataset_config
from mint.reproducibility import set_seed
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Reward configurations to test
REWARD_CONFIGS = {
    "accuracy_focused": {
        "lambda_acc": 0.9,
        "lambda_sim": 0.05,
        "lambda_div": 0.05,
        "description": "Maximize correctness - minimal similarity/diversity"
    },
    "balanced_default": {
        "lambda_acc": 0.6,
        "lambda_sim": 0.3,
        "lambda_div": 0.1,
        "description": "Current default - balanced with accuracy priority"
    },
    "diversity_focused": {
        "lambda_acc": 0.4,
        "lambda_sim": 0.5,
        "lambda_div": 0.1,
        "description": "Emphasize semantic similarity - explore varied approaches"
    },
    "balanced_equal": {
        "lambda_acc": 0.5,
        "lambda_sim": 0.25,
        "lambda_div": 0.25,
        "description": "Equal consideration of all objectives"
    },
    "efficiency_focused": {
        "lambda_acc": 0.5,
        "lambda_sim": 0.2,
        "lambda_div": 0.3,
        "description": "Prefer diverse examples - reduce redundancy"
    }
}


def train_and_evaluate(
    dataset: str,
    config_name: str,
    reward_config: RewardConfig,
    epochs: int,
    eval_samples: int,
    seed: int
) -> Dict[str, Any]:
    """
    Train policy with specific reward config and evaluate
    
    Args:
        dataset: Dataset name
        config_name: Configuration name
        reward_config: Reward configuration
        epochs: Training epochs
        eval_samples: Number of evaluation samples
        seed: Random seed
        
    Returns:
        Results dictionary with accuracy and token metrics
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"Config: {config_name}")
    logger.info(f"Reward: {reward_config}")
    logger.info(f"{'='*80}\n")
    
    # Set seed
    set_seed(seed)
    
    # Initialize trainer
    trainer = PolicyNetworkTrainer(
        dataset_name=dataset,
        candidates_dir="candidates",
        models_dir=f"models/reward_sensitivity",
        reward_config=reward_config
    )
    
    # Training
    logger.info(f"🔄 Training policy for {config_name}...")
    training_start = time.time()
    
    training_metrics = []
    try:
        for epoch in range(1, epochs + 1):
            metrics = trainer.train_epoch(epoch=epoch, n_samples=None)
            training_metrics.append(metrics)
            
            logger.info(
                f"Epoch {epoch}/{epochs} - "
                f"Loss: {metrics.get('loss', 0):.4f}, "
                f"Reward: {metrics.get('avg_reward', 0):.4f}, "
                f"Accuracy: {metrics.get('accuracy', 0):.2%}"
            )
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        raise
    
    training_time = time.time() - training_start
    
    # Save model
    model_path = f"models/reward_sensitivity/{dataset}_{config_name}_policy.pt"
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    try:
        trainer.save_model(model_path)
    except Exception as e:
        logger.error(f"❌ Failed to save model: {e}")
        # Continue anyway
    logger.info(f"✅ Model saved to {model_path}")
    
    # Evaluation
    logger.info(f"\n🧪 Evaluating on {eval_samples} samples...")
    eval_start = time.time()
    
    try:
        eval_results = trainer.evaluate_full(
            n_samples=eval_samples,
            verbose=False
        )
    except Exception as e:
        logger.error(f"❌ Evaluation failed: {e}")
        raise
    
    eval_time = time.time() - eval_start
    
    # Extract metrics
    results = {
        "config_name": config_name,
        "reward_config": {
            "lambda_acc": reward_config.accuracy_weight,
            "lambda_sim": reward_config.similarity_weight,
            "lambda_div": reward_config.diversity_weight
        },
        "training": {
            "epochs": epochs,
            "time_seconds": training_time,
            "final_loss": training_metrics[-1].get('loss', 0),
            "final_reward": training_metrics[-1].get('avg_reward', 0),
            "history": training_metrics
        },
        "evaluation": {
            "samples": eval_samples,
            "time_seconds": eval_time,
            "accuracy": eval_results.get('accuracy', 0),
            "accuracy_percent": eval_results.get('accuracy', 0) * 100,
            "correct": eval_results.get('correct', 0),
            "total": eval_results.get('total', 0),
            "avg_reward": eval_results.get('avg_reward', 0)
        },
        "timestamp": datetime.now().isoformat()
    }
    
    # Token usage (if available in eval_results)
    if 'token_usage' in eval_results:
        results['evaluation']['token_usage'] = eval_results['token_usage']
        results['evaluation']['avg_tokens'] = eval_results.get('avg_tokens', 0)
    
    logger.info(f"\n✅ Results for {config_name}:")
    logger.info(f"   Accuracy: {results['evaluation']['accuracy_percent']:.2f}%")
    logger.info(f"   Avg Reward: {results['evaluation']['avg_reward']:.4f}")
    if 'avg_tokens' in results['evaluation']:
        logger.info(f"   Avg Tokens: {results['evaluation']['avg_tokens']:.1f}")
    
    return results


def generate_markdown_table(all_results: List[Dict[str, Any]]) -> str:
    """Generate markdown table from results"""
    
    # Handle empty results
    if not all_results:
        return "# Reward Sensitivity Analysis Results\n\n**ERROR**: No results to display. All configurations failed.\n"
    
    # Sort by accuracy descending
    sorted_results = sorted(
        all_results, 
        key=lambda x: x.get('evaluation', {}).get('accuracy_percent', 0),
        reverse=True
    )
    
    # Build table
    table = "# Reward Sensitivity Analysis Results\n\n"
    table += f"**Dataset**: {sorted_results[0].get('dataset', 'Unknown')}\n"
    table += f"**Samples**: {sorted_results[0].get('evaluation', {}).get('samples', 'N/A')}\n"
    table += f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    
    table += "| Reward Config | λ_acc | λ_sim | λ_div | Accuracy (%) ↑ | Avg Reward | Training Loss |\n"
    table += "|---------------|-------|-------|-------|----------------|------------|---------------|\n"
    
    for result in sorted_results:
        config = result['reward_config']
        eval_data = result['evaluation']
        train_data = result['training']
        
        table += (
            f"| {result['config_name'].replace('_', ' ').title()} "
            f"| {config['lambda_acc']:.2f} "
            f"| {config['lambda_sim']:.2f} "
            f"| {config['lambda_div']:.2f} "
            f"| **{eval_data['accuracy_percent']:.2f}%** "
            f"| {eval_data['avg_reward']:.4f} "
            f"| {train_data['final_loss']:.4f} |\n"
        )
    
    # Add token usage table if available
    if 'avg_tokens' in sorted_results[0]['evaluation']:
        table += "\n## Token Usage Comparison\n\n"
        table += "| Reward Config | λ_acc | λ_sim | λ_div | Accuracy (%) | Avg Tokens ↓ | Efficiency Score |\n"
        table += "|---------------|-------|-------|-------|--------------|--------------|------------------|\n"
        
        for result in sorted_results:
            config = result['reward_config']
            eval_data = result['evaluation']
            accuracy = eval_data['accuracy_percent']
            tokens = eval_data.get('avg_tokens', 0)
            
            # Efficiency score: accuracy / (tokens / 100)
            efficiency = accuracy / (tokens / 100) if tokens > 0 else 0
            
            table += (
                f"| {result['config_name'].replace('_', ' ').title()} "
                f"| {config['lambda_acc']:.2f} "
                f"| {config['lambda_sim']:.2f} "
                f"| {config['lambda_div']:.2f} "
                f"| {accuracy:.2f}% "
                f"| {tokens:.0f} "
                f"| {efficiency:.2f} |\n"
            )
    
    # Add summary statistics
    table += "\n## Summary Statistics\n\n"
    accuracies = [r['evaluation']['accuracy_percent'] for r in all_results]
    table += f"- **Best Accuracy**: {max(accuracies):.2f}%\n"
    table += f"- **Worst Accuracy**: {min(accuracies):.2f}%\n"
    table += f"- **Average Accuracy**: {sum(accuracies) / len(accuracies):.2f}%\n"
    table += f"- **Std Dev**: {(sum((x - sum(accuracies)/len(accuracies))**2 for x in accuracies) / len(accuracies))**0.5:.2f}%\n"
    
    if 'avg_tokens' in all_results[0]['evaluation']:
        tokens = [r['evaluation'].get('avg_tokens', 0) for r in all_results]
        table += f"\n- **Avg Tokens (min)**: {min(tokens):.0f}\n"
        table += f"- **Avg Tokens (max)**: {max(tokens):.0f}\n"
        table += f"- **Avg Tokens (mean)**: {sum(tokens) / len(tokens):.0f}\n"
    
    return table


def main():
    parser = argparse.ArgumentParser(
        description='Reward Sensitivity Experiment',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--dataset', '-d',
        type=str,
        default='TAT-QA',
        choices=['GSM8K', 'SVAMP', 'TabMWP', 'TAT-QA', 'FinQA'],
        help='Dataset to run experiment on'
    )
    
    parser.add_argument(
        '--samples', '-s',
        type=int,
        default=151,
        help='Number of evaluation samples (default: 151)'
    )
    
    parser.add_argument(
        '--epochs', '-e',
        type=int,
        default=10,
        help='Training epochs per configuration (default: 10)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='results/reward_sensitivity',
        help='Output directory (default: results/reward_sensitivity)'
    )
    
    parser.add_argument(
        '--configs',
        type=str,
        nargs='+',
        default=None,
        help='Specific configs to test (default: all)'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Determine which configs to test
    configs_to_test = args.configs if args.configs else list(REWARD_CONFIGS.keys())
    
    logger.info("\n" + "="*80)
    logger.info("REWARD SENSITIVITY EXPERIMENT")
    logger.info("="*80)
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Evaluation samples: {args.samples}")
    logger.info(f"Training epochs: {args.epochs}")
    logger.info(f"Seed: {args.seed}")
    logger.info(f"Configurations to test: {len(configs_to_test)}")
    logger.info("="*80 + "\n")
    
    # Run experiments
    all_results = []
    
    for config_name in configs_to_test:
        if config_name not in REWARD_CONFIGS:
            logger.warning(f"Unknown config: {config_name}, skipping...")
            continue
        
        config_params = REWARD_CONFIGS[config_name]
        
        # Create RewardConfig
        reward_config = RewardConfig(
            accuracy_weight=config_params['lambda_acc'],
            similarity_weight=config_params['lambda_sim'],
            diversity_weight=config_params['lambda_div']
        )
        
        try:
            # Train and evaluate
            results = train_and_evaluate(
                dataset=args.dataset,
                config_name=config_name,
                reward_config=reward_config,
                epochs=args.epochs,
                eval_samples=args.samples,
                seed=args.seed
            )
            
            # Add dataset and description
            results['dataset'] = args.dataset
            results['description'] = config_params['description']
            
            all_results.append(results)
            
        except Exception as e:
            logger.error(f"❌ Error in config {config_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save JSON
    json_path = os.path.join(
        args.output_dir,
        f"{args.dataset}_reward_sensitivity_{timestamp}.json"
    )
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\n✅ Results saved to: {json_path}")
    
    # Generate and save markdown table
    markdown_table = generate_markdown_table(all_results)
    
    md_path = os.path.join(
        args.output_dir,
        f"{args.dataset}_reward_sensitivity_{timestamp}.md"
    )
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(markdown_table)
    
    logger.info(f"✅ Markdown table saved to: {md_path}")
    
    # Print summary
    logger.info("\n" + "="*80)
    logger.info("EXPERIMENT COMPLETE")
    logger.info("="*80)
    logger.info(markdown_table)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
