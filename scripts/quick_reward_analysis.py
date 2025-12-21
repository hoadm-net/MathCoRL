#!/usr/bin/env python3
"""
Quick Reward Sensitivity Analysis - Evaluation Only

Evaluate different reward configurations WITHOUT re-training.
Just run the existing comparison methods and track tokens.

Usage:
    python scripts/quick_reward_analysis.py --dataset TAT-QA --samples 151
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
import random

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mint.testing import DatasetLoader
from mint.config import get_dataset_config
from mint.tracking import get_api_stats, clear_api_logs
from mint.reproducibility import set_seed
from comparison_study_generic import GenericComparisonStudy
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def get_token_usage_from_tracking() -> Dict[str, int]:
    """Get token usage from last run"""
    stats = get_api_stats(hours=1)
    
    total_tokens = sum(s.get('total_tokens', 0) for s in stats)
    input_tokens = sum(s.get('input_tokens', 0) for s in stats)
    output_tokens = sum(s.get('output_tokens', 0) for s in stats)
    
    return {
        'total': total_tokens,
        'input': input_tokens,
        'output': output_tokens
    }


def run_method_evaluation(
    dataset: str,
    method: str,
    n_samples: int,
    seed: int
) -> Dict[str, Any]:
    """
    Run single method evaluation with token tracking
    
    Args:
        dataset: Dataset name
        method: Method name (zero-shot, random, policy, kate, cds)
        n_samples: Number of samples
        seed: Random seed
        
    Returns:
        Results with accuracy and token usage
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Method: {method.upper()}")
    logger.info(f"{'='*60}")
    
    # Set seed
    set_seed(seed)
    
    # Clear logs to start fresh
    # Note: This might clear all logs, so be careful
    
    # Get initial token count
    initial_stats = get_token_usage_from_tracking()
    initial_tokens = initial_stats['total']
    
    # Run evaluation
    logger.info(f"Running {method} on {n_samples} samples...")
    start_time = time.time()
    
    try:
        # Use GenericComparisonStudy
        study = GenericComparisonStudy(
            dataset_name=dataset,
            methods=[method],
            candidates_dir="candidates",
            models_dir="models"
        )
        
        results = study.run_comparison(
            n_samples=n_samples,
            verbose=False
        )
        
        elapsed_time = time.time() - start_time
        
        # Get final token count
        final_stats = get_token_usage_from_tracking()
        final_tokens = final_stats['total']
        tokens_used = final_tokens - initial_tokens
        
        # Extract accuracy
        method_result = results.get(method, {})
        accuracy = method_result.get('accuracy', 0.0)
        correct = method_result.get('correct', 0)
        total = method_result.get('total', n_samples)
        
        result = {
            'method': method,
            'accuracy': accuracy,
            'accuracy_percent': accuracy * 100,
            'correct': correct,
            'total': total,
            'tokens': {
                'total': tokens_used,
                'input': final_stats['input'] - initial_stats['input'],
                'output': final_stats['output'] - initial_stats['output'],
                'avg_per_sample': tokens_used / total if total > 0 else 0
            },
            'time_seconds': elapsed_time
        }
        
        logger.info(f"✅ {method}: Accuracy = {accuracy:.2%}, Tokens = {tokens_used}, Avg = {result['tokens']['avg_per_sample']:.0f}/sample")
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Error in {method}: {e}")
        import traceback
        traceback.print_exc()
        return {
            'method': method,
            'error': str(e),
            'accuracy': 0.0,
            'accuracy_percent': 0.0
        }


def generate_comparison_table(results: List[Dict[str, Any]], dataset: str) -> str:
    """Generate markdown comparison table"""
    
    # Map methods to reward configs (conceptual mapping)
    method_to_config = {
        'zero-shot': {
            'name': 'Zero-shot (No ICL)',
            'lambda_acc': 1.0,
            'lambda_sim': 0.0,
            'lambda_div': 0.0,
            'description': 'No examples - pure model capability'
        },
        'random': {
            'name': 'Random Selection',
            'lambda_acc': 0.33,
            'lambda_sim': 0.33,
            'lambda_div': 0.33,
            'description': 'Uniform random - no learning'
        },
        'kate': {
            'name': 'KATE (Similarity)',
            'lambda_acc': 0.0,
            'lambda_sim': 1.0,
            'lambda_div': 0.0,
            'description': 'Pure similarity-based selection'
        },
        'policy': {
            'name': 'Policy Network (Learned)',
            'lambda_acc': 0.6,
            'lambda_sim': 0.3,
            'lambda_div': 0.1,
            'description': 'RL-learned balanced selection'
        },
        'cds': {
            'name': 'CDS (Diversity)',
            'lambda_acc': 0.0,
            'lambda_sim': 0.5,
            'lambda_div': 0.5,
            'description': 'Clustering-based diverse selection'
        }
    }
    
    # Sort by accuracy
    sorted_results = sorted(
        results,
        key=lambda x: x.get('accuracy_percent', 0),
        reverse=True
    )
    
    # Build markdown
    md = f"# ICL Method Comparison - Token & Accuracy Analysis\n\n"
    md += f"**Dataset**: {dataset}\n"
    md += f"**Samples**: {sorted_results[0].get('total', 0)}\n"
    md += f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    
    md += "## Results Table\n\n"
    md += "| Method | Conceptual λ_acc | λ_sim | λ_div | Accuracy (%) ↑ | Avg Tokens | Total Tokens | Efficiency |\n"
    md += "|--------|------------------|-------|-------|----------------|------------|--------------|------------|\n"
    
    for result in sorted_results:
        method = result['method']
        config = method_to_config.get(method, {
            'name': method,
            'lambda_acc': 0,
            'lambda_sim': 0,
            'lambda_div': 0
        })
        
        accuracy = result.get('accuracy_percent', 0)
        tokens_info = result.get('tokens', {})
        avg_tokens = tokens_info.get('avg_per_sample', 0)
        total_tokens = tokens_info.get('total', 0)
        
        # Efficiency: accuracy / (tokens/100)
        efficiency = accuracy / (avg_tokens / 100) if avg_tokens > 0 else 0
        
        md += (
            f"| **{config['name']}** "
            f"| {config['lambda_acc']:.2f} "
            f"| {config['lambda_sim']:.2f} "
            f"| {config['lambda_div']:.2f} "
            f"| **{accuracy:.2f}%** "
            f"| {avg_tokens:.0f} "
            f"| {total_tokens:,} "
            f"| {efficiency:.2f} |\n"
        )
    
    # Add description section
    md += "\n## Method Descriptions\n\n"
    for result in sorted_results:
        method = result['method']
        config = method_to_config.get(method, {'name': method, 'description': ''})
        md += f"- **{config['name']}**: {config.get('description', 'N/A')}\n"
    
    # Add insights
    md += "\n## Key Insights\n\n"
    
    best_acc = max(r.get('accuracy_percent', 0) for r in results)
    best_method = next(r['method'] for r in results if r.get('accuracy_percent', 0) == best_acc)
    
    min_tokens = min(r.get('tokens', {}).get('avg_per_sample', float('inf')) for r in results)
    most_efficient = next(r['method'] for r in results if r.get('tokens', {}).get('avg_per_sample', 0) == min_tokens)
    
    md += f"1. **Highest Accuracy**: {best_method.upper()} ({best_acc:.2f}%)\n"
    md += f"2. **Most Token-Efficient**: {most_efficient.upper()} ({min_tokens:.0f} tokens/sample)\n"
    
    # Compare policy to random
    policy_result = next((r for r in results if r['method'] == 'policy'), None)
    random_result = next((r for r in results if r['method'] == 'random'), None)
    
    if policy_result and random_result:
        acc_improvement = policy_result.get('accuracy_percent', 0) - random_result.get('accuracy_percent', 0)
        token_diff = policy_result.get('tokens', {}).get('avg_per_sample', 0) - random_result.get('tokens', {}).get('avg_per_sample', 0)
        
        md += f"3. **Policy vs Random**: +{acc_improvement:.2f}% accuracy, {'+' if token_diff > 0 else ''}{token_diff:.0f} tokens/sample\n"
    
    return md


def main():
    parser = argparse.ArgumentParser(
        description='Quick reward sensitivity analysis - evaluation only'
    )
    
    parser.add_argument(
        '--dataset', '-d',
        type=str,
        default='TAT-QA',
        help='Dataset to evaluate (default: TAT-QA)'
    )
    
    parser.add_argument(
        '--samples', '-s',
        type=int,
        default=151,
        help='Number of samples (default: 151)'
    )
    
    parser.add_argument(
        '--methods',
        type=str,
        nargs='+',
        default=['zero-shot', 'random', 'policy', 'kate', 'cds'],
        help='Methods to compare (default: all)'
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
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    logger.info("\n" + "="*80)
    logger.info("QUICK REWARD SENSITIVITY ANALYSIS")
    logger.info("="*80)
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Samples: {args.samples}")
    logger.info(f"Methods: {', '.join(args.methods)}")
    logger.info(f"Seed: {args.seed}")
    logger.info("="*80 + "\n")
    
    # Run evaluations
    all_results = []
    
    for method in args.methods:
        result = run_method_evaluation(
            dataset=args.dataset,
            method=method,
            n_samples=args.samples,
            seed=args.seed
        )
        result['dataset'] = args.dataset
        all_results.append(result)
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save JSON
    json_path = os.path.join(
        args.output_dir,
        f"{args.dataset}_quick_analysis_{timestamp}.json"
    )
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\n✅ Results saved to: {json_path}")
    
    # Generate and save markdown
    markdown = generate_comparison_table(all_results, args.dataset)
    
    md_path = os.path.join(
        args.output_dir,
        f"{args.dataset}_quick_analysis_{timestamp}.md"
    )
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(markdown)
    
    logger.info(f"✅ Markdown table saved to: {md_path}")
    
    # Print summary
    logger.info("\n" + "="*80)
    logger.info("ANALYSIS COMPLETE")
    logger.info("="*80)
    print("\n" + markdown)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
