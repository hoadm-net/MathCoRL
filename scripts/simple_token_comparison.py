#!/usr/bin/env python3
"""
Simple Token Usage Comparison

Compare methods and measure token usage WITHOUT training new policies.
Just use existing comparison framework.

Usage:
    python scripts/simple_token_comparison.py --dataset TAT-QA --samples 151
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
import random

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mint.testing import DatasetLoader
from mint.config import get_dataset_config
from mint.reproducibility import set_seed
from dotenv import load_dotenv

# Import comparison methods
from comparison_study_generic import GenericComparisonStudy

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_api_logs():
    """Read API logs to get token counts"""
    log_file = "logs/api_usage.jsonl"
    if not os.path.exists(log_file):
        return []
    
    logs = []
    with open(log_file, 'r') as f:
        for line in f:
            try:
                logs.append(json.loads(line.strip()))
            except:
                pass
    return logs


def count_tokens_in_logs(start_time: float) -> int:
    """Count total tokens from logs after start_time"""
    logs = get_api_logs()
    total = 0
    for log in logs:
        # Check if log is after start_time
        log_time = log.get('timestamp', 0)
        if isinstance(log_time, str):
            try:
                log_time = datetime.fromisoformat(log_time).timestamp()
            except:
                continue
        
        if log_time >= start_time:
            total += log.get('total_tokens', 0)
    
    return total


def main():
    parser = argparse.ArgumentParser(
        description='Simple token usage comparison'
    )
    
    parser.add_argument('--dataset', '-d', type=str, default='TAT-QA')
    parser.add_argument('--samples', '-s', type=int, default=151)
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    logger.info("="*80)
    logger.info("TOKEN USAGE COMPARISON")
    logger.info("="*80)
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Samples: {args.samples}")
    logger.info(f"Seed: {args.seed}\n")
    
    # Methods to compare
    methods = ['zero-shot', 'random', 'policy']
    results = {}
    
    for method in methods:
        logger.info(f"\n{'='*60}")
        logger.info(f"Running: {method.upper()}")
        logger.info(f"{'='*60}")
        
        # Get initial timestamp
        start_time = time.time()
        
        try:
            # Run comparison
            study = GenericComparisonStudy(
                dataset_name=args.dataset,
                methods=[method],
                candidates_dir="candidates",
                models_dir="models"
            )
            
            comparison_results = study.run_comparison(
                n_samples=args.samples,
                verbose=False
            )
            
            elapsed = time.time() - start_time
            tokens_used = count_tokens_in_logs(start_time)
            
            # Extract results
            method_data = comparison_results.get(method, {})
            accuracy = method_data.get('accuracy', 0.0)
            correct = method_data.get('correct', 0)
            total = method_data.get('total', args.samples)
            
            results[method] = {
                'accuracy': accuracy,
                'accuracy_percent': accuracy * 100,
                'correct': correct,
                'total': total,
                'tokens': tokens_used,
                'avg_tokens': tokens_used / total if total > 0 else 0,
                'time_seconds': elapsed
            }
            
            logger.info(f"✅ {method}: {accuracy:.2%} accuracy, {tokens_used:,} tokens, {tokens_used/total:.0f} avg")
            
        except Exception as e:
            logger.error(f"❌ Error: {e}")
            results[method] = {'error': str(e)}
    
    # Print summary table
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    print(f"\nDataset: {args.dataset}, Samples: {args.samples}\n")
    
    print("| Method      | λ_acc | λ_sim | λ_div | Accuracy (%) | Total Tokens | Avg Tokens |")
    print("|-------------|-------|-------|-------|--------------|--------------|------------|")
    
    # Conceptual lambda mappings
    lambdas = {
        'zero-shot': (1.0, 0.0, 0.0),
        'random': (0.33, 0.33, 0.33),
        'policy': (0.6, 0.3, 0.1)
    }
    
    for method in methods:
        if method in results and 'error' not in results[method]:
            r = results[method]
            l = lambdas.get(method, (0, 0, 0))
            print(f"| {method:11} | {l[0]:.2f}  | {l[1]:.2f}  | {l[2]:.2f}  | "
                  f"{r['accuracy_percent']:12.2f} | {r['tokens']:12,} | {r['avg_tokens']:10.0f} |")
    
    # Save to JSON
    output_file = f"results/reward_sensitivity/{args.dataset}_token_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump({
            'dataset': args.dataset,
            'samples': args.samples,
            'seed': args.seed,
            'results': results,
            'timestamp': datetime.now().isoformat()
        }, f, indent=2)
    
    print(f"\n✅ Results saved to: {output_file}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
