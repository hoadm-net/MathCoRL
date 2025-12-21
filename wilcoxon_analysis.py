#!/usr/bin/env python3
"""
Wilcoxon Statistical Analysis: FPP vs PAL on TAT-QA

Compares FPP and PAL methods using Wilcoxon signed-rank test.
Tests on TAT-QA dataset.
"""

import json
import os
import sys
import glob
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import numpy as np
from scipy import stats

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from mint.cli import test_method


def run_method_test(method: str, dataset: str, n_samples: int = 5, run_id: int = 1) -> Dict:
    """
    Run a method on dataset and return results.
    
    Args:
        method: 'fpp' or 'pal'
        dataset: Dataset name (e.g., 'TAT-QA')
        n_samples: Number of samples to test
        run_id: Run identifier (1-5)
        
    Returns:
        Dict with keys: accuracy, correct, total, avg_time
    """
    print(f"\n{'='*70}")
    print(f"Testing {method.upper()} on {dataset} - Run {run_id}/5 ({n_samples} samples)")
    print(f"{'='*70}")
    
    # Run test using mint.cli.test_method
    results = test_method(
        method=method,
        dataset=dataset,
        limit=n_samples,
        verbose=False,  # Disable verbose for multiple runs
        output_dir=f'results/wilcoxon/{method}_run{run_id}'
    )
    
    # Normalize structure (test_method returns different keys)
    normalized = {
        'accuracy': results.get('accuracy', 0),
        'correct': results.get('correct_predictions', results.get('correct', 0)),
        'total': results.get('total_samples', results.get('total', n_samples)),
        'avg_time': results.get('avg_time', results.get('average_time', 0))
    }
    
    return normalized


def load_existing_results(method: str, run_id: int, dataset: str = 'TAT-QA') -> Dict:
    """
    Load results from existing JSON file.
    
    Args:
        method: 'fpp' or 'pal'
        run_id: Run identifier (1-5)
        dataset: Dataset name
        
    Returns:
        Dict with normalized structure or None if not found
    """
    import glob
    pattern = f"results/wilcoxon/{method}_run{run_id}/*.json"
    files = glob.glob(pattern)
    
    if files:
        with open(files[0], 'r') as f:
            data = json.load(f)
            return {
                'accuracy': data.get('accuracy', 0),
                'correct': data.get('correct_predictions', data.get('correct', 0)),
                'total': data.get('total_samples', data.get('total', 151)),
                'avg_time': data.get('avg_time', data.get('average_time', 0))
            }
    return None


def run_multiple_tests(method: str, dataset: str, n_samples: int = 5, n_runs: int = 3) -> List[Dict]:
    """
    Run a method multiple times on dataset.
    
    Args:
        method: 'fpp' or 'pal'
        dataset: Dataset name
        n_samples: Number of samples per run
        n_runs: Number of runs (default: 5)
        
    Returns:
        List of results from each run
    """
    results_list = []
    
    for run_id in range(1, n_runs + 1):
        results = run_method_test(method, dataset, n_samples, run_id)
        results_list.append(results)
        
        # Print summary for this run
        print(f"\n📊 {method.upper()} Run {run_id} Summary:")
        print(f"  Accuracy: {results['accuracy']:.2f}%")
        print(f"  Correct: {results['correct']}/{results['total']}")
        print(f"  Avg Time: {results['avg_time']:.2f}s")
    
    return results_list


def wilcoxon_analysis(fpp_results_list: List[Dict], pal_results_list: List[Dict]) -> Dict:
    """
    Perform Wilcoxon signed-rank test comparing FPP and PAL across multiple runs.
    
    Args:
        fpp_results_list: List of results from FPP runs
        pal_results_list: List of results from PAL runs
        
    Returns:
        Dict with statistical analysis results
    """
    print(f"\n{'='*70}")
    print("WILCOXON SIGNED-RANK TEST (Multiple Runs)")
    print(f"{'='*70}\n")
    
    n_runs = len(fpp_results_list)
    
    # Aggregate accuracies across runs
    fpp_accuracies = [r['accuracy'] for r in fpp_results_list]
    pal_accuracies = [r['accuracy'] for r in pal_results_list]
    
    # Aggregate times across runs
    fpp_times = [r['avg_time'] for r in fpp_results_list]
    pal_times = [r['avg_time'] for r in pal_results_list]
    
    # Calculate statistics
    fpp_acc_mean = np.mean(fpp_accuracies)
    fpp_acc_std = np.std(fpp_accuracies)
    pal_acc_mean = np.mean(pal_accuracies)
    pal_acc_std = np.std(pal_accuracies)
    
    fpp_time_mean = np.mean(fpp_times)
    fpp_time_std = np.std(fpp_times)
    pal_time_mean = np.mean(pal_times)
    pal_time_std = np.std(pal_times)
    
    # Wilcoxon signed-rank test on accuracies
    try:
        statistic, p_value = stats.wilcoxon(fpp_accuracies, pal_accuracies, alternative='two-sided')
    except Exception as e:
        print(f"⚠️  Warning: {e}")
        statistic, p_value = None, None
    
    # Display results
    print("📊 ACCURACY COMPARISON (across runs)")
    print(f"  FPP: {fpp_acc_mean:.2f}% ± {fpp_acc_std:.2f}%")
    print(f"    Individual runs: {[f'{a:.2f}%' for a in fpp_accuracies]}")
    print(f"  PAL: {pal_acc_mean:.2f}% ± {pal_acc_std:.2f}%")
    print(f"    Individual runs: {[f'{a:.2f}%' for a in pal_accuracies]}")
    print(f"  Mean Difference: {fpp_acc_mean - pal_acc_mean:+.2f}%")
    
    print(f"\n⏱️  TIME COMPARISON (across runs)")
    print(f"  FPP: {fpp_time_mean:.2f}s ± {fpp_time_std:.2f}s")
    print(f"    Individual runs: {[f'{t:.2f}s' for t in fpp_times]}")
    print(f"  PAL: {pal_time_mean:.2f}s ± {pal_time_std:.2f}s")
    print(f"    Individual runs: {[f'{t:.2f}s' for t in pal_times]}")
    print(f"  Mean Difference: {fpp_time_mean - pal_time_mean:+.2f}s")
    
    if statistic is not None:
        print(f"\n📈 WILCOXON TEST RESULTS")
        print(f"  Number of runs: {n_runs}")
        print(f"  Test statistic: {statistic:.2f}")
        print(f"  P-value: {p_value:.4f}")
        
        # Interpret p-value
        if p_value < 0.01:
            significance = "highly significant (p < 0.01)"
        elif p_value < 0.05:
            significance = "significant (p < 0.05)"
        elif p_value < 0.1:
            significance = "marginally significant (p < 0.1)"
        else:
            significance = "not significant (p >= 0.1)"
        
        print(f"  Interpretation: Difference is {significance}")
        
        # Determine which is better
        if fpp_acc_mean > pal_acc_mean:
            if p_value < 0.05:
                conclusion = "✅ FPP significantly outperforms PAL"
            else:
                conclusion = "⚖️  FPP performs better than PAL (not statistically significant)"
        elif pal_acc_mean > fpp_acc_mean:
            if p_value < 0.05:
                conclusion = "✅ PAL significantly outperforms FPP"
            else:
                conclusion = "⚖️  PAL performs better than FPP (not statistically significant)"
        else:
            conclusion = "⚖️  FPP and PAL perform equally"
        
        print(f"\n🎯 CONCLUSION")
        print(f"  {conclusion}")
    else:
        significance = None
        conclusion = None
    
    # Return analysis results
    return {
        'test': 'wilcoxon_signed_rank',
        'n_runs': n_runs,
        'n_samples_per_run': fpp_results_list[0]['total'],
        'fpp': {
            'accuracy_mean': fpp_acc_mean,
            'accuracy_std': fpp_acc_std,
            'accuracies': fpp_accuracies,
            'time_mean': fpp_time_mean,
            'time_std': fpp_time_std,
            'times': fpp_times
        },
        'pal': {
            'accuracy_mean': pal_acc_mean,
            'accuracy_std': pal_acc_std,
            'accuracies': pal_accuracies,
            'time_mean': pal_time_mean,
            'time_std': pal_time_std,
            'times': pal_times
        },
        'differences': {
            'accuracy_diff': fpp_acc_mean - pal_acc_mean,
            'time_diff': fpp_time_mean - pal_time_mean
        },
        'wilcoxon': {
            'statistic': float(statistic) if statistic is not None else None,
            'p_value': float(p_value) if p_value is not None else None,
            'significance': significance
        },
        'conclusion': conclusion
    }


def save_results(fpp_results_list: List[Dict], pal_results_list: List[Dict], analysis: Dict, output_dir: str = 'results/wilcoxon'):
    """Save all results to JSON files."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save aggregated results
    analysis_file = f"{output_dir}/wilcoxon_analysis_{timestamp}.json"
    
    # Save analysis with full context
    full_analysis = {
        'timestamp': timestamp,
        'dataset': 'TAT-QA',
        'n_runs': len(fpp_results_list),
        'n_samples_per_run': 151,
        'methods': ['FPP', 'PAL'],
        'analysis': analysis,
        'fpp_runs': [
            {
                'run_id': i+1,
                'accuracy': r['accuracy'],
                'correct': r['correct'],
                'total': r['total'],
                'avg_time': r['avg_time']
            }
            for i, r in enumerate(fpp_results_list)
        ],
        'pal_runs': [
            {
                'run_id': i+1,
                'accuracy': r['accuracy'],
                'correct': r['correct'],
                'total': r['total'],
                'avg_time': r['avg_time']
            }
            for i, r in enumerate(pal_results_list)
        ]
    }
    
    with open(analysis_file, 'w') as f:
        json.dump(full_analysis, f, indent=2)
    print(f"\n✅ Wilcoxon analysis saved to {analysis_file}")
    
    return analysis_file


def main():
    """Main function."""
    print("\n" + "="*70)
    print("WILCOXON STATISTICAL ANALYSIS: FPP vs PAL")
    print("Dataset: TAT-QA")
    print("Runs: 5 runs per method")
    print("Samples: 5 per run (test pipeline)")
    print("="*70)
    
    # Configuration
    dataset = 'TAT-QA'
    n_samples = 300
    n_runs = 5
    
    # Check if results already exist to avoid re-running
    results_dir = Path('results/wilcoxon')
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Run FPP multiple times
    print("\n🔵 Step 1/3: Running FPP (5 runs)...")
    fpp_results_list = run_multiple_tests('fpp', dataset, n_samples, n_runs)
    
    # Run PAL multiple times
    print("\n🔵 Step 2/3: Running PAL (5 runs)...")
    pal_results_list = run_multiple_tests('pal', dataset, n_samples, n_runs)
    
    # Perform Wilcoxon analysis
    print("\n🔵 Step 3/3: Performing Wilcoxon analysis...")
    analysis = wilcoxon_analysis(fpp_results_list, pal_results_list)
    
    # Save results
    analysis_file = save_results(fpp_results_list, pal_results_list, analysis)
    
    print("\n" + "="*70)
    print("✅ ANALYSIS COMPLETE")
    print(f"Results saved to: {analysis_file}")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
