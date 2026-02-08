#!/usr/bin/env python3
"""
Mann-Whitney U Test Analysis - Auto-scan results

Tự động scan các file results và thực hiện Mann-Whitney U test.
Hỗ trợ 3 metrics: Accuracy, Reasoning Length, Token Usage

Usage:
    # Analyze specific dataset
    python analyze_mann_whitney.py --dataset GSM8K
    
    # Analyze all datasets
    python analyze_mann_whitney.py --dataset all
    
    # Specify results directory
    python analyze_mann_whitney.py --dataset FinQA --results-dir results
"""

import argparse
import json
import os
import sys
import glob
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import numpy as np

try:
    from scipy import stats
except ImportError:
    print("❌ ERROR: scipy is not installed")
    print("   Please install it: pip install scipy")
    sys.exit(1)


DATASETS = ['GSM8K', 'FinQA', 'TAT-QA']
METHODS = ['fpp', 'pot', 'pal']  # Removed CoT due to answer extraction issues
METHOD_NAMES = {
    'fpp': 'FPP (MathCoRL)',
    'pot': 'PoT',
    'pal': 'PAL'
}


def find_latest_result(dataset: str, method: str, results_dir: str = 'results') -> Optional[str]:
    """
    Tìm file results mới nhất cho dataset và method.
    
    Args:
        dataset: Dataset name (GSM8K, FinQA, TAT-QA)
        method: Method name (fpp, cot, pot, pal)
        results_dir: Results directory
        
    Returns:
        Path to latest result file or None
    """
    # Try both lowercase dataset name and method
    pattern = f"{results_dir}/{dataset.lower()}_{method.lower()}_results_*samples_*.json"
    files = glob.glob(pattern)
    
    if not files:
        return None
    
    # Sort by modification time, newest first
    files.sort(key=os.path.getmtime, reverse=True)
    return files[0]


def extract_per_instance_metrics(results_file: str) -> Dict[str, np.ndarray]:
    """
    Extract per-instance metrics từ results JSON.
    
    Returns:
        Dict với:
            - correct: np.array of 0/1
            - reasoning_length: np.array of reasoning lengths
            - total_tokens: np.array of token counts
    """
    try:
        with open(results_file, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"   ❌ JSON parsing error in {results_file}")
        print(f"      Error: {e}")
        return None
    
    results = data.get('results', [])
    n = len(results)
    
    if n == 0:
        return None
    
    # Initialize arrays
    correct = np.zeros(n, dtype=int)
    reasoning_length = np.zeros(n)
    total_tokens = np.zeros(n)
    
    for i, result in enumerate(results):
        try:
            # 1. Accuracy (0/1)
            correct[i] = 1 if result.get('correct', False) else 0
            
            # 2. Reasoning length (number of lines in code/reasoning)
            if 'code' in result and result['code']:
                # For FPP, PoT, PAL: count lines of code
                reasoning_length[i] = len(result['code'].strip().split('\n'))
            elif 'reasoning' in result and result['reasoning']:
                # For CoT: count words in reasoning
                reasoning_length[i] = len(result['reasoning'].split())
            else:
                reasoning_length[i] = 0
            
            # 3. Token usage (NOTE: locals field is ignored, not needed for analysis)
            total_tokens[i] = result.get('total_tokens', 0)
        
        except Exception as e:
            print(f"   ⚠️  Warning: Error processing result {i+1}: {e}")
            # Keep zero values for failed instances
            continue
    
    return {
        'correct': correct,
        'reasoning_length': reasoning_length,
        'total_tokens': total_tokens,
        'n_samples': n
    }


def mann_whitney_test(data1: np.ndarray, data2: np.ndarray, 
                      method1: str, method2: str, metric: str) -> Dict:
    """
    Perform Mann-Whitney U test between two methods.
    
    Args:
        data1: Array of per-instance values for method 1
        data2: Array of per-instance values for method 2
        method1: Name of method 1
        method2: Name of method 2
        metric: Metric name
        
    Returns:
        Dict with test results
    """
    # Remove NaN/inf values
    valid_mask = ~(np.isnan(data1) | np.isnan(data2) | 
                   np.isinf(data1) | np.isinf(data2))
    data1_clean = data1[valid_mask]
    data2_clean = data2[valid_mask]
    
    if len(data1_clean) == 0 or len(data2_clean) == 0:
        return {
            'method1': method1,
            'method2': method2,
            'metric': metric,
            'p_value': None,
            'error': 'No valid data'
        }
    
    # Perform Mann-Whitney U test
    try:
        statistic, p_value = stats.mannwhitneyu(
            data1_clean, 
            data2_clean, 
            alternative='two-sided'
        )
    except Exception as e:
        return {
            'method1': method1,
            'method2': method2,
            'metric': metric,
            'p_value': None,
            'error': str(e)
        }
    
    # Calculate descriptive statistics
    mean1 = float(np.mean(data1_clean))
    mean2 = float(np.mean(data2_clean))
    std1 = float(np.std(data1_clean))
    std2 = float(np.std(data2_clean))
    
    # Significance level
    if p_value < 0.001:
        sig_level = '***'
    elif p_value < 0.01:
        sig_level = '**'
    elif p_value < 0.05:
        sig_level = '*'
    else:
        sig_level = 'ns'
    
    return {
        'method1': method1,
        'method2': method2,
        'metric': metric,
        'u_statistic': float(statistic),
        'p_value': float(p_value),
        'sig_level': sig_level,
        'mean1': mean1,
        'std1': std1,
        'mean2': mean2,
        'std2': std2,
        'mean_diff': mean1 - mean2,
        'n1': len(data1_clean),
        'n2': len(data2_clean)
    }


def analyze_dataset(dataset: str, results_dir: str = 'results') -> Optional[Dict]:
    """
    Analyze one dataset - scan results and perform Mann-Whitney tests.
    
    Args:
        dataset: Dataset name
        results_dir: Results directory
        
    Returns:
        Dict with analysis results
    """
    print(f"\n{'='*70}")
    print(f"📊 ANALYZING {dataset}")
    print(f"{'='*70}")
    
    # Load results for all methods
    method_data = {}
    method_files = {}
    
    for method in METHODS:
        result_file = find_latest_result(dataset, method, results_dir)
        
        if not result_file:
            print(f"⚠️  No results found for {method.upper()} on {dataset}")
            continue
        
        print(f"📂 Found {method.upper()}: {Path(result_file).name}")
        
        # Extract metrics
        metrics = extract_per_instance_metrics(result_file)
        
        if metrics is None:
            print(f"❌ Failed to extract metrics from {method.upper()}")
            continue
        
        method_data[method.upper()] = metrics
        method_files[method.upper()] = result_file
        
        print(f"   ✓ Loaded {metrics['n_samples']} samples")
        print(f"   ✓ Accuracy: {np.mean(metrics['correct'])*100:.2f}%")
    
    if len(method_data) < 2:
        print(f"\n❌ Not enough methods found for {dataset}")
        print(f"   Need at least 2 methods, found {len(method_data)}")
        return None
    
    print(f"\n✓ Loaded {len(method_data)} methods: {', '.join(method_data.keys())}")
    
    # Perform pairwise Mann-Whitney tests
    # FPP (MathCoRL) vs each baseline
    mathcorl = 'FPP'
    baselines = [m for m in ['COT', 'POT', 'PAL'] if m in method_data]
    
    if mathcorl not in method_data:
        print(f"\n⚠️  FPP not found, cannot perform comparisons")
        return None
    
    comparisons = []
    
    print(f"\n{'='*70}")
    print(f"🔬 MANN-WHITNEY U TESTS: {mathcorl} vs Baselines")
    print(f"{'='*70}")
    
    for baseline in baselines:
        print(f"\n{'-'*70}")
        print(f"📊 {mathcorl} vs {baseline}")
        print(f"{'-'*70}")
        
        # Test each metric
        for metric_key, metric_name in [
            ('correct', 'Accuracy'),
            ('reasoning_length', 'Reasoning Length'),
            ('total_tokens', 'Token Usage')
        ]:
            data_mathcorl = method_data[mathcorl][metric_key]
            data_baseline = method_data[baseline][metric_key]
            
            result = mann_whitney_test(
                data_mathcorl,
                data_baseline,
                mathcorl,
                baseline,
                metric_key
            )
            
            comparisons.append(result)
            
            # Print result
            print(f"\n  {metric_name}:")
            print(f"    {mathcorl}: {result['mean1']:.2f} ± {result.get('std1', 0):.2f}")
            print(f"    {baseline}: {result['mean2']:.2f} ± {result.get('std2', 0):.2f}")
            
            if result['p_value'] is not None:
                print(f"    p-value: {result['p_value']:.4f} {result['sig_level']}")
                
                if result['p_value'] < 0.05:
                    winner = mathcorl if result['mean1'] > result['mean2'] else baseline
                    print(f"    → {winner} significantly better (p < 0.05)")
                else:
                    print(f"    → No significant difference")
    
    return {
        'dataset': dataset,
        'method_data': method_data,
        'method_files': method_files,
        'comparisons': comparisons
    }


def create_summary_table(all_results: Dict[str, Dict]) -> str:
    """Create formatted summary table for paper."""
    lines = []
    
    lines.append("\n" + "="*80)
    lines.append("TABLE: Mann–Whitney U Statistical Significance (p-values)")
    lines.append("="*80)
    
    metric_display = {
        'correct': 'ACCURACY',
        'reasoning_length': 'REASONING LENGTH', 
        'total_tokens': 'TOKEN USAGE'
    }
    
    for metric_key, metric_name in metric_display.items():
        lines.append(f"\n{metric_name}:")
        lines.append(f"{'Dataset':<15} {'vs PoT':<15} {'vs PAL':<15}")
        lines.append("-" * 45)
        
        for dataset in DATASETS:
            if dataset not in all_results or not all_results[dataset]:
                continue
            
            row = [dataset]
            
            for baseline in ['POT', 'PAL']:
                # Find comparison
                p_value = None
                sig_level = None
                
                for comp in all_results[dataset]['comparisons']:
                    if (comp['method2'] == baseline and 
                        comp['metric'] == metric_key):
                        p_value = comp['p_value']
                        sig_level = comp['sig_level']
                        break
                
                if p_value is not None:
                    cell = f"{p_value:.4f} {sig_level}"
                else:
                    cell = "N/A"
                
                row.append(cell)
            
            lines.append(f"{row[0]:<15} {row[1]:<15} {row[2]:<15}")
    
    lines.append("\n" + "="*80)
    lines.append("Significance levels: *** p<0.001, ** p<0.01, * p<0.05, ns not significant")
    lines.append("="*80)
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description='Mann-Whitney U Test Analysis - Auto-scan results',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--dataset', '-d',
        required=True,
        help='Dataset to analyze (GSM8K, FinQA, TAT-QA, or "all")'
    )
    
    parser.add_argument(
        '--results-dir',
        default='results',
        help='Results directory (default: results)'
    )
    
    parser.add_argument(
        '--output-dir',
        default='results/mann_whitney',
        help='Output directory for analysis results'
    )
    
    args = parser.parse_args()
    
    # Determine which datasets to analyze
    if args.dataset.lower() == 'all':
        datasets_to_analyze = DATASETS
    elif args.dataset in DATASETS:
        datasets_to_analyze = [args.dataset]
    else:
        print(f"❌ Unknown dataset: {args.dataset}")
        print(f"   Available: {', '.join(DATASETS)} or 'all'")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"\n{'#'*70}")
    print(f"# MANN-WHITNEY U TEST ANALYSIS")
    print(f"# Datasets: {', '.join(datasets_to_analyze)}")
    print(f"# Results dir: {args.results_dir}")
    print(f"{'#'*70}")
    
    # Analyze each dataset
    all_results = {}
    
    for dataset in datasets_to_analyze:
        result = analyze_dataset(dataset, args.results_dir)
        if result:
            all_results[dataset] = result
    
    # Create summary table if we have results
    if all_results:
        print(f"\n{'#'*70}")
        print(f"# SUMMARY TABLE")
        print(f"{'#'*70}")
        
        summary = create_summary_table(all_results)
        print(summary)
        
        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save summary text
        summary_file = f"{args.output_dir}/mann_whitney_summary_{timestamp}.txt"
        with open(summary_file, 'w') as f:
            f.write(summary)
        
        # Save detailed JSON
        # Convert numpy arrays to lists for JSON serialization
        json_results = {}
        for dataset, data in all_results.items():
            json_results[dataset] = {
                'dataset': dataset,
                'comparisons': data['comparisons'],
                'method_files': data['method_files'],
                'method_summary': {
                    method: {
                        'n_samples': int(metrics['n_samples']),
                        'mean_accuracy': float(np.mean(metrics['correct'])),
                        'mean_reasoning_length': float(np.mean(metrics['reasoning_length'])),
                        'mean_tokens': float(np.mean(metrics['total_tokens']))
                    }
                    for method, metrics in data['method_data'].items()
                }
            }
        
        json_file = f"{args.output_dir}/mann_whitney_detailed_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"\n{'#'*70}")
        print(f"✅ ANALYSIS COMPLETE")
        print(f"{'#'*70}")
        print(f"\nResults saved to:")
        print(f"  📄 Summary: {summary_file}")
        print(f"  📊 Detailed: {json_file}")
        print(f"\nAnalyzed {len(all_results)} dataset(s)")
        
    else:
        print(f"\n{'#'*70}")
        print(f"❌ NO RESULTS FOUND")
        print(f"{'#'*70}")
        print(f"\nPlease run experiments first:")
        print(f"  python -m mint.cli test --method METHOD --dataset DATASET --limit N")


if __name__ == '__main__':
    main()
