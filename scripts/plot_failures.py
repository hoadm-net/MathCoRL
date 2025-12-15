#!/usr/bin/env python3
"""
Failure Case Visualization for MathCoRL

Generates plots for failure analysis across methods and error types.

Usage:
    python scripts/plot_failures.py --results results/failures/
    python scripts/plot_failures.py --output results/failures/plots/
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any, List
import argparse
import numpy as np


def plot_error_type_distribution(data: Dict[str, Any], output_dir: Path):
    """Plot pie chart and bar chart of error types"""
    error_breakdown = data['overall']['error_breakdown']
    
    # Filter non-zero errors
    error_types = [k.replace('_', ' ').title() for k, v in error_breakdown.items() if v > 0]
    error_counts = [v for v in error_breakdown.values() if v > 0]
    
    if not error_counts:
        print("No errors to plot")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Pie chart
    colors = sns.color_palette("husl", len(error_types))
    axes[0].pie(error_counts, labels=error_types, autopct='%1.1f%%', 
                startangle=90, colors=colors)
    axes[0].set_title('Error Type Distribution', fontsize=14, fontweight='bold')
    
    # Bar chart
    axes[1].barh(error_types, error_counts, color=colors)
    axes[1].set_xlabel('Number of Failures', fontsize=12)
    axes[1].set_title('Error Type Counts', fontsize=14, fontweight='bold')
    axes[1].grid(axis='x', alpha=0.3)
    
    # Add count labels
    for i, count in enumerate(error_counts):
        axes[1].text(count + max(error_counts)*0.02, i, str(count), 
                     va='center', fontsize=10)
    
    plt.tight_layout()
    output_file = output_dir / 'error_type_distribution.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {output_file}")


def plot_method_failure_rates(data: Dict[str, Any], output_dir: Path):
    """Plot failure rates by method"""
    by_method = data['by_method']
    
    methods = []
    failure_rates = []
    success_rates = []
    total_problems = []
    
    for method, stats in by_method.items():
        methods.append(method)
        failure_rates.append(stats['failure_rate'] * 100)
        success_rates.append(stats['success_rate'] * 100)
        total_problems.append(stats['total_problems'])
    
    # Sort by failure rate
    sorted_indices = sorted(range(len(failure_rates)), key=lambda i: failure_rates[i], reverse=True)
    methods = [methods[i] for i in sorted_indices]
    failure_rates = [failure_rates[i] for i in sorted_indices]
    success_rates = [success_rates[i] for i in sorted_indices]
    total_problems = [total_problems[i] for i in sorted_indices]
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Failure rates
    colors_fail = ['#e74c3c' if fr > 10 else '#3498db' if fr > 0 else '#2ecc71' 
                   for fr in failure_rates]
    bars1 = axes[0].barh(methods, failure_rates, color=colors_fail)
    axes[0].set_xlabel('Failure Rate (%)', fontsize=12)
    axes[0].set_title('Failure Rate by Method', fontsize=14, fontweight='bold')
    axes[0].grid(axis='x', alpha=0.3)
    
    # Add labels
    for i, (fr, total) in enumerate(zip(failure_rates, total_problems)):
        axes[0].text(fr + 0.5, i, f'{fr:.1f}% (n={total})', 
                     va='center', fontsize=9)
    
    # Success vs Failure stacked bar
    axes[1].barh(methods, success_rates, color='#2ecc71', label='Success')
    axes[1].barh(methods, failure_rates, left=success_rates, color='#e74c3c', label='Failure')
    axes[1].set_xlabel('Percentage', fontsize=12)
    axes[1].set_xlim(0, 100)
    axes[1].set_title('Success vs Failure Rate by Method', fontsize=14, fontweight='bold')
    axes[1].legend(loc='lower right')
    axes[1].grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    output_file = output_dir / 'method_failure_rates.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {output_file}")


def plot_method_error_breakdown(data: Dict[str, Any], failure_cases: List[Dict], output_dir: Path):
    """Plot error type breakdown per method"""
    # Count error types per method
    from collections import defaultdict
    method_errors = defaultdict(lambda: defaultdict(int))
    
    for case in failure_cases:
        method = case['method']
        error_type = case['error_type'].replace('_', ' ').title()
        method_errors[method][error_type] += 1
    
    # Filter methods with failures
    methods_with_failures = [m for m in method_errors.keys()]
    if not methods_with_failures:
        print("No failures to plot by method")
        return
    
    # Get all error types
    all_error_types = set()
    for errors in method_errors.values():
        all_error_types.update(errors.keys())
    error_types = sorted(all_error_types)
    
    # Prepare data matrix
    data_matrix = []
    for method in methods_with_failures:
        row = [method_errors[method].get(et, 0) for et in error_types]
        data_matrix.append(row)
    
    # Plot stacked bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(methods_with_failures))
    width = 0.6
    colors = sns.color_palette("husl", len(error_types))
    
    bottom = np.zeros(len(methods_with_failures))
    for i, error_type in enumerate(error_types):
        counts = [method_errors[method].get(error_type, 0) for method in methods_with_failures]
        ax.bar(x, counts, width, label=error_type, bottom=bottom, color=colors[i])
        bottom += counts
    
    ax.set_xlabel('Method', fontsize=12)
    ax.set_ylabel('Number of Failures', fontsize=12)
    ax.set_title('Error Type Breakdown by Method', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(methods_with_failures, rotation=45, ha='right')
    ax.legend(title='Error Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    output_file = output_dir / 'method_error_breakdown.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {output_file}")


def plot_failure_patterns(data: Dict[str, Any], output_dir: Path):
    """Plot top failure patterns"""
    patterns = data.get('top_patterns', [])
    
    if not patterns or len(patterns) < 2:
        print("Not enough failure patterns to plot")
        return
    
    # Take top 10
    patterns = patterns[:10]
    pattern_names = [p[0][:50] + '...' if len(p[0]) > 50 else p[0] for p, _ in patterns]
    pattern_counts = [count for _, count in patterns]
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    colors = sns.color_palette("YlOrRd_r", len(pattern_names))
    bars = ax.barh(pattern_names, pattern_counts, color=colors)
    
    ax.set_xlabel('Occurrences', fontsize=12)
    ax.set_title('Top 10 Most Common Failure Patterns', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    # Add count labels
    for i, count in enumerate(pattern_counts):
        ax.text(count + max(pattern_counts)*0.02, i, str(count), 
                va='center', fontsize=10)
    
    plt.tight_layout()
    output_file = output_dir / 'failure_patterns.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {output_file}")


def plot_overall_summary(data: Dict[str, Any], output_dir: Path):
    """Plot overall success/failure summary"""
    overall = data['overall']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Success vs Failure pie chart
    success_count = overall['total_problems'] - overall['total_failures']
    failure_count = overall['total_failures']
    
    axes[0, 0].pie([success_count, failure_count], 
                   labels=['Success', 'Failure'],
                   autopct='%1.1f%%',
                   colors=['#2ecc71', '#e74c3c'],
                   startangle=90)
    axes[0, 0].set_title('Overall Success vs Failure', fontsize=12, fontweight='bold')
    
    # 2. Error breakdown
    error_breakdown = overall['error_breakdown']
    error_types = [k.replace('_', ' ').title() for k, v in error_breakdown.items() if v > 0]
    error_counts = [v for v in error_breakdown.values() if v > 0]
    
    if error_counts:
        axes[0, 1].pie(error_counts, labels=error_types, autopct='%1.1f%%',
                       colors=sns.color_palette("husl", len(error_types)),
                       startangle=90)
        axes[0, 1].set_title('Error Type Distribution', fontsize=12, fontweight='bold')
    else:
        axes[0, 1].text(0.5, 0.5, 'No Errors', ha='center', va='center', fontsize=14)
        axes[0, 1].set_title('Error Type Distribution', fontsize=12, fontweight='bold')
    
    # 3. Summary statistics text
    axes[1, 0].axis('off')
    summary_text = f"""
    OVERALL STATISTICS
    
    Total Problems: {overall['total_problems']:,}
    Total Failures: {overall['total_failures']:,}
    
    Success Rate: {overall['success_rate']*100:.2f}%
    Failure Rate: {overall['failure_rate']*100:.2f}%
    
    ERROR BREAKDOWN
    """
    for error_type, count in error_breakdown.items():
        if count > 0:
            pct = 100 * count / max(overall['total_failures'], 1)
            summary_text += f"\n  {error_type.replace('_', ' ').title()}: {count} ({pct:.1f}%)"
    
    axes[1, 0].text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
                    va='center', ha='left')
    
    # 4. Method comparison
    by_method = data['by_method']
    methods = list(by_method.keys())[:8]  # Top 8 methods
    failure_counts_by_method = [by_method[m]['total_failures'] for m in methods]
    
    axes[1, 1].bar(range(len(methods)), failure_counts_by_method,
                   color=sns.color_palette("coolwarm", len(methods)))
    axes[1, 1].set_xticks(range(len(methods)))
    axes[1, 1].set_xticklabels(methods, rotation=45, ha='right', fontsize=9)
    axes[1, 1].set_ylabel('Failure Count', fontsize=10)
    axes[1, 1].set_title('Failures by Method', fontsize=12, fontweight='bold')
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    output_file = output_dir / 'overall_summary.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Generate failure analysis visualizations")
    parser.add_argument('--results', default='results/failures/',
                        help='Path to failure analysis results directory')
    parser.add_argument('--output', default='results/failures/plots/',
                        help='Output directory for plots')
    
    args = parser.parse_args()
    
    results_dir = Path(args.results)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    overall_file = results_dir / 'failure_analysis_overall.json'
    cases_file = results_dir / 'failure_cases.json'
    
    if not overall_file.exists():
        print(f"Error: {overall_file} not found")
        print("Run failure_analysis.py first to generate results")
        return
    
    print(f"Loading: {overall_file}")
    with open(overall_file) as f:
        overall_data = json.load(f)
    
    failure_cases = []
    if cases_file.exists():
        print(f"Loading: {cases_file}")
        with open(cases_file) as f:
            failure_cases = json.load(f)
    
    print(f"\nGenerating plots...")
    print("="*70)
    
    # Generate all plots
    plot_overall_summary(overall_data, output_dir)
    plot_error_type_distribution(overall_data, output_dir)
    plot_method_failure_rates(overall_data, output_dir)
    
    if failure_cases:
        plot_method_error_breakdown(overall_data, failure_cases, output_dir)
    
    plot_failure_patterns(overall_data, output_dir)
    
    print("="*70)
    print(f"\n✅ All plots saved to: {output_dir}\n")


if __name__ == "__main__":
    main()
