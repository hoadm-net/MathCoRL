#!/usr/bin/env python3
"""
Visualization module for prototype library coverage analysis.

Generates plots showing:
- Coverage rate by dataset
- Function usage frequency distribution
- Category breakdown (arithmetic, financial, table, etc.)
- Usage heatmap across datasets

Usage:
    python scripts/plot_prototypes.py --results results/prototype_analysis/GSM8K_analysis.json
    python scripts/plot_prototypes.py --all-datasets
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any
import logging

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    print("Warning: matplotlib/seaborn not available. Install with: pip install matplotlib seaborn")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style
if PLOTTING_AVAILABLE:
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 10


def plot_coverage_comparison(results_dict: Dict[str, Dict], output_path: Path):
    """Bar chart comparing coverage rates across datasets."""
    if not PLOTTING_AVAILABLE:
        return
    
    datasets = list(results_dict.keys())
    coverage_rates = [results_dict[d]['coverage_rate'] * 100 for d in datasets]
    used_counts = [results_dict[d]['used_functions'] for d in datasets]
    available = results_dict[datasets[0]]['available_functions']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Coverage rate plot
    colors = ['#2ecc71' if c > 70 else '#f39c12' if c > 50 else '#e74c3c' for c in coverage_rates]
    bars1 = ax1.bar(datasets, coverage_rates, color=colors, alpha=0.8, edgecolor='black')
    
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontweight='bold')
    
    ax1.set_ylabel('Coverage Rate (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Function Prototype Coverage by Dataset', fontsize=13, fontweight='bold')
    ax1.set_ylim(0, 110)
    ax1.axhline(y=100, color='green', linestyle='--', alpha=0.3, label='100% coverage')
    ax1.legend()
    
    # Used vs available plot
    x = np.arange(len(datasets))
    width = 0.35
    
    bars2 = ax2.bar(x - width/2, [available]*len(datasets), width, label='Available', 
                    color='lightgray', alpha=0.7, edgecolor='black')
    bars3 = ax2.bar(x + width/2, used_counts, width, label='Used', 
                    color='skyblue', alpha=0.8, edgecolor='black')
    
    ax2.set_xlabel('Dataset', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Number of Functions', fontsize=11, fontweight='bold')
    ax2.set_title('Available vs Used Functions', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(datasets)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved coverage comparison to {output_path}")
    plt.close()


def plot_usage_frequency(results: Dict[str, Any], dataset_name: str, output_path: Path):
    """Histogram showing function usage frequency distribution."""
    if not PLOTTING_AVAILABLE:
        return
    
    usage_counts = list(results['function_usage'].values())
    
    if not usage_counts:
        logger.warning(f"No usage data for {dataset_name}")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create histogram
    n, bins, patches = ax.hist(usage_counts, bins=20, color='steelblue', 
                               alpha=0.7, edgecolor='black')
    
    # Color patches based on frequency
    for i, patch in enumerate(patches):
        if bins[i] < results['problems_with_code'] * 0.05:
            patch.set_facecolor('#e74c3c')  # Low frequency - red
        elif bins[i] < results['problems_with_code'] * 0.2:
            patch.set_facecolor('#f39c12')  # Medium - orange
        else:
            patch.set_facecolor('#2ecc71')  # High - green
    
    ax.set_xlabel('Usage Count', fontsize=11, fontweight='bold')
    ax.set_ylabel('Number of Functions', fontsize=11, fontweight='bold')
    ax.set_title(f'Function Usage Frequency Distribution - {dataset_name}', 
                 fontsize=13, fontweight='bold')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ecc71', label='High (>20%)'),
        Patch(facecolor='#f39c12', label='Medium (5-20%)'),
        Patch(facecolor='#e74c3c', label='Low (<5%)')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    # Add statistics text
    textstr = f'Total functions used: {len(usage_counts)}\n'
    textstr += f'Mean usage: {np.mean(usage_counts):.1f}\n'
    textstr += f'Median usage: {np.median(usage_counts):.1f}'
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved usage frequency plot to {output_path}")
    plt.close()


def plot_top_functions(results: Dict[str, Any], dataset_name: str, output_path: Path, top_n: int = 15):
    """Horizontal bar chart of most frequently used functions."""
    if not PLOTTING_AVAILABLE:
        return
    
    most_common = results['most_common'][:top_n]
    
    if not most_common:
        logger.warning(f"No function usage data for {dataset_name}")
        return
    
    functions = [f[0] for f in most_common]
    counts = [f[1] for f in most_common]
    usage_rates = [c / results['problems_with_code'] * 100 for c in counts]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create horizontal bar chart
    colors = ['#2ecc71' if r > 20 else '#f39c12' if r > 5 else '#3498db' for r in usage_rates]
    bars = ax.barh(functions, usage_rates, color=colors, alpha=0.8, edgecolor='black')
    
    # Add value labels
    for i, (bar, count, rate) in enumerate(zip(bars, counts, usage_rates)):
        ax.text(rate + 1, i, f'{count} ({rate:.1f}%)', 
                va='center', fontsize=9, fontweight='bold')
    
    ax.set_xlabel('Usage Rate (%)', fontsize=11, fontweight='bold')
    ax.set_title(f'Top {top_n} Most Used Functions - {dataset_name}', 
                 fontsize=13, fontweight='bold')
    ax.invert_yaxis()  # Highest at top
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved top functions plot to {output_path}")
    plt.close()


def plot_usage_heatmap(results_dict: Dict[str, Dict], output_path: Path, top_n: int = 20):
    """Heatmap showing function usage across datasets."""
    if not PLOTTING_AVAILABLE:
        return
    
    # Get all functions used across all datasets
    all_functions = set()
    for results in results_dict.values():
        all_functions.update(results['function_usage'].keys())
    
    # Get top N most commonly used functions across all datasets
    total_usage = {}
    for func in all_functions:
        total_usage[func] = sum(
            results['function_usage'].get(func, 0) 
            for results in results_dict.values()
        )
    
    top_functions = sorted(total_usage.items(), key=lambda x: x[1], reverse=True)[:top_n]
    top_func_names = [f[0] for f in top_functions]
    
    # Create matrix
    datasets = list(results_dict.keys())
    matrix = []
    
    for func in top_func_names:
        row = []
        for dataset in datasets:
            results = results_dict[dataset]
            count = results['function_usage'].get(func, 0)
            usage_rate = count / results['problems_with_code'] * 100 if results['problems_with_code'] > 0 else 0
            row.append(usage_rate)
        matrix.append(row)
    
    matrix = np.array(matrix)
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(len(datasets)*1.5, top_n*0.4))
    
    im = sns.heatmap(matrix, annot=True, fmt='.1f', cmap='YlOrRd', 
                     xticklabels=datasets, yticklabels=top_func_names,
                     cbar_kws={'label': 'Usage Rate (%)'}, ax=ax)
    
    ax.set_title(f'Top {top_n} Function Usage Across Datasets', 
                 fontsize=13, fontweight='bold', pad=15)
    ax.set_ylabel('Function', fontsize=11, fontweight='bold')
    ax.set_xlabel('Dataset', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved usage heatmap to {output_path}")
    plt.close()


def plot_prototype_analysis(results_dict: Dict[str, Dict], output_dir: Path):
    """Generate all prototype analysis plots.
    
    Args:
        results_dict: Dictionary mapping dataset names to analysis results
        output_dir: Directory to save plots
    """
    if not PLOTTING_AVAILABLE:
        logger.warning("Plotting libraries not available. Skipping visualization.")
        return
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Generating prototype analysis plots in {output_dir}")
    
    # Plot 1: Coverage comparison
    if len(results_dict) > 1:
        plot_coverage_comparison(results_dict, output_dir / "coverage_comparison.png")
    
    # Plot 2-3: Per-dataset plots
    for dataset, results in results_dict.items():
        plot_usage_frequency(results, dataset, output_dir / f"{dataset}_usage_frequency.png")
        plot_top_functions(results, dataset, output_dir / f"{dataset}_top_functions.png")
    
    # Plot 4: Cross-dataset heatmap
    if len(results_dict) > 1:
        plot_usage_heatmap(results_dict, output_dir / "usage_heatmap.png")
    
    logger.info("✅ All plots generated successfully")


def main():
    """Main function for standalone plotting."""
    parser = argparse.ArgumentParser(description='Generate prototype analysis plots')
    
    parser.add_argument('--results', type=str, help='Path to analysis results JSON file')
    parser.add_argument('--all-datasets', action='store_true', help='Load results for all datasets')
    parser.add_argument('--results-dir', type=str, default='results/prototype_analysis',
                       help='Directory containing result files')
    parser.add_argument('--output-dir', type=str, default='results/prototype_analysis/plots',
                       help='Output directory for plots')
    
    args = parser.parse_args()
    
    # Load results
    results_dict = {}
    
    if args.results:
        dataset_name = Path(args.results).stem.replace('_analysis', '')
        with open(args.results, 'r') as f:
            results_dict[dataset_name] = json.load(f)
    elif args.all_datasets:
        results_dir = Path(args.results_dir)
        for result_file in results_dir.glob('*_analysis.json'):
            dataset_name = result_file.stem.replace('_analysis', '')
            with open(result_file, 'r') as f:
                results_dict[dataset_name] = json.load(f)
    else:
        parser.error("Must specify either --results or --all-datasets")
    
    if not results_dict:
        logger.error("No results found")
        sys.exit(1)
    
    # Generate plots
    output_dir = Path(args.output_dir)
    plot_prototype_analysis(results_dict, output_dir)


if __name__ == '__main__':
    main()
