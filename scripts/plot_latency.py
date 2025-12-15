#!/usr/bin/env python3
"""
Latency Visualization for MathCoRL

Generate comparative plots for latency analysis results.

Usage:
    python scripts/plot_latency.py results/latency/GSM8K_latency_*.json
    python scripts/plot_latency.py --input results/latency/ --dataset GSM8K
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


def load_results(file_path: Path) -> Dict[str, Any]:
    """Load latency results from JSON file"""
    with open(file_path, 'r') as f:
        return json.load(f)


def plot_selection_overhead(results: Dict[str, Any], output_dir: Path):
    """
    Plot selection overhead comparison across methods
    
    Args:
        results: Results dictionary with methods data
        output_dir: Output directory for plots
    """
    methods_data = results['methods']
    
    methods = []
    selection_means = []
    selection_stds = []
    colors = []
    
    color_map = {
        'zero_shot': '#2ecc71',
        'random': '#3498db',
        'similarity': '#e74c3c',
        'policy': '#f39c12'
    }
    
    for method_name in ['zero_shot', 'random', 'similarity', 'policy']:
        if method_name in methods_data:
            agg = methods_data[method_name]['aggregated']
            methods.append(method_name.replace('_', ' ').title())
            # Convert to milliseconds
            selection_means.append(agg['selection_mean'] * 1000)
            selection_stds.append(agg['selection_std'] * 1000)
            colors.append(color_map.get(method_name, '#95a5a6'))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(methods))
    bars = ax.bar(x, selection_means, yerr=selection_stds, 
                   color=colors, alpha=0.8, capsize=5, edgecolor='black', linewidth=1.5)
    
    ax.set_xlabel('Selection Method', fontsize=12, fontweight='bold')
    ax.set_ylabel('Selection Time (ms)', fontsize=12, fontweight='bold')
    ax.set_title(f'Selection Overhead Comparison - {results["dataset"]}', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, mean, std in zip(bars, selection_means, selection_stds):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{mean:.2f}ms\n±{std:.2f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    output_file = output_dir / f'{results["dataset"]}_selection_overhead.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def plot_generation_time(results: Dict[str, Any], output_dir: Path):
    """
    Plot LLM generation time comparison
    
    Args:
        results: Results dictionary
        output_dir: Output directory
    """
    methods_data = results['methods']
    
    methods = []
    gen_means = []
    gen_stds = []
    colors = []
    
    color_map = {
        'zero_shot': '#2ecc71',
        'random': '#3498db',
        'similarity': '#e74c3c',
        'policy': '#f39c12'
    }
    
    for method_name in ['zero_shot', 'random', 'similarity', 'policy']:
        if method_name in methods_data:
            agg = methods_data[method_name]['aggregated']
            methods.append(method_name.replace('_', ' ').title())
            gen_means.append(agg['generation_mean'])
            gen_stds.append(agg['generation_std'])
            colors.append(color_map.get(method_name, '#95a5a6'))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(methods))
    bars = ax.bar(x, gen_means, yerr=gen_stds,
                   color=colors, alpha=0.8, capsize=5, edgecolor='black', linewidth=1.5)
    
    ax.set_xlabel('Method', fontsize=12, fontweight='bold')
    ax.set_ylabel('Generation Time (seconds)', fontsize=12, fontweight='bold')
    ax.set_title(f'LLM Generation Time - {results["dataset"]}',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, mean, std in zip(bars, gen_means, gen_stds):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{mean:.2f}s\n±{std:.2f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    output_file = output_dir / f'{results["dataset"]}_generation_time.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def plot_total_latency(results: Dict[str, Any], output_dir: Path):
    """
    Plot total end-to-end latency
    
    Args:
        results: Results dictionary
        output_dir: Output directory
    """
    methods_data = results['methods']
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    methods = []
    total_means = []
    total_stds = []
    success_rates = []
    colors = []
    
    color_map = {
        'zero_shot': '#2ecc71',
        'random': '#3498db',
        'similarity': '#e74c3c',
        'policy': '#f39c12'
    }
    
    for method_name in ['zero_shot', 'random', 'similarity', 'policy']:
        if method_name in methods_data:
            agg = methods_data[method_name]['aggregated']
            methods.append(method_name.replace('_', ' ').title())
            total_means.append(agg['total_mean'])
            total_stds.append(agg['total_std'])
            success_rates.append(agg['success_rate'] * 100)
            colors.append(color_map.get(method_name, '#95a5a6'))
    
    x = np.arange(len(methods))
    bars = ax.bar(x, total_means, yerr=total_stds,
                   color=colors, alpha=0.8, capsize=5, edgecolor='black', linewidth=1.5)
    
    ax.set_xlabel('Method', fontsize=12, fontweight='bold')
    ax.set_ylabel('Total Latency (seconds)', fontsize=12, fontweight='bold')
    ax.set_title(f'End-to-End Latency - {results["dataset"]}',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels with success rate
    for bar, mean, std, success in zip(bars, total_means, total_stds, success_rates):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{mean:.2f}s ±{std:.2f}\n({success:.0f}% success)',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    output_file = output_dir / f'{results["dataset"]}_total_latency.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def plot_latency_breakdown(results: Dict[str, Any], output_dir: Path):
    """
    Stacked bar chart showing latency breakdown
    
    Args:
        results: Results dictionary
        output_dir: Output directory
    """
    methods_data = results['methods']
    
    methods = []
    selection_times = []
    generation_times = []
    
    for method_name in ['zero_shot', 'random', 'similarity', 'policy']:
        if method_name in methods_data:
            agg = methods_data[method_name]['aggregated']
            methods.append(method_name.replace('_', ' ').title())
            selection_times.append(agg['selection_mean'])
            generation_times.append(agg['generation_mean'])
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(methods))
    width = 0.6
    
    # Stack bars
    p1 = ax.bar(x, selection_times, width, label='Selection', 
                color='#3498db', alpha=0.8, edgecolor='black', linewidth=1.5)
    p2 = ax.bar(x, generation_times, width, bottom=selection_times,
                label='Generation + Execution', color='#e74c3c', alpha=0.8, 
                edgecolor='black', linewidth=1.5)
    
    ax.set_xlabel('Method', fontsize=12, fontweight='bold')
    ax.set_ylabel('Time (seconds)', fontsize=12, fontweight='bold')
    ax.set_title(f'Latency Breakdown - {results["dataset"]}',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    # Add percentage labels
    for i, (sel, gen) in enumerate(zip(selection_times, generation_times)):
        total = sel + gen
        if total > 0:
            sel_pct = (sel / total) * 100
            gen_pct = (gen / total) * 100
            
            # Selection percentage (if > 1%)
            if sel_pct > 1:
                ax.text(i, sel/2, f'{sel_pct:.1f}%',
                       ha='center', va='center', fontsize=9, fontweight='bold', color='white')
            
            # Generation percentage
            ax.text(i, sel + gen/2, f'{gen_pct:.1f}%',
                   ha='center', va='center', fontsize=9, fontweight='bold', color='white')
    
    plt.tight_layout()
    output_file = output_dir / f'{results["dataset"]}_latency_breakdown.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def plot_percentile_comparison(results: Dict[str, Any], output_dir: Path):
    """
    Plot latency percentiles (p50, p90, p95, p99)
    
    Args:
        results: Results dictionary
        output_dir: Output directory
    """
    methods_data = results['methods']
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    percentile_names = ['p50', 'p90', 'p95', 'p99']
    x = np.arange(len(percentile_names))
    width = 0.2
    
    color_map = {
        'zero_shot': '#2ecc71',
        'random': '#3498db',
        'similarity': '#e74c3c',
        'policy': '#f39c12'
    }
    
    offset = 0
    for method_name in ['zero_shot', 'random', 'similarity', 'policy']:
        if method_name in methods_data:
            agg = methods_data[method_name]['aggregated']
            percentiles = agg['percentiles']
            
            values = [percentiles[p] for p in percentile_names]
            
            ax.bar(x + offset, values, width, 
                   label=method_name.replace('_', ' ').title(),
                   color=color_map.get(method_name, '#95a5a6'),
                   alpha=0.8, edgecolor='black', linewidth=1)
            
            offset += width
    
    ax.set_xlabel('Percentile', fontsize=12, fontweight='bold')
    ax.set_ylabel('Latency (seconds)', fontsize=12, fontweight='bold')
    ax.set_title(f'Latency Percentiles - {results["dataset"]}',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(['50th', '90th', '95th', '99th'])
    ax.legend(loc='upper left', fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    output_file = output_dir / f'{results["dataset"]}_percentiles.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Generate latency visualization plots'
    )
    
    parser.add_argument('input', nargs='?', type=str,
                       help='Input JSON file or directory')
    
    parser.add_argument('--dataset', type=str,
                       help='Dataset name for filtering files')
    
    parser.add_argument('--output', '-o', type=str, default='results/latency/plots',
                       help='Output directory for plots')
    
    args = parser.parse_args()
    
    # Find input file
    if args.input:
        input_path = Path(args.input)
        if input_path.is_file():
            results_file = input_path
        elif input_path.is_dir():
            # Find latest file for dataset
            pattern = f"{args.dataset}_latency_*.json" if args.dataset else "*_latency_*.json"
            files = sorted(input_path.glob(pattern))
            if not files:
                print(f"No latency files found in {input_path}")
                return 1
            results_file = files[-1]  # Latest
        else:
            print(f"Input not found: {input_path}")
            return 1
    else:
        # Default: find latest in results/latency
        latency_dir = Path('results/latency')
        files = sorted(latency_dir.glob('*_latency_*.json'))
        if not files:
            print("No latency files found in results/latency/")
            return 1
        results_file = files[-1]
    
    print(f"Loading results from: {results_file}")
    results = load_results(results_file)
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nGenerating plots for {results['dataset']}...")
    print(f"Methods: {', '.join(results['methods'].keys())}")
    print(f"Samples: {results['num_samples']}\n")
    
    # Generate all plots
    plot_selection_overhead(results, output_dir)
    plot_generation_time(results, output_dir)
    plot_total_latency(results, output_dir)
    plot_latency_breakdown(results, output_dir)
    plot_percentile_comparison(results, output_dir)
    
    print(f"\n✅ All plots saved to: {output_dir}")


if __name__ == '__main__':
    sys.exit(main())
