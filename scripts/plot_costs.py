#!/usr/bin/env python3
"""
Cost Visualization for MathCoRL

Generate plots for token cost analysis.

Usage:
    python scripts/plot_costs.py --logs logs/api_usage.jsonl
    python scripts/plot_costs.py --input results/cost/cost_analysis_*.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.cost_analysis import CostAnalyzer

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


def plot_cost_by_method(analyzer: CostAnalyzer, output_dir: Path):
    """Plot total cost by method"""
    method_metrics = analyzer.analyze_by_method()
    
    methods = []
    costs = []
    colors = []
    
    color_map = {
        'FPP': '#3498db',
        'CoT': '#2ecc71',
        'PoT': '#e74c3c',
        'Zero-Shot': '#f39c12',
        'ICRL-CandidateGen': '#9b59b6',
        'ICRL-Evaluator': '#1abc9c'
    }
    
    for method, metrics in sorted(method_metrics.items(), key=lambda x: x[1].total_cost, reverse=True):
        methods.append(method)
        costs.append(metrics.total_cost)
        colors.append(color_map.get(method, '#95a5a6'))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars = ax.barh(methods, costs, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    ax.set_xlabel('Total Cost ($)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Method', fontsize=12, fontweight='bold')
    ax.set_title('Total API Cost by Method', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for bar, cost in zip(bars, costs):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2,
                f'${cost:.4f}',
                ha='left', va='center', fontsize=9, fontweight='bold', 
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    output_file = output_dir / 'cost_by_method.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def plot_cost_per_request(analyzer: CostAnalyzer, output_dir: Path):
    """Plot average cost per request"""
    method_metrics = analyzer.analyze_by_method()
    
    methods = []
    avg_costs = []
    colors = []
    
    color_map = {
        'FPP': '#3498db',
        'CoT': '#2ecc71',
        'PoT': '#e74c3c',
        'Zero-Shot': '#f39c12',
        'ICRL-CandidateGen': '#9b59b6',
        'ICRL-Evaluator': '#1abc9c'
    }
    
    for method, metrics in sorted(method_metrics.items(), key=lambda x: x[1].avg_cost_per_request, reverse=True):
        methods.append(method)
        avg_costs.append(metrics.avg_cost_per_request)
        colors.append(color_map.get(method, '#95a5a6'))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars = ax.barh(methods, avg_costs, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    ax.set_xlabel('Average Cost per Request ($)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Method', fontsize=12, fontweight='bold')
    ax.set_title('Average Cost Efficiency by Method', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for bar, cost in zip(bars, avg_costs):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2,
                f'${cost:.6f}',
                ha='left', va='center', fontsize=9, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    output_file = output_dir / 'cost_per_request.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def plot_token_usage(analyzer: CostAnalyzer, output_dir: Path):
    """Plot token usage breakdown"""
    method_metrics = analyzer.analyze_by_method()
    
    methods = []
    input_tokens = []
    output_tokens = []
    
    for method, metrics in sorted(method_metrics.items(), key=lambda x: x[1].total_tokens, reverse=True):
        methods.append(method)
        input_tokens.append(metrics.avg_input_tokens)
        output_tokens.append(metrics.avg_output_tokens)
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    x = np.arange(len(methods))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, input_tokens, width, label='Input Tokens',
                   color='#3498db', alpha=0.8, edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x + width/2, output_tokens, width, label='Output Tokens',
                   color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=1.5)
    
    ax.set_xlabel('Method', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Tokens per Request', fontsize=12, fontweight='bold')
    ax.set_title('Token Usage by Method', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha='right')
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    output_file = output_dir / 'token_usage.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def plot_cost_breakdown(analyzer: CostAnalyzer, output_dir: Path):
    """Plot input/output cost breakdown"""
    model_metrics = analyzer.analyze_by_model()
    
    # Focus on the primary model
    if 'gpt-4o-mini' in model_metrics:
        metrics = model_metrics['gpt-4o-mini']
    else:
        # Use first available model
        metrics = list(model_metrics.values())[0]
    
    labels = ['Input Tokens', 'Output Tokens']
    costs = [metrics.input_cost, metrics.output_cost]
    colors = ['#3498db', '#e74c3c']
    explode = (0.05, 0.05)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    wedges, texts, autotexts = ax.pie(costs, explode=explode, labels=labels,
                                        colors=colors, autopct='%1.1f%%',
                                        shadow=True, startangle=90,
                                        textprops={'fontsize': 12, 'fontweight': 'bold'})
    
    ax.set_title(f'Cost Breakdown - {metrics.model}\nTotal: ${metrics.total_cost:.4f}',
                fontsize=14, fontweight='bold')
    
    # Add legend with values
    legend_labels = [
        f'Input: ${metrics.input_cost:.4f}',
        f'Output: ${metrics.output_cost:.4f}'
    ]
    ax.legend(legend_labels, loc='best', fontsize=11)
    
    plt.tight_layout()
    output_file = output_dir / 'cost_breakdown.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def plot_cost_efficiency(analyzer: CostAnalyzer, output_dir: Path):
    """Plot cost per success vs success rate"""
    method_metrics = analyzer.analyze_by_method()
    
    methods = []
    cost_per_success = []
    success_rates = []
    colors = []
    sizes = []
    
    color_map = {
        'FPP': '#3498db',
        'CoT': '#2ecc71',
        'PoT': '#e74c3c',
        'Zero-Shot': '#f39c12',
        'ICRL-CandidateGen': '#9b59b6',
        'ICRL-Evaluator': '#1abc9c'
    }
    
    for method, metrics in method_metrics.items():
        if metrics.cost_per_success != float('inf'):
            methods.append(method)
            cost_per_success.append(metrics.cost_per_success * 1000)  # Convert to cents
            success_rates.append(metrics.success_rate * 100)
            colors.append(color_map.get(method, '#95a5a6'))
            sizes.append(metrics.num_requests * 3)  # Size by request count
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    scatter = ax.scatter(success_rates, cost_per_success, s=sizes, c=colors,
                        alpha=0.6, edgecolors='black', linewidth=2)
    
    # Add method labels
    for i, method in enumerate(methods):
        ax.annotate(method, (success_rates[i], cost_per_success[i]),
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    ax.set_xlabel('Success Rate (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cost per Success (cents)', fontsize=12, fontweight='bold')
    ax.set_title('Cost Efficiency: Success Rate vs Cost per Success\n(bubble size = request count)',
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add ideal region annotation
    ax.axhline(y=np.median(cost_per_success), color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=np.median(success_rates), color='gray', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    output_file = output_dir / 'cost_efficiency.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def plot_requests_distribution(analyzer: CostAnalyzer, output_dir: Path):
    """Plot request count distribution"""
    method_metrics = analyzer.analyze_by_method()
    
    methods = []
    requests = []
    colors = []
    
    color_map = {
        'FPP': '#3498db',
        'CoT': '#2ecc71',
        'PoT': '#e74c3c',
        'Zero-Shot': '#f39c12',
        'ICRL-CandidateGen': '#9b59b6',
        'ICRL-Evaluator': '#1abc9c'
    }
    
    for method, metrics in sorted(method_metrics.items(), key=lambda x: x[1].num_requests, reverse=True):
        methods.append(method)
        requests.append(metrics.num_requests)
        colors.append(color_map.get(method, '#95a5a6'))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    wedges, texts, autotexts = ax.pie(requests, labels=methods, colors=colors,
                                        autopct='%1.1f%%', shadow=True, startangle=90,
                                        textprops={'fontsize': 10, 'fontweight': 'bold'})
    
    total_requests = sum(requests)
    ax.set_title(f'Request Distribution by Method\nTotal Requests: {total_requests}',
                fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    output_file = output_dir / 'requests_distribution.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Generate cost visualization plots'
    )
    
    parser.add_argument('--logs', type=str, default='logs/api_usage.jsonl',
                       help='Path to API usage log file')
    
    parser.add_argument('--output', '-o', type=str, default='results/cost/plots',
                       help='Output directory for plots')
    
    parser.add_argument('--method', '-m', type=str,
                       help='Filter by method')
    
    parser.add_argument('--hours', type=int,
                       help='Analyze last N hours')
    
    args = parser.parse_args()
    
    try:
        # Initialize analyzer
        print(f"Loading logs from: {args.logs}")
        analyzer = CostAnalyzer(args.logs)
        
        # Filter if requested
        if args.method or args.hours:
            analyzer.logs = analyzer.filter_logs(method=args.method, hours=args.hours)
            print(f"Filtered to {len(analyzer.logs)} entries")
        
        # Create output directory
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nGenerating plots...")
        
        # Generate all plots
        plot_cost_by_method(analyzer, output_dir)
        plot_cost_per_request(analyzer, output_dir)
        plot_token_usage(analyzer, output_dir)
        plot_cost_breakdown(analyzer, output_dir)
        plot_cost_efficiency(analyzer, output_dir)
        plot_requests_distribution(analyzer, output_dir)
        
        print(f"\n✅ All plots saved to: {output_dir}")
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
