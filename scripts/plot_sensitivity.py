#!/usr/bin/env python3
"""
Visualization module for reward sensitivity analysis results.

Generates plots showing:
- Accuracy comparison across reward configurations
- Training curves for each configuration
- Reward component breakdown
- Sensitivity heatmap

Usage:
    python scripts/plot_sensitivity.py --results results/reward_sensitivity/GSM8K_all_configs.json
    python scripts/plot_sensitivity.py --results-dir results/reward_sensitivity --dataset GSM8K
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


def plot_accuracy_comparison(results: Dict[str, Any], output_path: Path):
    """Bar chart comparing accuracy across configurations."""
    if not PLOTTING_AVAILABLE:
        return
    
    config_names = []
    accuracies = []
    colors = []
    
    # Color mapping
    color_map = {
        'baseline': '#2ecc71',
        'accuracy_focused': '#e74c3c',
        'diversity_focused': '#3498db',
        'balanced': '#f39c12',
        'length_penalized': '#9b59b6'
    }
    
    for config_name, result in results.items():
        config_names.append(config_name.replace('_', ' ').title())
        accuracies.append(result['final_metrics']['accuracy'] * 100)
        colors.append(color_map.get(config_name, '#95a5a6'))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(config_names, accuracies, color=colors, alpha=0.8, edgecolor='black')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontweight='bold')
    
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Policy Accuracy by Reward Configuration', fontsize=14, fontweight='bold')
    ax.set_ylim(0, max(accuracies) * 1.15)
    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved accuracy comparison to {output_path}")
    plt.close()


def plot_training_curves(results: Dict[str, Any], output_path: Path):
    """Line plot showing training curves for each configuration."""
    if not PLOTTING_AVAILABLE:
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    color_map = {
        'baseline': '#2ecc71',
        'accuracy_focused': '#e74c3c',
        'diversity_focused': '#3498db',
        'balanced': '#f39c12',
        'length_penalized': '#9b59b6'
    }
    
    for config_name, result in results.items():
        history = result['training_history']
        epochs = range(1, len(history) + 1)
        
        # Extract metrics
        losses = [h['loss'] for h in history]
        rewards = [h['reward'] for h in history]
        
        label = config_name.replace('_', ' ').title()
        color = color_map.get(config_name, '#95a5a6')
        
        # Plot loss
        ax1.plot(epochs, losses, marker='o', label=label, color=color, linewidth=2, markersize=6)
        
        # Plot reward
        ax2.plot(epochs, rewards, marker='s', label=label, color=color, linewidth=2, markersize=6)
    
    # Configure loss plot
    ax1.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Training Loss', fontsize=11, fontweight='bold')
    ax1.set_title('Training Loss Over Epochs', fontsize=12, fontweight='bold')
    ax1.legend(loc='best', framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    
    # Configure reward plot
    ax2.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Average Reward', fontsize=11, fontweight='bold')
    ax2.set_title('Average Reward Over Epochs', fontsize=12, fontweight='bold')
    ax2.legend(loc='best', framealpha=0.9)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved training curves to {output_path}")
    plt.close()


def plot_reward_weights_heatmap(results: Dict[str, Any], output_path: Path):
    """Heatmap showing reward weight configurations and resulting accuracy."""
    if not PLOTTING_AVAILABLE:
        return
    
    # Prepare data
    config_names = []
    weights_matrix = []
    accuracies = []
    
    for config_name, result in results.items():
        config_names.append(config_name.replace('_', '\n').title())
        config = result['config']
        weights_matrix.append([
            config['lambda_accuracy'],
            config['lambda_similarity'],
            config['lambda_diversity']
        ])
        accuracies.append(result['final_metrics']['accuracy'])
    
    weights_matrix = np.array(weights_matrix).T
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 4))
    
    im = ax.imshow(weights_matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
    
    # Set ticks
    ax.set_xticks(np.arange(len(config_names)))
    ax.set_yticks(np.arange(3))
    ax.set_xticklabels(config_names)
    ax.set_yticklabels(['λ_accuracy', 'λ_similarity', 'λ_diversity'])
    
    # Rotate labels
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center")
    
    # Add text annotations
    for i in range(3):
        for j in range(len(config_names)):
            text = ax.text(j, i, f'{weights_matrix[i, j]:.2f}',
                          ha="center", va="center", color="black", fontweight='bold')
    
    # Add accuracy row below
    ax_acc = fig.add_subplot(4, 1, 4)
    ax_acc.bar(range(len(config_names)), [a * 100 for a in accuracies], 
               color='skyblue', alpha=0.7, edgecolor='black')
    ax_acc.set_xticks(range(len(config_names)))
    ax_acc.set_xticklabels(config_names, rotation=0)
    ax_acc.set_ylabel('Accuracy (%)', fontsize=10, fontweight='bold')
    ax_acc.set_ylim(0, max(accuracies) * 115)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, orientation='vertical', pad=0.02)
    cbar.set_label('Weight Value', rotation=270, labelpad=20, fontweight='bold')
    
    ax.set_title('Reward Weight Configurations', fontsize=13, fontweight='bold', pad=15)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved weight heatmap to {output_path}")
    plt.close()


def plot_weight_vs_accuracy(results: Dict[str, Any], output_path: Path):
    """Scatter plot showing relationship between weights and accuracy."""
    if not PLOTTING_AVAILABLE:
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for config_name, result in results.items():
        config = result['config']
        accuracy = result['final_metrics']['accuracy'] * 100
        
        # Plot for each weight component
        axes[0].scatter(config['lambda_accuracy'], accuracy, s=100, alpha=0.7, 
                       label=config_name.replace('_', ' ').title())
        axes[1].scatter(config['lambda_similarity'], accuracy, s=100, alpha=0.7)
        axes[2].scatter(config['lambda_diversity'], accuracy, s=100, alpha=0.7)
    
    # Configure subplots
    axes[0].set_xlabel('λ_accuracy', fontsize=11, fontweight='bold')
    axes[0].set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
    axes[0].set_title('Accuracy vs λ_accuracy', fontsize=12, fontweight='bold')
    axes[0].legend(loc='best', fontsize=8)
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_xlabel('λ_similarity', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
    axes[1].set_title('Accuracy vs λ_similarity', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    axes[2].set_xlabel('λ_diversity', fontsize=11, fontweight='bold')
    axes[2].set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
    axes[2].set_title('Accuracy vs λ_diversity', fontsize=12, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved weight-accuracy scatter to {output_path}")
    plt.close()


def plot_sensitivity_results(results: Dict[str, Any], output_dir: Path):
    """Generate all sensitivity analysis plots.
    
    Args:
        results: Dictionary with results for each configuration
        output_dir: Directory to save plots
    """
    if not PLOTTING_AVAILABLE:
        logger.warning("Plotting libraries not available. Skipping visualization.")
        return
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Generating sensitivity analysis plots in {output_dir}")
    
    # Generate plots
    plot_accuracy_comparison(results, output_dir / "accuracy_comparison.png")
    plot_training_curves(results, output_dir / "training_curves.png")
    plot_reward_weights_heatmap(results, output_dir / "weight_heatmap.png")
    plot_weight_vs_accuracy(results, output_dir / "weight_accuracy_scatter.png")
    
    logger.info("✅ All plots generated successfully")


def main():
    """Main function for standalone plotting."""
    parser = argparse.ArgumentParser(description='Generate sensitivity analysis plots')
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--results', type=str, help='Path to combined results JSON file')
    group.add_argument('--results-dir', type=str, help='Directory containing result files')
    
    parser.add_argument('--dataset', type=str, help='Dataset name (required with --results-dir)')
    parser.add_argument('--output-dir', type=str, default='results/reward_sensitivity/plots',
                       help='Output directory for plots')
    
    args = parser.parse_args()
    
    # Load results
    if args.results:
        with open(args.results, 'r') as f:
            results = json.load(f)
    else:
        if not args.dataset:
            parser.error("--dataset is required when using --results-dir")
        
        results_dir = Path(args.results_dir)
        results = {}
        
        # Load individual config files
        for result_file in results_dir.glob(f"{args.dataset}_*.json"):
            if 'all_configs' in result_file.name:
                continue
            with open(result_file, 'r') as f:
                result = json.load(f)
                results[result['config_name']] = result
    
    if not results:
        logger.error("No results found")
        sys.exit(1)
    
    # Generate plots
    output_dir = Path(args.output_dir)
    plot_sensitivity_results(results, output_dir)


if __name__ == '__main__':
    main()
