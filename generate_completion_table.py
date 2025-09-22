#!/usr/bin/env python3
"""
Generate completion table from FPP ablation study results.

This script aggregates results from multiple ablation runs and formats them
into the completion table format requested by the user.
"""

import json
import glob
from pathlib import Path
from typing import Dict, List
import argparse


def load_ablation_results(results_dir: str = "results/fpp_ablation") -> Dict[str, Dict]:
    """Load all FPP ablation results from the results directory."""
    results_path = Path(results_dir)
    if not results_path.exists():
        print(f"Results directory not found: {results_dir}")
        return {}
    
    # Find all result files
    result_files = list(results_path.glob("fpp_ablation_*_*.json"))
    
    if not result_files:
        print(f"No result files found in {results_dir}")
        return {}
    
    dataset_results = {}
    
    for file_path in result_files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            dataset = data.get('dataset', 'unknown')
            timestamp = data.get('timestamp', '')
            
            # Keep most recent results for each dataset
            if dataset not in dataset_results or timestamp > dataset_results[dataset].get('timestamp', ''):
                dataset_results[dataset] = data
                print(f"Loaded results for {dataset}: {file_path.name}")
        
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    return dataset_results


def generate_completion_table(dataset_results: Dict[str, Dict]) -> str:
    """Generate the completion table in the requested format."""
    
    # Define variant mappings
    variant_labels = {
        'baseline': 'MathCoRL',
        'policy_retrieval': 'Policy Network',
        'policy_random': 'Policy Random',
        'no_policy': 'No Policy',
        'reduced_library': 'Reduced Library',
        'no_constraints': 'No Constraints'
    }
    
    # Generate table
    table_lines = []
    table_lines.append("FPP Ablation Study - Completion Table")
    table_lines.append("=" * 50)
    table_lines.append("")
    
    # Header
    datasets = list(dataset_results.keys())
    if datasets:
        header = f"{'Variant':<20}"
        for dataset in datasets:
            header += f"{dataset:>12}"
        table_lines.append(header)
        table_lines.append("-" * len(header))
    
    # Data rows
    for variant_key, variant_label in variant_labels.items():
        row = f"{variant_label:<20}"
        
        for dataset in datasets:
            if dataset in dataset_results:
                metrics = dataset_results[dataset].get('metrics', {})
                if variant_key in metrics:
                    accuracy = metrics[variant_key]['accuracy']
                    row += f"{accuracy:>12.2f}"
                else:
                    row += f"{'N/A':>12}"
            else:
                row += f"{'N/A':>12}"
        
        table_lines.append(row)
    
    table_lines.append("")
    
    # Add metadata
    table_lines.append("Metadata:")
    for dataset, data in dataset_results.items():
        samples = data.get('n_samples', 'unknown')
        model = data.get('model', 'unknown')
        timestamp = data.get('timestamp', 'unknown')[:19]  # Remove milliseconds
        table_lines.append(f"  {dataset}: {samples} samples, {model}, {timestamp}")
    
    return "\n".join(table_lines)


def generate_detailed_summary(dataset_results: Dict[str, Dict]) -> str:
    """Generate detailed summary with additional metrics."""
    
    summary_lines = []
    summary_lines.append("Detailed FPP Ablation Summary")
    summary_lines.append("=" * 40)
    summary_lines.append("")
    
    for dataset, data in dataset_results.items():
        summary_lines.append(f"{dataset} Results:")
        summary_lines.append("-" * (len(dataset) + 9))
        
        metrics = data.get('metrics', {})
        
        # Sort variants by accuracy
        sorted_variants = sorted(
            metrics.items(),
            key=lambda x: x[1]['accuracy'],
            reverse=True
        )
        
        for variant, metric in sorted_variants:
            acc = metric['accuracy']
            correct = metric['correct_count']
            total = metric['total_count']
            success_rate = metric['success_rate']
            
            summary_lines.append(
                f"  {variant:<18} Accuracy: {acc:.3f} ({correct}/{total})  "
                f"Success: {success_rate:.3f}"
            )
        
        summary_lines.append("")
    
    return "\n".join(summary_lines)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Generate completion table from FPP ablation results"
    )
    
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results/fpp_ablation",
        help="Directory containing ablation results"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (default: print to console)"
    )
    
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Include detailed summary"
    )
    
    args = parser.parse_args()
    
    # Load results
    print(f"Loading results from: {args.results_dir}")
    dataset_results = load_ablation_results(args.results_dir)
    
    if not dataset_results:
        print("No results found. Please run the ablation study first.")
        return 1
    
    # Generate table
    table = generate_completion_table(dataset_results)
    
    # Generate detailed summary if requested
    if args.detailed:
        detailed = generate_detailed_summary(dataset_results)
        output_text = table + "\n\n" + detailed
    else:
        output_text = table
    
    # Output results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            f.write(output_text)
        
        print(f"Results saved to: {output_path}")
    else:
        print("\n" + output_text)
    
    return 0


if __name__ == "__main__":
    exit(main())