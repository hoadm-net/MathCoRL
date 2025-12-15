#!/usr/bin/env python3
"""
Token Cost Analysis for MathCoRL

Analyze token usage and API costs from tracking logs.
Calculate cost per problem, cost per correct answer, and compare methods.

Usage:
    python scripts/cost_analysis.py --logs logs/api_usage.jsonl
    python scripts/cost_analysis.py --hours 24 --method FPP
    python scripts/cost_analysis.py --dataset GSM8K --export results/cost/
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import defaultdict
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from mint.tracking import MODEL_PRICING

@dataclass
class CostMetrics:
    """Cost metrics for a single method/dataset"""
    method: str
    dataset: Optional[str]
    num_requests: int
    total_cost: float
    input_tokens: int
    output_tokens: int
    total_tokens: int
    avg_cost_per_request: float
    avg_input_tokens: float
    avg_output_tokens: float
    success_rate: float
    cost_per_success: float
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ModelCostBreakdown:
    """Cost breakdown by model"""
    model: str
    requests: int
    total_cost: float
    input_cost: float
    output_cost: float
    total_tokens: int
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class CostAnalyzer:
    """Analyze API costs from tracking logs"""
    
    def __init__(self, log_file: Path):
        """
        Initialize cost analyzer
        
        Args:
            log_file: Path to API usage log (JSONL format)
        """
        self.log_file = Path(log_file)
        if not self.log_file.exists():
            raise FileNotFoundError(f"Log file not found: {log_file}")
        
        self.logs = self._load_logs()
        print(f"Loaded {len(self.logs)} log entries from {log_file}")
    
    def _load_logs(self, hours: Optional[int] = None) -> List[Dict[str, Any]]:
        """Load logs from file, optionally filtered by time"""
        logs = []
        cutoff_time = None
        
        if hours:
            cutoff_time = datetime.now().timestamp() - (hours * 3600)
        
        with open(self.log_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        log = json.loads(line.strip())
                        
                        # Filter by time if specified
                        if cutoff_time:
                            log_time = datetime.fromisoformat(log['timestamp']).timestamp()
                            if log_time < cutoff_time:
                                continue
                        
                        logs.append(log)
                    except json.JSONDecodeError as e:
                        print(f"Warning: Invalid JSON line: {e}")
                        continue
        
        return logs
    
    def filter_logs(self, method: Optional[str] = None, 
                    dataset: Optional[str] = None,
                    hours: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Filter logs by method, dataset, or time
        
        Args:
            method: Method name (FPP, CoT, etc.)
            dataset: Dataset name (extracted from question)
            hours: Last N hours
            
        Returns:
            Filtered logs
        """
        filtered = self.logs
        
        if method:
            filtered = [log for log in filtered if log.get('method', '').upper() == method.upper()]
        
        if dataset:
            # Try to infer dataset from question (heuristic)
            filtered = [log for log in filtered if dataset.upper() in log.get('question', '').upper()]
        
        if hours:
            cutoff = datetime.now().timestamp() - (hours * 3600)
            filtered = [log for log in filtered 
                       if datetime.fromisoformat(log['timestamp']).timestamp() >= cutoff]
        
        return filtered
    
    def analyze_by_method(self, logs: Optional[List[Dict]] = None) -> Dict[str, CostMetrics]:
        """
        Analyze costs grouped by method
        
        Args:
            logs: Log entries to analyze (default: all logs)
            
        Returns:
            Dictionary mapping method name to CostMetrics
        """
        logs = logs or self.logs
        
        # Group by method
        method_data = defaultdict(lambda: {
            'costs': [],
            'input_tokens': [],
            'output_tokens': [],
            'successes': []
        })
        
        for log in logs:
            method = log.get('method', 'Unknown')
            method_data[method]['costs'].append(log.get('total_cost', 0))
            method_data[method]['input_tokens'].append(log.get('input_tokens', 0))
            method_data[method]['output_tokens'].append(log.get('output_tokens', 0))
            method_data[method]['successes'].append(log.get('success', True))
        
        # Compute metrics
        results = {}
        for method, data in method_data.items():
            num_requests = len(data['costs'])
            total_cost = sum(data['costs'])
            input_tokens = sum(data['input_tokens'])
            output_tokens = sum(data['output_tokens'])
            successes = sum(data['successes'])
            success_rate = successes / num_requests if num_requests > 0 else 0
            
            results[method] = CostMetrics(
                method=method,
                dataset=None,
                num_requests=num_requests,
                total_cost=total_cost,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
                avg_cost_per_request=total_cost / num_requests if num_requests > 0 else 0,
                avg_input_tokens=input_tokens / num_requests if num_requests > 0 else 0,
                avg_output_tokens=output_tokens / num_requests if num_requests > 0 else 0,
                success_rate=success_rate,
                cost_per_success=total_cost / successes if successes > 0 else float('inf')
            )
        
        return results
    
    def analyze_by_model(self, logs: Optional[List[Dict]] = None) -> Dict[str, ModelCostBreakdown]:
        """
        Analyze costs grouped by model
        
        Args:
            logs: Log entries to analyze
            
        Returns:
            Dictionary mapping model name to ModelCostBreakdown
        """
        logs = logs or self.logs
        
        model_data = defaultdict(lambda: {
            'requests': 0,
            'total_cost': 0,
            'input_cost': 0,
            'output_cost': 0,
            'total_tokens': 0
        })
        
        for log in logs:
            model = log.get('model', 'unknown')
            model_data[model]['requests'] += 1
            model_data[model]['total_cost'] += log.get('total_cost', 0)
            model_data[model]['input_cost'] += log.get('input_cost', 0)
            model_data[model]['output_cost'] += log.get('output_cost', 0)
            model_data[model]['total_tokens'] += log.get('total_tokens', 0)
        
        results = {}
        for model, data in model_data.items():
            results[model] = ModelCostBreakdown(
                model=model,
                requests=data['requests'],
                total_cost=data['total_cost'],
                input_cost=data['input_cost'],
                output_cost=data['output_cost'],
                total_tokens=data['total_tokens']
            )
        
        return results
    
    def compute_cost_efficiency(self, logs: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """
        Compute cost efficiency metrics
        
        Returns:
            Dictionary with efficiency metrics
        """
        logs = logs or self.logs
        
        if not logs:
            return {"error": "No logs to analyze"}
        
        # Overall statistics
        total_cost = sum(log.get('total_cost', 0) for log in logs)
        total_tokens = sum(log.get('total_tokens', 0) for log in logs)
        total_requests = len(logs)
        successful = sum(1 for log in logs if log.get('success', True))
        
        # Token efficiency
        avg_input = np.mean([log.get('input_tokens', 0) for log in logs])
        avg_output = np.mean([log.get('output_tokens', 0) for log in logs])
        
        # Cost distribution
        costs = [log.get('total_cost', 0) for log in logs]
        cost_percentiles = {
            'p50': float(np.percentile(costs, 50)),
            'p90': float(np.percentile(costs, 90)),
            'p95': float(np.percentile(costs, 95)),
            'p99': float(np.percentile(costs, 99))
        }
        
        return {
            'total_cost': total_cost,
            'total_tokens': total_tokens,
            'total_requests': total_requests,
            'successful_requests': successful,
            'success_rate': successful / total_requests if total_requests > 0 else 0,
            'avg_cost_per_request': total_cost / total_requests if total_requests > 0 else 0,
            'cost_per_success': total_cost / successful if successful > 0 else float('inf'),
            'avg_input_tokens': avg_input,
            'avg_output_tokens': avg_output,
            'cost_per_1k_tokens': (total_cost / total_tokens * 1000) if total_tokens > 0 else 0,
            'cost_percentiles': cost_percentiles
        }
    
    def compare_methods(self, methods: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Compare cost efficiency across methods
        
        Args:
            methods: List of methods to compare (default: all)
            
        Returns:
            Comparison dictionary
        """
        method_metrics = self.analyze_by_method()
        
        if methods:
            method_metrics = {k: v for k, v in method_metrics.items() if k in methods}
        
        # Sort by cost per success
        sorted_methods = sorted(
            method_metrics.items(),
            key=lambda x: x[1].cost_per_success
        )
        
        comparison = {
            'methods': {name: metrics.to_dict() for name, metrics in method_metrics.items()},
            'ranking_by_efficiency': [
                {
                    'method': name,
                    'cost_per_success': metrics.cost_per_success,
                    'success_rate': metrics.success_rate,
                    'avg_cost': metrics.avg_cost_per_request
                }
                for name, metrics in sorted_methods
            ]
        }
        
        return comparison
    
    def export_results(self, results: Dict[str, Any], output_dir: Path, 
                      name: str = "cost_analysis"):
        """
        Export analysis results to JSON
        
        Args:
            results: Results dictionary
            output_dir: Output directory
            name: Base filename
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = output_dir / f"{name}_{timestamp}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results exported to: {filename}")
        return filename
    
    def print_summary(self, method: Optional[str] = None):
        """Print cost summary to console"""
        logs = self.filter_logs(method=method) if method else self.logs
        
        if not logs:
            print("No logs found")
            return
        
        print("\n" + "="*70)
        print(f"COST ANALYSIS SUMMARY")
        if method:
            print(f"Method: {method}")
        print("="*70)
        
        # Overall metrics
        efficiency = self.compute_cost_efficiency(logs)
        print(f"\nOverall Statistics:")
        print(f"  Total Requests: {efficiency['total_requests']}")
        print(f"  Successful: {efficiency['successful_requests']} ({efficiency['success_rate']*100:.1f}%)")
        print(f"  Total Cost: ${efficiency['total_cost']:.4f}")
        print(f"  Total Tokens: {efficiency['total_tokens']:,}")
        print(f"\nCost Efficiency:")
        print(f"  Avg Cost/Request: ${efficiency['avg_cost_per_request']:.6f}")
        print(f"  Cost/Success: ${efficiency['cost_per_success']:.6f}")
        print(f"  Cost/1K Tokens: ${efficiency['cost_per_1k_tokens']:.6f}")
        print(f"\nToken Usage:")
        print(f"  Avg Input Tokens: {efficiency['avg_input_tokens']:.1f}")
        print(f"  Avg Output Tokens: {efficiency['avg_output_tokens']:.1f}")
        print(f"\nCost Distribution:")
        for pct, value in efficiency['cost_percentiles'].items():
            print(f"  {pct.upper()}: ${value:.6f}")
        
        # By method
        if not method:
            print(f"\n{'='*70}")
            print("By Method:")
            print(f"{'='*70}")
            method_metrics = self.analyze_by_method(logs)
            
            for method_name, metrics in sorted(method_metrics.items()):
                print(f"\n{method_name}:")
                print(f"  Requests: {metrics.num_requests}")
                print(f"  Total Cost: ${metrics.total_cost:.4f}")
                print(f"  Avg Cost: ${metrics.avg_cost_per_request:.6f}")
                print(f"  Success Rate: {metrics.success_rate*100:.1f}%")
                print(f"  Cost/Success: ${metrics.cost_per_success:.6f}")
        
        # By model
        print(f"\n{'='*70}")
        print("By Model:")
        print(f"{'='*70}")
        model_metrics = self.analyze_by_model(logs)
        
        for model_name, metrics in sorted(model_metrics.items()):
            print(f"\n{model_name}:")
            print(f"  Requests: {metrics.requests}")
            print(f"  Total Cost: ${metrics.total_cost:.4f}")
            print(f"  Input Cost: ${metrics.input_cost:.4f} ({metrics.input_cost/metrics.total_cost*100:.1f}%)")
            print(f"  Output Cost: ${metrics.output_cost:.4f} ({metrics.output_cost/metrics.total_cost*100:.1f}%)")
            print(f"  Total Tokens: {metrics.total_tokens:,}")
        
        print(f"\n{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze API token costs from tracking logs',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--logs', type=str, default='logs/api_usage.jsonl',
                       help='Path to API usage log file (default: logs/api_usage.jsonl)')
    
    parser.add_argument('--method', '-m', type=str,
                       help='Filter by method (FPP, CoT, etc.)')
    
    parser.add_argument('--dataset', '-d', type=str,
                       help='Filter by dataset (GSM8K, SVAMP, etc.)')
    
    parser.add_argument('--hours', type=int,
                       help='Analyze last N hours only')
    
    parser.add_argument('--export', '-o', type=str,
                       help='Export results to directory')
    
    parser.add_argument('--compare', action='store_true',
                       help='Compare methods by efficiency')
    
    args = parser.parse_args()
    
    try:
        # Initialize analyzer
        analyzer = CostAnalyzer(args.logs)
        
        # Print summary
        analyzer.print_summary(method=args.method)
        
        # Export if requested
        if args.export:
            logs = analyzer.filter_logs(method=args.method, dataset=args.dataset, hours=args.hours)
            
            results = {
                'timestamp': datetime.now().isoformat(),
                'log_file': str(analyzer.log_file),
                'filters': {
                    'method': args.method,
                    'dataset': args.dataset,
                    'hours': args.hours
                },
                'efficiency': analyzer.compute_cost_efficiency(logs),
                'by_method': {k: v.to_dict() for k, v in analyzer.analyze_by_method(logs).items()},
                'by_model': {k: v.to_dict() for k, v in analyzer.analyze_by_model(logs).items()}
            }
            
            if args.compare:
                results['comparison'] = analyzer.compare_methods()
            
            analyzer.export_results(results, args.export)
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
