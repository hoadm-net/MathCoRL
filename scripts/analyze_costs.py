#!/usr/bin/env python3
"""
Cost Analysis from API Usage Logs

Analyzes API usage logs to compute total costs and compare methods.

Usage:
    python scripts/analyze_costs.py --logs logs/api_usage.jsonl
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List

def analyze_api_logs(log_file: str) -> Dict:
    """Analyze API usage from logs."""
    
    costs_by_method = defaultdict(lambda: {'calls': 0, 'tokens': 0, 'cost': 0.0, 'time': 0.0})
    
    with open(log_file) as f:
        for line in f:
            try:
                entry = json.loads(line)
                method = entry.get('method', 'unknown')
                
                costs_by_method[method]['calls'] += 1
                costs_by_method[method]['tokens'] += entry.get('total_tokens', 0)
                costs_by_method[method]['cost'] += entry.get('cost', 0.0)
                costs_by_method[method]['time'] += entry.get('duration', 0.0)
            except:
                continue
    
    return dict(costs_by_method)

def main():
    parser = argparse.ArgumentParser(description='Analyze API costs from logs')
    parser.add_argument('--logs', default='logs/api_usage.jsonl', help='API usage log file')
    parser.add_argument('--output', default='results/cost/cost_analysis.json', help='Output file')
    
    args = parser.parse_args()
    
    # Analyze logs
    results = analyze_api_logs(args.logs)
    
    # Calculate totals
    total_calls = sum(m['calls'] for m in results.values())
    total_cost = sum(m['cost'] for m in results.values())
    total_tokens = sum(m['tokens'] for m in results.values())
    total_time = sum(m['time'] for m in results.values())
    
    # Print summary
    print('='*70)
    print('💰 API COST ANALYSIS')
    print('='*70)
    print(f"Total API Calls: {total_calls:,}")
    print(f"Total Tokens: {total_tokens:,}")
    print(f"Total Cost: ${total_cost:.4f}")
    print(f"Total Time: {total_time:.2f}s")
    print()
    print('By Method:')
    print('-'*70)
    
    for method, stats in sorted(results.items(), key=lambda x: x[1]['cost'], reverse=True):
        print(f"{method:20s}: ${stats['cost']:8.4f} ({stats['calls']:4d} calls, "
              f"{stats['tokens']:7,} tokens, {stats['time']:6.1f}s)")
    print('='*70)
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    summary = {
        'total_calls': total_calls,
        'total_tokens': total_tokens,
        'total_cost': total_cost,
        'total_time': total_time,
        'by_method': results
    }
    
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✅ Results saved to {output_path}")

if __name__ == '__main__':
    main()
