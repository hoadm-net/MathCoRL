#!/usr/bin/env python3
"""
Prototype Library Coverage Analysis for MathCoRL

Analyzes function prototype usage across generated code to:
1. Quantify which functions are actually used
2. Measure coverage by dataset and problem type
3. Identify essential vs rarely-used functions
4. Justify prototype library design with empirical data

Usage:
    python scripts/analyze_prototypes.py --dataset GSM8K
    python scripts/analyze_prototypes.py --all-datasets --visualize
    python scripts/analyze_prototypes.py --candidates candidates/GSM8K.json
"""

import argparse
import ast
import json
import re
import sys
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Set, Tuple, Any
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FunctionExtractor(ast.NodeVisitor):
    """AST visitor to extract function calls from Python code."""
    
    def __init__(self):
        self.functions = []
        
    def visit_Call(self, node):
        """Visit function call nodes."""
        if isinstance(node.func, ast.Name):
            # Direct function call: func()
            self.functions.append(node.func.id)
        elif isinstance(node.func, ast.Attribute):
            # Method call: obj.method()
            self.functions.append(node.func.attr)
        self.generic_visit(node)


def extract_functions_from_code(code: str) -> List[str]:
    """Extract all function calls from Python code using AST parsing.
    
    Args:
        code: Python code string
        
    Returns:
        List of function names called in the code
    """
    try:
        tree = ast.parse(code)
        extractor = FunctionExtractor()
        extractor.visit(tree)
        return extractor.functions
    except SyntaxError as e:
        logger.warning(f"Syntax error parsing code: {e}")
        return []
    except Exception as e:
        logger.warning(f"Error extracting functions: {e}")
        return []


def load_prototype_definitions(prototype_file: str) -> Set[str]:
    """Load function names defined in prototype file.
    
    Args:
        prototype_file: Path to function prototypes template
        
    Returns:
        Set of function names defined
    """
    functions = set()
    
    with open(prototype_file, 'r') as f:
        content = f.read()
    
    # Extract function definitions using regex
    pattern = r'^def\s+(\w+)\s*\('
    matches = re.findall(pattern, content, re.MULTILINE)
    functions.update(matches)
    
    logger.info(f"Loaded {len(functions)} function definitions from {prototype_file}")
    return functions


def analyze_candidates(candidates_file: str, available_functions: Set[str]) -> Dict[str, Any]:
    """Analyze function usage in candidate solutions.
    
    Args:
        candidates_file: Path to candidates JSON file
        available_functions: Set of available function names
        
    Returns:
        Dictionary with analysis results
    """
    with open(candidates_file, 'r') as f:
        candidates = json.load(f)
    
    # Track usage
    function_usage = Counter()
    problem_function_usage = defaultdict(list)  # problem_id -> [functions]
    total_problems = len(candidates)
    problems_with_code = 0
    
    for idx, candidate in enumerate(candidates):
        code = candidate.get('code', '')
        if not code:
            continue
            
        problems_with_code += 1
        
        # Extract functions from code
        used_functions = extract_functions_from_code(code)
        
        # Filter to only count functions from our prototype library
        prototype_functions = [f for f in used_functions if f in available_functions]
        
        # Update counters
        function_usage.update(prototype_functions)
        problem_function_usage[idx] = prototype_functions
    
    # Calculate coverage
    used_functions = set(function_usage.keys())
    unused_functions = available_functions - used_functions
    coverage_rate = len(used_functions) / len(available_functions) if available_functions else 0
    
    # Categorize by usage frequency
    high_freq = {f: count for f, count in function_usage.items() if count >= problems_with_code * 0.2}
    medium_freq = {f: count for f, count in function_usage.items() if problems_with_code * 0.05 <= count < problems_with_code * 0.2}
    low_freq = {f: count for f, count in function_usage.items() if count < problems_with_code * 0.05}
    
    results = {
        'total_problems': total_problems,
        'problems_with_code': problems_with_code,
        'available_functions': len(available_functions),
        'used_functions': len(used_functions),
        'unused_functions': len(unused_functions),
        'coverage_rate': coverage_rate,
        'function_usage': dict(function_usage),
        'high_frequency': high_freq,  # Used in >20% of problems
        'medium_frequency': medium_freq,  # Used in 5-20% of problems
        'low_frequency': low_freq,  # Used in <5% of problems
        'unused_list': sorted(unused_functions),
        'most_common': function_usage.most_common(10),
        'problem_function_mapping': dict(problem_function_usage)
    }
    
    return results


def categorize_functions_by_type(functions: Set[str]) -> Dict[str, List[str]]:
    """Categorize functions by operation type.
    
    Args:
        functions: Set of function names
        
    Returns:
        Dictionary mapping category to list of functions
    """
    categories = {
        'arithmetic': ['add', 'sub', 'mul', 'div', 'mod', 'pow', 'abs', 'round', 'ceil', 'floor'],
        'aggregation': ['sum', 'avg', 'min', 'max', 'count', 'median'],
        'percentage': ['percentage', 'percentage_of', 'percentage_change', 'percentage_increase', 'percentage_decrease'],
        'financial': ['compound_interest', 'simple_interest', 'growth_rate', 'discount', 'profit', 'loss', 'roi'],
        'ratio': ['ratio', 'proportion', 'rate'],
        'table': ['extract_column', 'filter_rows', 'group_by', 'aggregate_table', 'join_tables'],
        'comparison': ['greater_than', 'less_than', 'equal', 'between'],
        'conversion': ['to_float', 'to_int', 'to_percentage'],
        'other': []
    }
    
    result = defaultdict(list)
    categorized = set()
    
    # Categorize known functions
    for category, func_list in categories.items():
        for func in functions:
            if func in func_list:
                result[category].append(func)
                categorized.add(func)
    
    # Add uncategorized to 'other'
    for func in functions:
        if func not in categorized:
            result['other'].append(func)
    
    return dict(result)


def print_analysis_report(results: Dict[str, Any], dataset_name: str):
    """Print formatted analysis report.
    
    Args:
        results: Analysis results dictionary
        dataset_name: Name of dataset analyzed
    """
    print("\n" + "="*80)
    print(f"FUNCTION PROTOTYPE COVERAGE ANALYSIS - {dataset_name}")
    print("="*80)
    
    print(f"\n📊 Overview:")
    print(f"  Total problems: {results['total_problems']}")
    print(f"  Problems with code: {results['problems_with_code']}")
    print(f"  Available functions: {results['available_functions']}")
    print(f"  Used functions: {results['used_functions']}")
    print(f"  Unused functions: {results['unused_functions']}")
    print(f"  Coverage rate: {results['coverage_rate']:.1%}")
    
    print(f"\n🔥 High-frequency functions (>20% usage):")
    for func, count in sorted(results['high_frequency'].items(), key=lambda x: x[1], reverse=True):
        usage_rate = count / results['problems_with_code']
        print(f"  {func:<20} {count:>4} times ({usage_rate:>6.1%})")
    
    print(f"\n📈 Medium-frequency functions (5-20% usage):")
    for func, count in sorted(results['medium_frequency'].items(), key=lambda x: x[1], reverse=True):
        usage_rate = count / results['problems_with_code']
        print(f"  {func:<20} {count:>4} times ({usage_rate:>6.1%})")
    
    print(f"\n📉 Low-frequency functions (<5% usage):")
    low_freq_list = sorted(results['low_frequency'].items(), key=lambda x: x[1], reverse=True)[:10]
    for func, count in low_freq_list:
        usage_rate = count / results['problems_with_code']
        print(f"  {func:<20} {count:>4} times ({usage_rate:>6.1%})")
    if len(results['low_frequency']) > 10:
        print(f"  ... and {len(results['low_frequency']) - 10} more")
    
    print(f"\n❌ Unused functions ({len(results['unused_list'])}):")
    if results['unused_list']:
        print(f"  {', '.join(results['unused_list'][:15])}")
        if len(results['unused_list']) > 15:
            print(f"  ... and {len(results['unused_list']) - 15} more")
    else:
        print("  None - 100% coverage!")
    
    print(f"\n🏆 Top 10 most used functions:")
    for i, (func, count) in enumerate(results['most_common'], 1):
        usage_rate = count / results['problems_with_code']
        print(f"  {i:>2}. {func:<18} {count:>4} times ({usage_rate:>6.1%})")
    
    print("\n" + "="*80 + "\n")


def save_results(results: Dict[str, Any], output_file: str):
    """Save analysis results to JSON file.
    
    Args:
        results: Analysis results
        output_file: Output file path
    """
    # Remove problem_function_mapping for cleaner output (can be large)
    output_results = {k: v for k, v in results.items() if k != 'problem_function_mapping'}
    
    with open(output_file, 'w') as f:
        json.dump(output_results, f, indent=2)
    
    logger.info(f"Results saved to {output_file}")


def main():
    """Main function with CLI interface."""
    parser = argparse.ArgumentParser(
        description='Analyze function prototype coverage in generated code',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Analyze single dataset
    python scripts/analyze_prototypes.py --dataset GSM8K
    
    # Analyze all datasets
    python scripts/analyze_prototypes.py --all-datasets
    
    # Use specific prototype file and candidates
    python scripts/analyze_prototypes.py --prototypes templates/function_prototypes_fin.txt --candidates candidates/FinQA.json
    
    # Generate visualizations
    python scripts/analyze_prototypes.py --dataset TabMWP --visualize
        """
    )
    
    parser.add_argument(
        '--dataset', '-d',
        type=str,
        choices=['GSM8K', 'SVAMP', 'TabMWP', 'TAT-QA', 'FinQA'],
        help='Dataset to analyze'
    )
    
    parser.add_argument(
        '--all-datasets',
        action='store_true',
        help='Analyze all available datasets'
    )
    
    parser.add_argument(
        '--prototypes',
        type=str,
        default='templates/function_prototypes.txt',
        help='Path to function prototypes file (default: templates/function_prototypes.txt)'
    )
    
    parser.add_argument(
        '--candidates',
        type=str,
        help='Path to candidates JSON file (default: candidates/{dataset}.json)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        help='Output JSON file for results (default: results/prototype_analysis/{dataset}_analysis.json)'
    )
    
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Generate visualization plots'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.all_datasets and not args.dataset:
        parser.error("Must specify either --dataset or --all-datasets")
    
    # Load available functions
    prototypes_path = Path(args.prototypes)
    if not prototypes_path.exists():
        logger.error(f"Prototypes file not found: {prototypes_path}")
        return 1
    
    available_functions = load_prototype_definitions(str(prototypes_path))
    
    # Analyze datasets
    datasets = ['GSM8K', 'SVAMP', 'TabMWP', 'TAT-QA', 'FinQA'] if args.all_datasets else [args.dataset]
    
    all_results = {}
    
    for dataset in datasets:
        logger.info(f"\nAnalyzing {dataset}...")
        
        # Determine candidates file
        candidates_file = args.candidates or f"candidates/{dataset}.json"
        candidates_path = Path(candidates_file)
        
        if not candidates_path.exists():
            logger.warning(f"Candidates file not found: {candidates_path}, skipping {dataset}")
            continue
        
        # Analyze
        results = analyze_candidates(str(candidates_path), available_functions)
        all_results[dataset] = results
        
        # Print report
        print_analysis_report(results, dataset)
        
        # Save results
        output_dir = Path('results/prototype_analysis')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = args.output or output_dir / f"{dataset}_analysis.json"
        save_results(results, str(output_file))
    
    # Generate visualizations if requested
    if args.visualize:
        try:
            from scripts.plot_prototypes import plot_prototype_analysis
            plot_dir = Path('results/prototype_analysis/plots')
            plot_dir.mkdir(parents=True, exist_ok=True)
            plot_prototype_analysis(all_results, output_dir=plot_dir)
            logger.info(f"Plots saved to {plot_dir}")
        except ImportError:
            logger.warning("Matplotlib not available. Skipping visualization.")
            logger.warning("Install with: pip install matplotlib seaborn")
    
    logger.info("\n✅ Prototype analysis complete!")


if __name__ == '__main__':
    main()
