#!/usr/bin/env python3
"""
Failure Case Analysis for MathCoRL

Categorizes and analyzes failure modes across different methods.
Error types: parsing, execution, logic, numerical precision.

Usage:
    python scripts/failure_analysis.py --results results/latency/*.json
    python scripts/failure_analysis.py --dataset GSM8K --method FPP
"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional
from collections import Counter, defaultdict
import argparse


@dataclass
class FailureCase:
    """Single failure case"""
    question: str
    expected_answer: Any
    predicted_answer: Any
    method: str
    error_type: str
    error_message: Optional[str] = None
    code: Optional[str] = None
    traceback: Optional[str] = None
    dataset: str = ""
    problem_id: Optional[str] = None


@dataclass
class ErrorStatistics:
    """Aggregate error statistics"""
    total_problems: int = 0
    total_failures: int = 0
    success_rate: float = 0.0
    
    # By error type
    parsing_errors: int = 0
    execution_errors: int = 0
    logic_errors: int = 0
    numerical_errors: int = 0
    timeout_errors: int = 0
    api_errors: int = 0
    unknown_errors: int = 0
    
    # Examples
    failure_cases: List[FailureCase] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_problems": self.total_problems,
            "total_failures": self.total_failures,
            "success_rate": self.success_rate,
            "failure_rate": 1.0 - self.success_rate,
            "error_breakdown": {
                "parsing": self.parsing_errors,
                "execution": self.execution_errors,
                "logic": self.logic_errors,
                "numerical": self.numerical_errors,
                "timeout": self.timeout_errors,
                "api": self.api_errors,
                "unknown": self.unknown_errors
            },
            "num_cases": len(self.failure_cases)
        }


class FailureAnalyzer:
    """Analyze failure modes across methods and datasets"""
    
    def __init__(self):
        self.results_by_method: Dict[str, ErrorStatistics] = defaultdict(ErrorStatistics)
        self.results_by_dataset: Dict[str, ErrorStatistics] = defaultdict(ErrorStatistics)
        self.overall_stats = ErrorStatistics()
        
    def classify_error_type(self, result: Dict[str, Any], expected: Any) -> str:
        """
        Classify error into categories:
        - parsing: Failed to extract answer from LLM output
        - execution: Python code execution failed
        - logic: Wrong algorithm/approach
        - numerical: Rounding/precision issues
        - timeout: Execution timeout
        - api: API call failure
        """
        error_msg = result.get('error', '') or ''
        code = result.get('code', '')
        predicted = result.get('result')
        
        # Check for explicit error messages
        if 'SyntaxError' in error_msg or 'IndentationError' in error_msg:
            return 'parsing'
        
        if any(x in error_msg for x in ['NameError', 'AttributeError', 'ImportError', 
                                          'TypeError', 'KeyError', 'IndexError']):
            return 'execution'
        
        if 'timeout' in error_msg.lower() or 'timed out' in error_msg.lower():
            return 'timeout'
        
        if 'API' in error_msg or 'rate limit' in error_msg.lower():
            return 'api'
        
        # No explicit error - wrong answer
        if predicted is not None and expected is not None:
            # Check for numerical precision issues
            try:
                pred_float = float(str(predicted).replace(',', '').replace('$', ''))
                exp_float = float(str(expected).replace(',', '').replace('$', ''))
                
                # Close but not exact (within 0.1%)
                if abs(pred_float - exp_float) / max(abs(exp_float), 1) < 0.001:
                    return 'numerical'
                    
            except (ValueError, TypeError):
                pass
            
            # Wrong answer - logic error
            return 'logic'
        
        # No result produced
        if predicted is None:
            if 'def solve' in code or 'return' in code:
                return 'execution'  # Code didn't run
            else:
                return 'parsing'  # Failed to generate valid code
        
        return 'unknown'
    
    def analyze_result(self, result: Dict[str, Any], method: str, dataset: str = "") -> Optional[FailureCase]:
        """Analyze a single result and return failure case if applicable"""
        success = result.get('success', True)
        
        if success:
            return None
        
        # Extract information
        question = result.get('question', '')
        expected = result.get('expected_answer') or result.get('answer')
        predicted = result.get('result') or result.get('predicted_answer')
        error_msg = result.get('error')
        code = result.get('code', '')
        
        # Classify error
        error_type = self.classify_error_type(result, expected)
        
        return FailureCase(
            question=question,
            expected_answer=expected,
            predicted_answer=predicted,
            method=method,
            error_type=error_type,
            error_message=error_msg,
            code=code,
            dataset=dataset,
            problem_id=result.get('id') or result.get('problem_id')
        )
    
    def update_statistics(self, stats: ErrorStatistics, failure: FailureCase):
        """Update error statistics with a failure case"""
        stats.total_failures += 1
        
        if failure.error_type == 'parsing':
            stats.parsing_errors += 1
        elif failure.error_type == 'execution':
            stats.execution_errors += 1
        elif failure.error_type == 'logic':
            stats.logic_errors += 1
        elif failure.error_type == 'numerical':
            stats.numerical_errors += 1
        elif failure.error_type == 'timeout':
            stats.timeout_errors += 1
        elif failure.error_type == 'api':
            stats.api_errors += 1
        else:
            stats.unknown_errors += 1
        
        stats.failure_cases.append(failure)
    
    def load_latency_results(self, filepath: str):
        """Load and analyze latency test results"""
        with open(filepath) as f:
            data = json.load(f)
        
        dataset = data.get('dataset', 'Unknown')
        
        for method_name, method_data in data.get('methods', {}).items():
            individual = method_data.get('individual', [])
            
            for result in individual:
                # Track total
                self.results_by_method[method_name].total_problems += 1
                self.results_by_dataset[dataset].total_problems += 1
                self.overall_stats.total_problems += 1
                
                # Analyze failures
                failure = self.analyze_result(result, method_name, dataset)
                if failure:
                    self.update_statistics(self.results_by_method[method_name], failure)
                    self.update_statistics(self.results_by_dataset[dataset], failure)
                    self.update_statistics(self.overall_stats, failure)
    
    def load_api_logs(self, filepath: str, max_entries: int = None):
        """Load and analyze API usage logs"""
        count = 0
        with open(filepath) as f:
            for line in f:
                if max_entries and count >= max_entries:
                    break
                    
                entry = json.loads(line)
                method = entry.get('method', 'Unknown')
                
                # Track total
                self.results_by_method[method].total_problems += 1
                self.overall_stats.total_problems += 1
                count += 1
                
                # Check for failure
                if not entry.get('success', True):
                    failure = self.analyze_result(entry, method)
                    if failure:
                        self.update_statistics(self.results_by_method[method], failure)
                        self.update_statistics(self.overall_stats, failure)
    
    def compute_success_rates(self):
        """Compute success rates for all statistics"""
        for stats in self.results_by_method.values():
            if stats.total_problems > 0:
                stats.success_rate = 1.0 - (stats.total_failures / stats.total_problems)
        
        for stats in self.results_by_dataset.values():
            if stats.total_problems > 0:
                stats.success_rate = 1.0 - (stats.total_failures / stats.total_problems)
        
        if self.overall_stats.total_problems > 0:
            self.overall_stats.success_rate = 1.0 - (
                self.overall_stats.total_failures / self.overall_stats.total_problems
            )
    
    def get_top_failure_patterns(self, n: int = 10) -> List[tuple]:
        """Get most common failure patterns"""
        error_messages = []
        for failure in self.overall_stats.failure_cases:
            if failure.error_message:
                # Extract first line of error
                first_line = failure.error_message.split('\n')[0]
                error_messages.append(first_line)
        
        counter = Counter(error_messages)
        return counter.most_common(n)
    
    def get_failures_by_type(self, error_type: str) -> List[FailureCase]:
        """Get all failures of a specific type"""
        return [f for f in self.overall_stats.failure_cases if f.error_type == error_type]
    
    def print_summary(self):
        """Print summary of failure analysis"""
        print("\n" + "="*70)
        print("FAILURE ANALYSIS SUMMARY")
        print("="*70)
        
        print(f"\nOverall Statistics:")
        print(f"  Total Problems: {self.overall_stats.total_problems}")
        print(f"  Total Failures: {self.overall_stats.total_failures}")
        print(f"  Success Rate: {self.overall_stats.success_rate*100:.1f}%")
        print(f"  Failure Rate: {(1-self.overall_stats.success_rate)*100:.1f}%")
        
        print(f"\nError Breakdown:")
        print(f"  Parsing Errors: {self.overall_stats.parsing_errors} "
              f"({100*self.overall_stats.parsing_errors/max(self.overall_stats.total_failures,1):.1f}%)")
        print(f"  Execution Errors: {self.overall_stats.execution_errors} "
              f"({100*self.overall_stats.execution_errors/max(self.overall_stats.total_failures,1):.1f}%)")
        print(f"  Logic Errors: {self.overall_stats.logic_errors} "
              f"({100*self.overall_stats.logic_errors/max(self.overall_stats.total_failures,1):.1f}%)")
        print(f"  Numerical Errors: {self.overall_stats.numerical_errors} "
              f"({100*self.overall_stats.numerical_errors/max(self.overall_stats.total_failures,1):.1f}%)")
        print(f"  Timeout Errors: {self.overall_stats.timeout_errors} "
              f"({100*self.overall_stats.timeout_errors/max(self.overall_stats.total_failures,1):.1f}%)")
        print(f"  API Errors: {self.overall_stats.api_errors} "
              f"({100*self.overall_stats.api_errors/max(self.overall_stats.total_failures,1):.1f}%)")
        print(f"  Unknown Errors: {self.overall_stats.unknown_errors} "
              f"({100*self.overall_stats.unknown_errors/max(self.overall_stats.total_failures,1):.1f}%)")
        
        print(f"\nBy Method:")
        for method, stats in sorted(self.results_by_method.items()):
            print(f"  {method}: {stats.total_failures}/{stats.total_problems} failures "
                  f"({(1-stats.success_rate)*100:.1f}% failure rate)")
        
        if self.results_by_dataset:
            print(f"\nBy Dataset:")
            for dataset, stats in sorted(self.results_by_dataset.items()):
                print(f"  {dataset}: {stats.total_failures}/{stats.total_problems} failures "
                      f"({(1-stats.success_rate)*100:.1f}% failure rate)")
        
        # Top failure patterns
        patterns = self.get_top_failure_patterns(5)
        if patterns:
            print(f"\nTop 5 Failure Patterns:")
            for i, (pattern, count) in enumerate(patterns, 1):
                print(f"  {i}. [{count}x] {pattern[:80]}")
    
    def export_results(self, output_dir: str):
        """Export analysis results to JSON"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Overall stats
        overall_file = output_path / "failure_analysis_overall.json"
        with open(overall_file, 'w') as f:
            json.dump({
                "overall": self.overall_stats.to_dict(),
                "by_method": {
                    method: stats.to_dict() 
                    for method, stats in self.results_by_method.items()
                },
                "by_dataset": {
                    dataset: stats.to_dict()
                    for dataset, stats in self.results_by_dataset.items()
                },
                "top_patterns": self.get_top_failure_patterns(10)
            }, f, indent=2)
        
        print(f"\n✓ Exported overall statistics to: {overall_file}")
        
        # Detailed failure cases
        cases_file = output_path / "failure_cases.json"
        cases_data = []
        for failure in self.overall_stats.failure_cases:
            cases_data.append({
                "question": failure.question,
                "expected": failure.expected_answer,
                "predicted": failure.predicted_answer,
                "method": failure.method,
                "dataset": failure.dataset,
                "error_type": failure.error_type,
                "error_message": failure.error_message,
                "code": failure.code,
                "problem_id": failure.problem_id
            })
        
        with open(cases_file, 'w') as f:
            json.dump(cases_data, f, indent=2)
        
        print(f"✓ Exported {len(cases_data)} failure cases to: {cases_file}")


def main():
    parser = argparse.ArgumentParser(description="Analyze failure cases in MathCoRL")
    parser.add_argument('--results', nargs='+', help='Path(s) to result JSON files')
    parser.add_argument('--logs', help='Path to API usage logs (JSONL)')
    parser.add_argument('--dataset', help='Filter by dataset')
    parser.add_argument('--method', help='Filter by method')
    parser.add_argument('--export', default='results/failures/', help='Export directory')
    parser.add_argument('--max-logs', type=int, help='Max log entries to process')
    
    args = parser.parse_args()
    
    analyzer = FailureAnalyzer()
    
    # Load latency results
    if args.results:
        for result_file in args.results:
            print(f"Loading: {result_file}")
            analyzer.load_latency_results(result_file)
    
    # Load API logs
    if args.logs:
        print(f"Loading API logs: {args.logs}")
        analyzer.load_api_logs(args.logs, args.max_logs)
    
    # If no input provided, try default locations
    if not args.results and not args.logs:
        print("No input specified. Trying default locations...")
        
        # Try latency results
        latency_dir = Path("results/latency")
        if latency_dir.exists():
            for result_file in latency_dir.glob("*.json"):
                print(f"Loading: {result_file}")
                analyzer.load_latency_results(str(result_file))
        
        # Try API logs
        api_log = Path("logs/api_usage.jsonl")
        if api_log.exists():
            print(f"Loading: {api_log}")
            analyzer.load_api_logs(str(api_log), args.max_logs)
    
    # Compute statistics
    analyzer.compute_success_rates()
    
    # Print summary
    analyzer.print_summary()
    
    # Export results
    if args.export:
        analyzer.export_results(args.export)
    
    print("\n" + "="*70)
    print(f"Analysis complete. Found {analyzer.overall_stats.total_failures} failures "
          f"out of {analyzer.overall_stats.total_problems} problems.")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
