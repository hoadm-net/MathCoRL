#!/usr/bin/env python3
"""
Unified CLI for MathCoRL - Mathematical Intelligence with Advanced Prompting Methods

This CLI provides a comprehensive interface for mathematical problem solving using
Function Prototype Prompting (FPP) and Chain-of-Thought (CoT) methods.

Usage Examples:
    # Interactive mode
    python -m mint.cli interactive

    # Single problem solving
    python -m mint.cli solve --method fpp --question "What is 15 + 27?"
    python -m mint.cli solve --method cot --question "John has 20 apples..."

    # Dataset testing
    python -m mint.cli test --method fpp --dataset SVAMP --limit 50
    python -m mint.cli test --method cot --dataset GSM8K --limit 100

    # Method comparison
    python -m mint.cli compare --dataset SVAMP --limit 20

    # List available datasets
    python -m mint.cli datasets

    # API usage tracking
    python -m mint.cli stats --hours 24        # Show usage stats
    python -m mint.cli export --format csv     # Export tracking data
    python -m mint.cli clear-logs              # Clear tracking logs
"""

import argparse
import sys
import logging
from typing import Optional, Dict, Any

from .core import FunctionPrototypePrompting
from .cot import ChainOfThoughtPrompting
from .pot import ProgramOfThoughtsPrompting
from .zero_shot import ZeroShotPrompting
from .pal import ProgramAidedLanguageModel
from .testing import TestRunner, DatasetLoader, create_fpp_solver, create_cot_solver, create_pot_solver, create_zero_shot_solver, create_pal_solver
from .evaluation import get_tolerance_function
from .tracking import get_tracker


def setup_logging(level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def interactive_mode():
    """Interactive mode for solving problems."""
    print("🚀 MathCoRL - Interactive Mathematical Problem Solver")
    print("=" * 60)
    print("Choose your method:")
    print("1. FPP (Function Prototype Prompting) - Code generation with function prototypes")
    print("2. CoT (Chain-of-Thought) - Step-by-step reasoning")
    print("3. PoT (Program of Thoughts) - Generate Python code to solve problems")
    print("4. Zero-Shot - Direct problem solving without examples")
    print("5. PAL (Program-aided Language Models) - Reasoning + code generation")
    print("Type 'exit' to quit, 'help' for help, 'switch' to change method\n")
    
    # Choose initial method
    while True:
        method_choice = input("Select method (1 for FPP, 2 for CoT, 3 for PoT, 4 for Zero-Shot, 5 for PAL): ").strip()
        if method_choice == '1':
            method = 'fpp'
            solver = FunctionPrototypePrompting()
            print("✅ FPP (Function Prototype Prompting) selected!\n")
            break
        elif method_choice == '2':
            method = 'cot'
            solver = ChainOfThoughtPrompting()
            print("✅ CoT (Chain-of-Thought) selected!\n")
            break
        elif method_choice == '3':
            method = 'pot'
            solver = ProgramOfThoughtsPrompting()
            print("✅ PoT (Program of Thoughts) selected!\n")
            break
        elif method_choice == '4':
            method = 'zero_shot'
            solver = ZeroShotPrompting()
            print("✅ Zero-Shot selected!\n")
            break
        elif method_choice == '5':
            method = 'pal'
            solver = ProgramAidedLanguageModel()
            print("✅ PAL (Program-aided Language Models) selected!\n")
            break
        else:
            print("Please enter 1, 2, 3, 4, or 5")
    
    while True:
        try:
            # Get question from user
            question = input(f"🔍 Enter mathematical question ({method.upper()}): ").strip()
            
            if question.lower() == 'exit':
                print("👋 Goodbye!")
                break
            elif question.lower() == 'help':
                print_interactive_help()
                continue
            elif question.lower() == 'switch':
                print("Switch to:")
                print("1. FPP (Function Prototype Prompting)")
                print("2. CoT (Chain-of-Thought)")
                print("3. PoT (Program of Thoughts)")
                print("4. Zero-Shot")
                print("5. PAL (Program-aided Language Models)")
                
                choice = input("Select method (1, 2, 3, 4, or 5): ").strip()
                if choice == '1':
                    method = 'fpp'
                    solver = FunctionPrototypePrompting()
                    print("🔄 Switched to FPP (Function Prototype Prompting)")
                elif choice == '2':
                    method = 'cot'
                    solver = ChainOfThoughtPrompting()
                    print("🔄 Switched to CoT (Chain-of-Thought)")
                elif choice == '3':
                    method = 'pot'
                    solver = ProgramOfThoughtsPrompting()
                    print("🔄 Switched to PoT (Program of Thoughts)")
                elif choice == '4':
                    method = 'zero_shot'
                    solver = ZeroShotPrompting()
                    print("🔄 Switched to Zero-Shot")
                elif choice == '5':
                    method = 'pal'
                    solver = ProgramAidedLanguageModel()
                    print("🔄 Switched to PAL (Program-aided Language Models)")
                else:
                    print("❌ Invalid choice. Staying with current method.")
                continue
            elif not question:
                continue
            
            # Get optional context
            context = input("📝 Enter context (optional, press Enter to skip): ").strip()
            
            # Solve problem
            print("🔄 Solving...")
            
            if method == 'fpp':
                result = solver.solve_detailed(question, context)
                
                # Show generated code
                if result['code']:
                    print("\n🐍 Generated Python Code:")
                    print("-" * 30)
                    print(result['code'])
                    print("-" * 30)
                
                # Show result
                if result['success']:
                    print(f"✅ Result: {result['result']}")
                else:
                    print("❌ Could not solve the problem")
                    if result['error']:
                        print(f"Error: {result['error']}")
            
            elif method == 'cot':
                result = solver.solve(question, context, show_reasoning=True)
                print(f"✅ Final Answer: {result['result']}")
            
            elif method == 'pot':
                result = solver.solve(question, context, show_reasoning=True)
                print(f"✅ Final Answer: {result['result']}")
            
            elif method == 'zero_shot':
                result = solver.solve(question, context, show_reasoning=True)
                print(f"✅ Final Answer: {result['result']}")
            
            else:  # pal
                result = solver.solve(question, context, show_reasoning=True)
                print(f"✅ Final Answer: {result['result']}")
            
            print("-" * 50)
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


def print_interactive_help():
    """Print help information for interactive mode."""
    print("""
📖 Interactive Help - MathCoRL

🔧 FPP (Function Prototype Prompting):
• Solves problems by generating Python code with function prototypes
• Shows the generated code and result
• High accuracy for computational problems

🧠 CoT (Chain-of-Thought):
• Solves problems with step-by-step reasoning
• Shows detailed reasoning process
• Good for understanding problem-solving logic

💻 PoT (Program of Thoughts):
• Generates Python code to solve numerical problems
• Separates computation from reasoning
• Excellent for mathematical calculations

🎯 Zero-Shot:
• Direct problem solving without examples or step-by-step guidance
• Simple and fast approach
• Good baseline for comparison

🧠 PAL (Program-aided Language Models):
• Combines natural language reasoning with code generation
• First generates reasoning steps, then writes executable code
• Best of both worlds: interpretable reasoning + computational accuracy

Examples:
• "What is 15 + 27?"
• "John has 25 marbles. He gives 7 to his friend. How many marbles does John have left?"
• "A pizza is cut into 8 slices. If 3 slices are eaten, how many remain?"

Commands:
• exit - Quit the program
• help - Show this help message
• switch - Switch between FPP and CoT methods
    """)


def solve_single(method: str, question: str, context: str = "", show_code: bool = True) -> Dict[str, Any]:
    """
    Solve a single problem using the specified method.
    
    Args:
        method: 'fpp', 'cot', 'pot', 'zero_shot', or 'pal'
        question: Mathematical question to solve
        context: Optional context information
        show_code: Whether to display generated code (FPP, PoT, and PAL)
        
    Returns:
        Dictionary with results
    """
    try:
        if method.lower() == 'fpp':
            fpp = FunctionPrototypePrompting()
            result = fpp.solve_detailed(question, context)
            
            if show_code and result['code']:
                print("🐍 Generated Python Code:")
                print("-" * 30)
                print(result['code'])
                print("-" * 30)
            
            return result
        
        elif method.lower() == 'cot':
            cot = ChainOfThoughtPrompting()
            result = cot.solve(question, context, show_reasoning=True)
            return result
        
        elif method.lower() == 'pot':
            pot = ProgramOfThoughtsPrompting()
            result = pot.solve(question, context, show_reasoning=True)
            return result
        
        elif method.lower() == 'zero_shot':
            zs = ZeroShotPrompting()
            result = zs.solve(question, context, show_reasoning=True)
            return result
        
        elif method.lower() == 'pal':
            pal = ProgramAidedLanguageModel()
            result = pal.solve(question, context, show_reasoning=True)
            
            if show_code and result.get('code'):
                print("🐍 Generated Code:")
                print("-" * 30)
                print(result['code'])
                print("-" * 30)
            
            return result
        
        else:
            raise ValueError(f"Unknown method: {method}. Use 'fpp', 'cot', 'pot', 'zero_shot', or 'pal'")
            
    except Exception as e:
        return {
            'result': None,
            'error': str(e),
            'success': False
        }


def test_method(method: str, dataset: str, limit: Optional[int] = None, 
               verbose: bool = False, output_dir: str = "results") -> Dict[str, Any]:
    """
    Test a method on a dataset.
    
    Args:
        method: 'fpp', 'cot', 'pot', 'zero_shot', or 'pal'
        dataset: Dataset name
        limit: Maximum number of samples
        verbose: Show detailed output
        output_dir: Directory to save results
        
    Returns:
        Test results
    """
    if method.lower() == 'fpp':
        solver = create_fpp_solver()
        runner = TestRunner('FPP', solver)
    elif method.lower() == 'cot':
        solver = create_cot_solver()
        runner = TestRunner('CoT', solver)
    elif method.lower() == 'pot':
        solver = create_pot_solver()
        runner = TestRunner('PoT', solver)
    elif method.lower() == 'zero_shot':
        solver = create_zero_shot_solver()
        runner = TestRunner('Zero-Shot', solver)
    elif method.lower() == 'pal':
        solver = create_pal_solver()
        runner = TestRunner('PAL', solver)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'fpp', 'cot', 'pot', 'zero_shot', or 'pal'")
    
    return runner.test_dataset(dataset, limit, verbose, output_dir)


def compare_methods(dataset: str, limit: Optional[int] = None) -> Dict[str, Any]:
    """
    Compare FPP and CoT methods on a dataset.
    
    Args:
        dataset: Dataset name
        limit: Maximum number of samples
        
    Returns:
        Comparison results
    """
    fpp_solver = create_fpp_solver()
    cot_solver = create_cot_solver()
    
    fpp_runner = TestRunner('FPP', fpp_solver)
    cot_runner = TestRunner('CoT', cot_solver)
    
    return fpp_runner.compare_methods(cot_runner, dataset, limit)


def list_datasets():
    """List available datasets."""
    datasets = DatasetLoader.get_supported_datasets()
    print("📊 Available Datasets:")
    print("=" * 30)
    for dataset in datasets:
        print(f"• {dataset}")
    print()
    print("Usage: python -m mint.cli test --dataset DATASET_NAME")


def show_tracking_stats(hours: int = 24):
    """Show API usage tracking statistics with detailed breakdown."""
    print(f"📊 API Usage Statistics (Last {hours} hours)")
    print("=" * 50)
    
    try:
        tracker = get_tracker()
        summary = tracker.get_usage_summary(last_n_hours=hours)
        
        if 'error' in summary:
            print(f"❌ {summary['error']}")
            return
        
        if 'message' in summary:
            print(f"💡 {summary['message']}")
            return
        
        # Overall summary
        stats = summary['summary']
        print("🔥 OVERVIEW")
        print(f"   Total Requests: {stats['total_requests']}")
        print(f"   ✅ Successful: {stats['successful_requests']}")
        print(f"   ❌ Failed: {stats['total_requests'] - stats['successful_requests']}")
        print(f"   📈 Success Rate: {stats['success_rate']}")
        print(f"   🔢 Total Tokens: {stats['total_tokens']:,}")
        print(f"   💰 Total Cost: {stats['total_cost']}")
        print(f"   ⏱️  Avg Time: {stats['avg_execution_time']}")
        print()
        
        # Enhanced method breakdown with averages
        if summary['by_method']:
            print("🔧 BY METHOD (Detailed Analysis)")
            print("-" * 70)
            print(f"{'Method':<12} {'Reqs':<5} {'Avg Input':<10} {'Avg Output':<11} {'Avg Time':<9} {'Avg Cost':<10}")
            print("-" * 70)
            
            for method, data in summary['by_method'].items():
                avg_input = data['avg_input_tokens'] if 'avg_input_tokens' in data else 0
                avg_output = data['avg_output_tokens'] if 'avg_output_tokens' in data else 0
                avg_time = data['avg_time'] if 'avg_time' in data else 0
                avg_cost = data['cost'] / data['requests'] if data['requests'] > 0 else 0
                
                print(f"{method:<12} {data['requests']:<5} {avg_input:<10.0f} {avg_output:<11.0f} {avg_time:<9.2f}s ${avg_cost:<9.6f}")
            print()
        
        # Model breakdown
        if summary['by_model']:
            print("🤖 BY MODEL")
            for model, data in summary['by_model'].items():
                print(f"   {model}: {data['requests']} requests, {data['tokens']:,} tokens, ${data['cost']:.6f}")
            print()
            
        print("💡 Use 'python mathcorl.py stats --hours N' to see stats for different time periods")
        
    except Exception as e:
        print(f"❌ Error loading tracking data: {e}")


def clear_tracking_logs():
    """Clear API usage tracking logs."""
    import os
    from pathlib import Path
    from datetime import datetime
    
    log_file = Path("logs/api_usage.jsonl")
    
    if not log_file.exists():
        print("💡 No tracking logs found to clear.")
        return
    
    # Create backup
    backup_name = f"logs/api_usage_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    backup_path = Path(backup_name)
    
    try:
        # Backup current log
        if log_file.stat().st_size > 0:
            os.rename(str(log_file), str(backup_path))
            print(f"✅ Logs backed up to: {backup_path}")
        
        # Create empty log file
        log_file.touch()
        print("✅ Tracking logs cleared.")
        
    except Exception as e:
        print(f"❌ Error clearing logs: {e}")


def export_tracking_data(format: str = "csv"):
    """Export tracking data to different formats."""
    import json
    from pathlib import Path
    from datetime import datetime
    
    log_file = Path("logs/api_usage.jsonl")
    
    if not log_file.exists():
        print("❌ No tracking logs found to export.")
        return
    
    try:
        # Load data
        data = []
        with open(log_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line.strip()))
        
        if not data:
            print("💡 No tracking data found.")
            return
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if format.lower() == "csv":
            try:
                import pandas as pd
                df = pd.DataFrame(data)
                output_file = f"tracking_export_{timestamp}.csv"
                df.to_csv(output_file, index=False)
                print(f"✅ Data exported to: {output_file}")
            except ImportError:
                print("❌ pandas not installed. Install with: pip install pandas")
        
        elif format.lower() == "json":
            output_file = f"tracking_export_{timestamp}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"✅ Data exported to: {output_file}")
        
        else:
            print("❌ Unsupported format. Use 'csv' or 'json'.")
            
    except Exception as e:
        print(f"❌ Error exporting data: {e}")


def generate_charts(chart_type: str = "all", hours: int = 24, save: bool = False):
    """Generate visualization charts for tracking data."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        import pandas as pd
        import numpy as np
        from datetime import datetime, timedelta
        import json
        from pathlib import Path
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
    except ImportError as e:
        print("❌ Required libraries not installed. Install with:")
        print("   pip install matplotlib seaborn pandas")
        return
    
    # Load data
    log_file = Path("logs/api_usage.jsonl")
    if not log_file.exists():
        print("❌ No tracking logs found.")
        return
    
    try:
        # Load and filter data
        data = []
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        with open(log_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    log_entry = json.loads(line.strip())
                    log_time = datetime.fromisoformat(log_entry['timestamp'])
                    if log_time >= cutoff_time:
                        data.append(log_entry)
        
        if not data:
            print(f"💡 No tracking data found in the last {hours} hours.")
            return
        
        df = pd.DataFrame(data)
        
        # Generate charts based on type
        if chart_type in ['comparison', 'all']:
            _create_method_comparison_chart(df, save)
        
        if chart_type in ['cost', 'all']:
            _create_cost_analysis_chart(df, save)
        
        if chart_type in ['time', 'all']:
            _create_time_analysis_chart(df, save)
        
        if chart_type in ['tokens', 'all']:
            _create_token_analysis_chart(df, save)
        
        if not save:
            plt.show()
        
        print(f"✅ Charts generated successfully!")
        
    except Exception as e:
        print(f"❌ Error generating charts: {e}")


def _create_method_comparison_chart(df, save=False):
    """Create method comparison chart."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    from datetime import datetime
    
    # Group by method
    method_stats = df.groupby('method').agg({
        'input_tokens': 'mean',
        'output_tokens': 'mean',
        'execution_time': 'mean',
        'total_cost': 'mean'
    }).round(2)
    
    # Define method order and colors (matching actual data)
    method_order = ['Zero-Shot', 'CoT', 'PAL', 'PoT', 'FPP']
    method_colors = {
        'Zero-Shot': '#FF6B6B',  # Red
        'CoT': '#4ECDC4',        # Teal
        'PAL': '#45B7D1',        # Blue
        'PoT': '#96CEB4',        # Green
        'FPP': '#FECA57'         # Yellow
    }
    
    # Reorder data according to method_order (only include methods that exist in data)
    available_methods = [method for method in method_order if method in method_stats.index]
    ordered_stats = method_stats.reindex(available_methods)
    
    # Create colors list in the same order
    colors = [method_colors[method] for method in available_methods]
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Average Input Tokens
    bars1 = ax1.bar(ordered_stats.index, ordered_stats['input_tokens'], color=colors)
    ax1.set_title('Average Input Tokens by Method')
    ax1.set_ylabel('Tokens')
    ax1.tick_params(axis='x', rotation=0)
    
    # Average Output Tokens
    bars2 = ax2.bar(ordered_stats.index, ordered_stats['output_tokens'], color=colors)
    ax2.set_title('Average Output Tokens by Method')
    ax2.set_ylabel('Tokens')
    ax2.tick_params(axis='x', rotation=0)
    
    # Average Execution Time
    bars3 = ax3.bar(ordered_stats.index, ordered_stats['execution_time'], color=colors)
    ax3.set_title('Average Execution Time by Method')
    ax3.set_ylabel('Seconds')
    ax3.tick_params(axis='x', rotation=0)
    
    # Average Cost
    bars4 = ax4.bar(ordered_stats.index, ordered_stats['total_cost'], color=colors)
    ax4.set_title('Average Cost by Method')
    ax4.set_ylabel('USD')
    ax4.tick_params(axis='x', rotation=0)
    
    # Add legend to the last subplot
    legend_handles = [plt.Rectangle((0,0),1,1, color=method_colors[method]) for method in available_methods]
    ax4.legend(legend_handles, available_methods, loc='upper right', fontsize=10)
    
    plt.tight_layout()
    
    if save:
        plt.savefig(f'charts/method_comparison_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()


def _create_cost_analysis_chart(df, save=False):
    """Create cost analysis chart."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    from datetime import datetime
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('💰 Cost Analysis', fontsize=16, fontweight='bold')
    
    # Cost by Method (Pie Chart)
    method_costs = df.groupby('method')['total_cost'].sum()
    ax1.pie(method_costs.values, labels=method_costs.index, autopct='%1.1f%%', startangle=90)
    ax1.set_title('Cost Distribution by Method')
    
    # Cost vs Tokens Scatter
    ax2.scatter(df['total_tokens'], df['total_cost'], c=df['method'].astype('category').cat.codes, 
               alpha=0.6, s=60)
    ax2.set_xlabel('Total Tokens')
    ax2.set_ylabel('Cost (USD)')
    ax2.set_title('Cost vs Tokens Relationship')
    
    # Add method legend
    methods = df['method'].unique()
    for i, method in enumerate(methods):
        ax2.scatter([], [], c=f'C{i}', label=method)
    ax2.legend()
    
    plt.tight_layout()
    
    if save:
        plt.savefig(f'charts/cost_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()


def _create_time_analysis_chart(df, save=False):
    """Create time analysis chart."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    from datetime import datetime
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('⏱️ Time Analysis', fontsize=16, fontweight='bold')
    
    # Execution Time by Method (Box Plot)
    sns.boxplot(data=df, x='method', y='execution_time', ax=ax1)
    ax1.set_title('Execution Time Distribution by Method')
    ax1.set_ylabel('Seconds')
    ax1.tick_params(axis='x', rotation=45)
    
    # Time vs Tokens
    ax2.scatter(df['total_tokens'], df['execution_time'], c=df['method'].astype('category').cat.codes, 
               alpha=0.6, s=60)
    ax2.set_xlabel('Total Tokens')
    ax2.set_ylabel('Execution Time (seconds)')
    ax2.set_title('Execution Time vs Tokens')
    
    # Add method legend
    methods = df['method'].unique()
    for i, method in enumerate(methods):
        ax2.scatter([], [], c=f'C{i}', label=method)
    ax2.legend()
    
    plt.tight_layout()
    
    if save:
        plt.savefig(f'charts/time_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()


def _create_token_analysis_chart(df, save=False):
    """Create token analysis chart."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    from datetime import datetime
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('🔢 Token Analysis', fontsize=16, fontweight='bold')
    
    # Input vs Output Tokens by Method
    method_stats = df.groupby('method')[['input_tokens', 'output_tokens']].mean()
    method_stats.plot(kind='bar', ax=ax1, stacked=True)
    ax1.set_title('Average Input vs Output Tokens by Method')
    ax1.set_ylabel('Tokens')
    ax1.tick_params(axis='x', rotation=45)
    ax1.legend(['Input Tokens', 'Output Tokens'])
    
    # Token Efficiency (Output/Input Ratio)
    df['token_efficiency'] = df['output_tokens'] / df['input_tokens']
    df['token_efficiency'].replace([np.inf, -np.inf], np.nan, inplace=True)
    token_eff = df.groupby('method')['token_efficiency'].mean()
    token_eff.plot(kind='bar', ax=ax2, color='purple')
    ax2.set_title('Token Efficiency (Output/Input Ratio)')
    ax2.set_ylabel('Ratio')
    ax2.tick_params(axis='x', rotation=45)
    
    # Input Tokens Distribution
    sns.histplot(data=df, x='input_tokens', hue='method', ax=ax3, alpha=0.7)
    ax3.set_title('Input Tokens Distribution')
    ax3.set_xlabel('Input Tokens')
    
    # Output Tokens Distribution
    sns.histplot(data=df, x='output_tokens', hue='method', ax=ax4, alpha=0.7)
    ax4.set_title('Output Tokens Distribution')
    ax4.set_xlabel('Output Tokens')
    
    plt.tight_layout()
    
    if save:
        plt.savefig(f'charts/token_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="MathCoRL - Mathematical Intelligence with Advanced Prompting Methods",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Interactive mode
    interactive_parser = subparsers.add_parser('interactive', help='Interactive problem solving')
    
    # Single problem solving
    solve_parser = subparsers.add_parser('solve', help='Solve a single problem')
    solve_parser.add_argument('--method', '-m', choices=['fpp', 'cot', 'pot', 'zero_shot', 'pal'], required=True,
                             help='Prompting method to use')
    solve_parser.add_argument('--question', '-q', required=True,
                             help='Mathematical question to solve')
    solve_parser.add_argument('--context', '-c', default='',
                             help='Optional context for the problem')
    solve_parser.add_argument('--no-code', action='store_true',
                             help='Hide generated code (FPP and PoT)')
    
    # Dataset testing
    test_parser = subparsers.add_parser('test', help='Test method on dataset')
    test_parser.add_argument('--method', '-m', choices=['fpp', 'cot', 'pot', 'zero_shot', 'pal'], required=True,
                            help='Prompting method to use')
    test_parser.add_argument('--dataset', '-d', required=True,
                            help='Dataset to test on')
    test_parser.add_argument('--limit', '-l', type=int,
                            help='Maximum number of samples to test')
    test_parser.add_argument('--verbose', '-v', action='store_true',
                            help='Show detailed output')
    test_parser.add_argument('--output', '-o', default='results',
                            help='Output directory for results')
    
    # Method comparison
    compare_parser = subparsers.add_parser('compare', help='Compare FPP vs CoT')
    compare_parser.add_argument('--dataset', '-d', required=True,
                               help='Dataset to test on')
    compare_parser.add_argument('--limit', '-l', type=int,
                               help='Maximum number of samples to test')
    
    # List datasets
    datasets_parser = subparsers.add_parser('datasets', help='List available datasets')
    
    # API tracking statistics
    stats_parser = subparsers.add_parser('stats', help='Show API usage statistics')
    stats_parser.add_argument('--hours', type=int, default=24,
                             help='Number of hours to look back (default: 24)')
    
    # Clear tracking logs
    clear_parser = subparsers.add_parser('clear-logs', help='Clear API tracking logs')
    
    # Export tracking data
    export_parser = subparsers.add_parser('export', help='Export tracking data')
    export_parser.add_argument('--format', choices=['csv', 'json'], default='csv',
                              help='Export format (default: csv)')
    
    # Generate charts
    chart_parser = subparsers.add_parser('chart', help='Generate visualization charts')
    chart_parser.add_argument('--type', choices=['comparison', 'cost', 'time', 'tokens', 'all'], 
                             default='all', help='Chart type to generate (default: all)')
    chart_parser.add_argument('--hours', type=int, default=24,
                             help='Number of hours to look back (default: 24)')
    chart_parser.add_argument('--save', action='store_true',
                             help='Save charts to files instead of displaying')
    
    # Global options
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default='INFO', help='Logging level')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    try:
        if args.command == 'interactive':
            interactive_mode()
        
        elif args.command == 'solve':
            result = solve_single(args.method, args.question, args.context, 
                                not args.no_code)
            # Check if we have a valid result (works for all methods)
            if result.get('result') is not None:
                print(f"✅ Answer: {result['result']}")
            else:
                print("❌ Could not solve the problem")
                if 'error' in result:
                    print(f"Error: {result['error']}")
                if 'execution_error' in result and result['execution_error']:
                    print(f"Execution Error: {result['execution_error']}")
            
            # Show tracking stats for this request
            print("\n📊 Request Stats:")
            try:
                tracker = get_tracker()
                recent_summary = tracker.get_usage_summary(last_n_hours=1)
                if 'summary' in recent_summary and recent_summary['summary']['total_requests'] > 0:
                    last_request_cost = recent_summary['summary']['total_cost']
                    last_request_tokens = recent_summary['summary']['total_tokens']
                    print(f"   💰 Cost: ${last_request_cost:.6f}")
                    print(f"   🔢 Tokens: {last_request_tokens}")
            except:
                pass  # Don't fail if tracking unavailable
        
        elif args.command == 'test':
            test_method(args.method, args.dataset, args.limit, 
                       args.verbose, args.output)
        
        elif args.command == 'compare':
            compare_methods(args.dataset, args.limit)
        
        elif args.command == 'datasets':
            list_datasets()
        
        elif args.command == 'stats':
            show_tracking_stats(args.hours)
        
        elif args.command == 'clear-logs':
            clear_tracking_logs()
        
        elif args.command == 'export':
            export_tracking_data(args.format)
        
        elif args.command == 'chart':
            generate_charts(args.type, args.hours, args.save)
        
        else:
            parser.print_help()
    
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 