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
"""

import argparse
import sys
import logging
from typing import Optional, Dict, Any

from .core import FunctionPrototypePrompting
from .cot import ChainOfThoughtPrompting
from .pot import ProgramOfThoughtsPrompting
from .testing import TestRunner, DatasetLoader, create_fpp_solver, create_cot_solver, create_pot_solver
from .evaluation import get_tolerance_function


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
    print("Type 'exit' to quit, 'help' for help, 'switch' to change method\n")
    
    # Choose initial method
    while True:
        method_choice = input("Select method (1 for FPP, 2 for CoT, 3 for PoT): ").strip()
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
        else:
            print("Please enter 1, 2, or 3")
    
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
                
                choice = input("Select method (1, 2, or 3): ").strip()
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
            
            else:  # PoT
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
        method: 'fpp', 'cot', or 'pot'
        question: Mathematical question to solve
        context: Optional context information
        show_code: Whether to display generated code (FPP and PoT)
        
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
        
        else:
            raise ValueError(f"Unknown method: {method}. Use 'fpp', 'cot', or 'pot'")
            
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
        method: 'fpp', 'cot', or 'pot'
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
    else:
        raise ValueError(f"Unknown method: {method}. Use 'fpp', 'cot', or 'pot'")
    
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
    solve_parser.add_argument('--method', '-m', choices=['fpp', 'cot', 'pot'], required=True,
                             help='Prompting method to use')
    solve_parser.add_argument('--question', '-q', required=True,
                             help='Mathematical question to solve')
    solve_parser.add_argument('--context', '-c', default='',
                             help='Optional context for the problem')
    solve_parser.add_argument('--no-code', action='store_true',
                             help='Hide generated code (FPP and PoT)')
    
    # Dataset testing
    test_parser = subparsers.add_parser('test', help='Test method on dataset')
    test_parser.add_argument('--method', '-m', choices=['fpp', 'cot', 'pot'], required=True,
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
            if result.get('success', False):
                print(f"✅ Answer: {result['result']}")
            else:
                print("❌ Could not solve the problem")
                if 'error' in result:
                    print(f"Error: {result['error']}")
        
        elif args.command == 'test':
            test_method(args.method, args.dataset, args.limit, 
                       args.verbose, args.output)
        
        elif args.command == 'compare':
            compare_methods(args.dataset, args.limit)
        
        elif args.command == 'datasets':
            list_datasets()
        
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