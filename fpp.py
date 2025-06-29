#!/usr/bin/env python3
"""
Simple Function Prototype Prompting (FPP) script.

This script provides a simple interface for mathematical problem solving:
- Input: question (str), context (str, optional)  
- Output: result (numerical answer)

Usage:
    python fpp.py
    python fpp.py --question "What is 5 + 3?"
    python fpp.py --question "John has 10 apples" --context "He gives away 3"
"""

import argparse
import logging
import sys
from typing import Optional
import json
from mint.utils import load_svamp_test_data

# Import mint library
try:
    from mint import FunctionPrototypePrompting, solve_math_problem
except ImportError as e:
    print("❌ Error: mint library not found. Please install it first:")
    print("   python install_mint.py")
    sys.exit(1)


def setup_logging(level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def solve_interactive():
    """Interactive mode for solving problems."""
    print("🚀 Function Prototype Prompting (FPP)")
    print("=" * 50)
    print("Interactive Mathematical Problem Solver")
    print("Type 'exit' to quit, 'help' for help\n")
    
    # Initialize FPP
    try:
        fpp = FunctionPrototypePrompting()
        print("✅ FPP initialized successfully!\n")
    except Exception as e:
        print(f"❌ Error initializing FPP: {e}")
        print("Make sure you have set OPENAI_API_KEY in your .env file")
        return
    
    while True:
        try:
            # Get question from user
            question = input("🔍 Enter mathematical question: ").strip()
            
            if question.lower() == 'exit':
                print("👋 Goodbye!")
                break
            elif question.lower() == 'help':
                print_help()
                continue
            elif not question:
                continue
            
            # Get optional context
            context = input("📝 Enter context (optional, press Enter to skip): ").strip()
            
            # Solve problem
            print("🔄 Solving...")
            detailed_result = fpp.solve_detailed(question, context)
            
            # Show generated code
            if detailed_result['code']:
                print("\n🐍 Generated Python Code:")
                print("-" * 30)
                print(detailed_result['code'])
                print("-" * 30)
            
            # Show result
            if detailed_result['success']:
                print(f"✅ Result: {detailed_result['result']}")
            else:
                print("❌ Could not solve the problem")
                if detailed_result['error']:
                    print(f"Error: {detailed_result['error']}")
            
            print("-" * 50)
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


def print_help():
    """Print help information."""
    print("""
📖 Help - Function Prototype Prompting

FPP solves mathematical word problems by generating Python code.

Examples:
• "What is 15 + 27?"
• "John has 25 marbles. He gives 7 to his friend. How many marbles does John have left?"
• "A pizza is cut into 8 slices. If 3 slices are eaten, how many remain?"
• "There are 12 students. Each student has 3 books. How many books in total?"

Features:
• Displays both the generated Python code and the result
• Shows step-by-step mathematical reasoning in code form
• Uses predefined mathematical functions for reliable computation

Available functions:
• Basic: add, sub, mul, div, mod, pow
• Math: min, max, abs, round, floor, ceil
• Lists: sum, mean, median, mode, count
• Utilities: percentage, greater_than, less_than, equal, gcd, lcm

Commands:
• exit - Quit the program
• help - Show this help message

Command Line Options:
• --question "question" - Solve single question
• --context "context" - Add context information
• --no-code - Hide generated Python code
• --show-code - Show generated Python code (default)
    """)


def solve_single(question: str, context: str = "", show_code: bool = True) -> Optional[float]:
    """
    Solve a single mathematical question.
    
    Args:
        question: Mathematical question to solve
        context: Optional context information
        show_code: Whether to display generated Python code
        
    Returns:
        Numerical result or None if solving failed
    """
    try:
        # Use FPP class directly to get detailed output
        fpp = FunctionPrototypePrompting()
        detailed_result = fpp.solve_detailed(question, context)
        
        if show_code and detailed_result['code']:
            print("🐍 Generated Python Code:")
            print("-" * 30)
            print(detailed_result['code'])
            print("-" * 30)
        
        if detailed_result['success']:
            return detailed_result['result']
        else:
            if detailed_result['error']:
                print(f"❌ Error: {detailed_result['error']}")
            return None
            
    except Exception as e:
        print(f"❌ Error solving problem: {e}")
        return None


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Function Prototype Prompting (FPP) for Mathematical Problem Solving"
    )
    parser.add_argument(
        "--question", "-q",
        help="Mathematical question to solve"
    )
    parser.add_argument(
        "--context", "-c", 
        default="",
        help="Optional context for the question"
    )
    parser.add_argument(
        "--log-level", 
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed output"
    )
    parser.add_argument(
        "--show-code",
        action="store_true",
        default=True,
        help="Show generated Python code (default: True)"
    )
    parser.add_argument(
        "--no-code",
        action="store_true",
        help="Hide generated Python code"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    # Single question mode
    if args.question:
        print("🚀 Function Prototype Prompting (FPP)")
        print("=" * 50)
        print(f"Question: {args.question}")
        if args.context:
            print(f"Context: {args.context}")
        print()
        
        # Determine whether to show code
        show_code = not args.no_code if args.no_code else args.show_code
        
        print("🔄 Solving...")
        result = solve_single(args.question, args.context, show_code)
        
        if result is not None:
            print(f"✅ Result: {result}")
        else:
            print("❌ Could not solve the problem")
            sys.exit(1)
    
    # Interactive mode
    else:
        solve_interactive()


if __name__ == "__main__":
    main()

# Load test data
svamp_test_data = load_svamp_test_data('datasets/SVAMP/test.json')

# Initialize results list
results = []

# Test with FPP
for sample in svamp_test_data:
    question = sample['question']
    context = sample['context']
    ground_truth = sample['ground_truth']
    
    # Solve using FPP
    result = solve_single(question, context)
    
    # Append result
    results.append({
        'question': question,
        'context': context,
        'ground_truth': ground_truth,
        'result': result,
        'correct': result == ground_truth
    })

# Save results to file
with open('svamp_test_results.json', 'w') as f:
    json.dump(results, f, indent=4)

# Calculate statistics
correct_count = sum(1 for r in results if r['correct'])
total = len(results)
accuracy = correct_count / total * 100

# Print statistics
print(f"Tested {total} samples.")
print(f"Correct: {correct_count}")
print(f"Accuracy: {accuracy:.2f}%") 