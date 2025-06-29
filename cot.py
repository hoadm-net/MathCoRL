#!/usr/bin/env python3
"""
Simple Chain-of-Thought (CoT) Prompting Script

This script provides a simple interface for solving mathematical problems
using Chain-of-Thought prompting methodology.

Usage:
    python cot.py --question "What is 2 + 3?"
    python cot.py --question "John has 10 apples..." --context "Additional info"
"""

import argparse
import sys
from mint.cot import ChainOfThoughtPrompting


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Simple Chain-of-Thought (CoT) Prompting Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cot.py --question "What is 15 + 27?"
  python cot.py --question "John has 20 apples. He gives 8 to his friend. How many does he have left?"
  python cot.py --question "Calculate the average" --context "Numbers: 10, 20, 30"
        """
    )
    
    parser.add_argument(
        '--question', '-q',
        required=True,
        help='Mathematical question to solve'
    )
    
    parser.add_argument(
        '--context', '-c',
        default='',
        help='Additional context for the problem'
    )
    
    parser.add_argument(
        '--model',
        default='gpt-4',
        help='OpenAI model to use (default: gpt-4)'
    )
    
    parser.add_argument(
        '--no-reasoning',
        action='store_true',
        help='Hide the reasoning steps (show only the answer)'
    )
    
    args = parser.parse_args()
    
    try:
        print("🚀 Chain-of-Thought Mathematical Problem Solver")
        print("=" * 50)
        
        # Initialize CoT
        cot = ChainOfThoughtPrompting(model_name=args.model)
        
        # Solve the problem
        print(f"📝 Question: {args.question}")
        if args.context:
            print(f"🔍 Context: {args.context}")
        print()
        
        result = cot.solve(
            question=args.question,
            context=args.context,
            show_reasoning=not args.no_reasoning
        )
        
        if args.no_reasoning:
            print(f"📊 Answer: {result['result']}")
        
        print("=" * 50)
        print("✅ Problem solved successfully!")
        
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 