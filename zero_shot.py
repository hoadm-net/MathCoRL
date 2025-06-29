#!/usr/bin/env python3
"""
Legacy script for Zero-Shot prompting.

This script is maintained for backward compatibility. 
For new projects, use the unified interface:
    python mathcorl.py solve --method zero_shot "your question"
    python -m mint.cli solve --method zero_shot "your question"
"""

import sys
import argparse
from mint.zero_shot import ZeroShotPrompting


def main():
    parser = argparse.ArgumentParser(
        description="Solve mathematical problems using Zero-Shot prompting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python zero_shot.py "What is 15 + 27?"
    python zero_shot.py "John has 20 apples. He gives 1/4 to his friend." --context "John wants to know how many are left"
    python zero_shot.py "A train travels 120 miles in 2 hours. What is its speed?" --model gpt-3.5-turbo

For more advanced features, use the unified interface:
    python mathcorl.py solve --method zero_shot "your question"
        """
    )
    
    parser.add_argument(
        "question",
        help="The mathematical question to solve"
    )
    
    parser.add_argument(
        "--context", "-c",
        default="",
        help="Additional context for the problem"
    )
    
    # Load default model from config
    from mint.config import load_config
    config = load_config()
    default_model = config['model']
    
    parser.add_argument(
        "--model", "-m",
        default=default_model,
        help=f"OpenAI model to use (default: {default_model})"
    )
    
    parser.add_argument(
        "--no-reasoning",
        action="store_true",
        help="Don't show the reasoning process"
    )
    
    args = parser.parse_args()
    
    try:
        print("🎯 Zero-Shot Mathematical Problem Solver")
        print("=" * 50)
        
        # Initialize Zero-Shot
        zs = ZeroShotPrompting(model_name=args.model)
        
        # Solve the problem
        print(f"📝 Question: {args.question}")
        if args.context:
            print(f"🔍 Context: {args.context}")
        print()
        
        result = zs.solve(
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