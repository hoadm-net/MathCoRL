"""
Command Line Interface for MINT library.
"""

import argparse
import logging
import sys
from typing import Optional

from .core import FunctionPrototypePrompting, solve_math_problem


def setup_logging(level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def solve_command(args):
    """Handle solve command."""
    try:
        result = solve_math_problem(args.question, args.context)
        if result is not None:
            print(f"Result: {result}")
        else:
            print("Could not solve the problem")
            sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


def interactive_command(args):
    """Handle interactive command."""
    print("MINT - Mathematical Intelligence Library")
    print("Interactive Mode - Type 'exit' to quit")
    print()
    
    try:
        fpp = FunctionPrototypePrompting()
    except Exception as e:
        print(f"Error initializing FPP: {e}")
        sys.exit(1)
    
    while True:
        try:
            question = input("Question: ").strip()
            if question.lower() == 'exit':
                break
            if not question:
                continue
                
            context = input("Context (optional): ").strip()
            result = fpp.solve(question, context)
            
            if result is not None:
                print(f"Result: {result}")
            else:
                print("Could not solve the problem")
            print()
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="mint-fpp",
        description="MINT - Mathematical Intelligence Library"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Solve command
    solve_parser = subparsers.add_parser("solve", help="Solve a mathematical question")
    solve_parser.add_argument("question", help="Mathematical question to solve")
    solve_parser.add_argument("--context", "-c", default="", help="Optional context")
    solve_parser.set_defaults(func=solve_command)
    
    # Interactive command
    interactive_parser = subparsers.add_parser("interactive", help="Interactive mode")
    interactive_parser.set_defaults(func=interactive_command)
    
    # Global options
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO")
    
    args = parser.parse_args()
    setup_logging(args.log_level)
    
    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main() 