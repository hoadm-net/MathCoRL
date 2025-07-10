#!/usr/bin/env python3
"""
MathCoRL - Unified Entry Point for Mathematical Intelligence

This script replaces the old separate scripts (fpp.py, cot.py, fpp_prompting.py, cot_prompting.py)
with a single, clean interface for all mathematical problem solving tasks.

Quick Start:
    # Interactive mode - choose your method interactively
    python mathcorl.py

    # Single problem solving
    python mathcorl.py solve --method fpp "What is 15 + 27?"
    python mathcorl.py solve --method cot "John has 20 apples. He gives 8 to his friend."

    # Dataset testing
    python mathcorl.py test fpp SVAMP --limit 50
    python mathcorl.py test cot GSM8K --limit 100

    # Compare methods
    python mathcorl.py compare SVAMP --limit 20
    
    # API Usage Tracking
    python mathcorl.py stats                    # Show usage statistics
    python mathcorl.py export --format json    # Export tracking data
    python mathcorl.py chart --type all        # Generate visualization charts
    python mathcorl.py clear-logs              # Clear tracking logs

Legacy Compatibility:
    This script maintains compatibility with the old usage patterns while providing
    a cleaner, more unified interface.
"""

import sys
import argparse
from mint.cli import main as mint_cli_main


def legacy_fpp_mode(args):
    """Handle legacy FPP script compatibility."""
    if args.question:
        # Convert to new solve command
        sys.argv = ['mathcorl.py', 'solve', '--method', 'fpp', '--question', args.question]
        if args.context:
            sys.argv.extend(['--context', args.context])
        if args.no_code:
            sys.argv.append('--no-code')
    else:
        # Interactive mode
        sys.argv = ['mathcorl.py', 'interactive']
    
    mint_cli_main()


def legacy_cot_mode(args):
    """Handle legacy CoT script compatibility."""
    # Convert to new solve command
    sys.argv = ['mathcorl.py', 'solve', '--method', 'cot', '--question', args.question]
    if args.context:
        sys.argv.extend(['--context', args.context])
    
    mint_cli_main()


def legacy_dataset_testing(method, dataset, args):
    """Handle legacy dataset testing compatibility."""
    sys.argv = ['mathcorl.py', 'test', '--method', method, '--dataset', dataset]
    if args.limit:
        sys.argv.extend(['--limit', str(args.limit)])
    if getattr(args, 'verbose', False):
        sys.argv.append('--verbose')
    if hasattr(args, 'output') and args.output != 'results':
        sys.argv.extend(['--output', args.output])
    
    mint_cli_main()


def main():
    """Main entry point with legacy compatibility."""
    
    # Check if being called with legacy patterns
    if len(sys.argv) > 1:
        
        # Legacy fpp_prompting.py pattern: python script.py DATASET [options]
        if (len(sys.argv) >= 2 and 
            sys.argv[1].upper() in ['SVAMP', 'GSM8K', 'TABMWP', 'TAT-QA', 'TATQA', 'FINQA', 'FIN-QA']):
            
            dataset = sys.argv[1]
            
            # Parse remaining arguments
            parser = argparse.ArgumentParser()
            parser.add_argument('dataset')
            parser.add_argument('--limit', type=int)
            parser.add_argument('--verbose', '-v', action='store_true')
            parser.add_argument('--output', default='results')
            parser.add_argument('--method', default='fpp', choices=['fpp', 'cot', 'pot'])
            
            args = parser.parse_args()
            legacy_dataset_testing(args.method, dataset, args)
            return
        
        # Check for solve patterns
        elif sys.argv[1] in ['solve', 'test', 'compare', 'interactive', 'datasets', 'stats', 'clear-logs', 'export', 'chart']:
            # New format, pass directly to CLI
            mint_cli_main()
            return
    
    # Default: if no arguments or unrecognized pattern, show help and start interactive
    if len(sys.argv) == 1:
        print("🚀 MathCoRL - Mathematical Intelligence")
        print("=" * 40)
        print("Starting interactive mode...")
        print("Use 'python mathcorl.py --help' for all options")
        print()
        sys.argv = ['mathcorl.py', 'interactive']
        mint_cli_main()
    else:
        # Fallback to standard argument parsing
        parser = argparse.ArgumentParser(
            description="MathCoRL - Mathematical Intelligence",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog=__doc__
        )
        
        subparsers = parser.add_subparsers(dest='command', help='Commands')
        
        # Solve command
        solve_parser = subparsers.add_parser('solve', help='Solve a single problem')
        solve_parser.add_argument('--method', '-m', choices=['fpp', 'cot', 'pot'], required=True)
        solve_parser.add_argument('question')
        solve_parser.add_argument('--context', '-c', default='')
        solve_parser.add_argument('--no-code', action='store_true')
        
        # Test command  
        test_parser = subparsers.add_parser('test', help='Test on dataset')
        test_parser.add_argument('method', choices=['fpp', 'cot', 'pot'])
        test_parser.add_argument('dataset')
        test_parser.add_argument('--limit', type=int)
        test_parser.add_argument('--verbose', '-v', action='store_true')
        test_parser.add_argument('--output', default='results')
        
        # Compare command
        compare_parser = subparsers.add_parser('compare', help='Compare methods')
        compare_parser.add_argument('dataset')
        compare_parser.add_argument('--limit', type=int)
        
        # Interactive and datasets
        subparsers.add_parser('interactive', help='Interactive mode')
        subparsers.add_parser('datasets', help='List datasets')
        
        args = parser.parse_args()
        
        # Convert to mint CLI format and call
        if args.command == 'solve':
            sys.argv = ['mathcorl.py', 'solve', '--method', args.method, '--question', args.question]
            if args.context:
                sys.argv.extend(['--context', args.context])
            if args.no_code:
                sys.argv.append('--no-code')
        elif args.command == 'test':
            sys.argv = ['mathcorl.py', 'test', '--method', args.method, '--dataset', args.dataset]
            if args.limit:
                sys.argv.extend(['--limit', str(args.limit)])
            if args.verbose:
                sys.argv.append('--verbose')
            if args.output != 'results':
                sys.argv.extend(['--output', args.output])
        elif args.command == 'compare':
            sys.argv = ['mathcorl.py', 'compare', '--dataset', args.dataset]
            if args.limit:
                sys.argv.extend(['--limit', str(args.limit)])
        elif args.command in ['interactive', 'datasets']:
            sys.argv = ['mathcorl.py', args.command]
        
        mint_cli_main()


if __name__ == "__main__":
    main() 