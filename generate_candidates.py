#!/usr/bin/env python3
"""
MathCoRL - Step 1: Candidate Generation

Generate training candidates for In-Context Reinforcement Learning from mathematical reasoning datasets.

Usage:
    python generate_candidates.py --dataset FinQA --n-candidates 100
    python generate_candidates.py --dataset GSM8K --n-candidates 50 --output-dir custom_candidates
    python generate_candidates.py --help
"""

import argparse
import os
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from mint.icrl.candidate_generator import CandidateGenerator
from mint.config import load_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(
        description='Generate candidates for In-Context Reinforcement Learning',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python generate_candidates.py --dataset FinQA --n-candidates 100
    python generate_candidates.py --dataset GSM8K --n-candidates 50
    python generate_candidates.py --dataset SVAMP --n-candidates 75 --output-dir my_candidates
    python generate_candidates.py --dataset TabMWP --n-candidates 80 --verbose
    python generate_candidates.py --dataset TAT-QA --n-candidates 60

Supported Datasets:
    - GSM8K: Grade School Math word problems
    - SVAMP: Simple arithmetic word problems  
    - TabMWP: Tabular math word problems
    - TAT-QA: Table-and-text QA (financial)
    - FinQA: Financial reasoning QA
        """
    )
    
    # Required arguments
    parser.add_argument(
        '--dataset', '-d',
        type=str,
        required=True,
        choices=['GSM8K', 'SVAMP', 'TabMWP', 'TAT-QA', 'FinQA'],
        help='Dataset to generate candidates from'
    )
    
    # Optional arguments
    parser.add_argument(
        '--n-candidates', '-n',
        type=int,
        default=100,
        help='Number of candidates to generate (default: 100)'
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='candidates',
        help='Output directory for candidates (default: candidates)'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='OpenAI model for code generation (default: from config)'
    )
    
    parser.add_argument(
        '--embedding-model',
        type=str, 
        default=None,
        help='OpenAI embedding model (default: from config)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing candidate files'
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.info("Verbose logging enabled")
    
    # Load configuration
    config = load_config()
    
    # Display configuration
    logger.info("🎯 MathCoRL - Candidate Generation")
    logger.info("=" * 50)
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Candidates: {args.n_candidates}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Chat model: {args.model or config['model']}")
    logger.info(f"Embedding model: {args.embedding_model or config['embedding_model']}")
    
    # Check if output file already exists
    output_file = os.path.join(args.output_dir, f"{args.dataset}.json")
    if os.path.exists(output_file) and not args.overwrite:
        logger.error(f"❌ Output file already exists: {output_file}")
        logger.error("Use --overwrite to replace existing file")
        return 1
    
    # Dataset path mapping
    dataset_paths = {
        'GSM8K': 'datasets/GSM8K/train.jsonl',
        'SVAMP': 'datasets/SVAMP/train.json', 
        'TabMWP': 'datasets/TabMWP/train.json',
        'TAT-QA': 'datasets/TAT-QA/train.json',
        'FinQA': 'datasets/FinQA/train.json'
    }
    
    dataset_path = dataset_paths[args.dataset]
    
    # Check if dataset exists
    if not os.path.exists(dataset_path):
        logger.error(f"❌ Dataset file not found: {dataset_path}")
        logger.error("Please ensure dataset files are in the datasets/ directory")
        return 1
    
    try:
        # Initialize candidate generator
        logger.info("🔄 Initializing candidate generator...")
        generator = CandidateGenerator(
            model=args.model,
            embedding_model=args.embedding_model
        )
        
        # Generate candidates
        logger.info(f"🚀 Starting candidate generation for {args.dataset}...")
        summary = generator.generate_candidates(
            dataset_name=args.dataset,
            dataset_path=dataset_path,
            n_candidates=args.n_candidates,
            output_dir=args.output_dir
        )
        
        # Display results
        logger.info("✅ Candidate generation completed!")
        logger.info("=" * 50)
        logger.info(f"📊 SUMMARY - {args.dataset}")
        logger.info(f"Requested: {summary['requested_candidates']}")
        logger.info(f"Generated: {summary['successful_candidates']}")
        logger.info(f"Failed: {summary['failed_generations']}")
        logger.info(f"Success rate: {summary['successful_candidates']/summary['total_attempts']*100:.1f}%")
        logger.info(f"💾 Saved to: {summary['output_path']}")
        
        if summary['type_distribution']:
            logger.info("📈 Type distribution:")
            for type_name, count in summary['type_distribution'].items():
                logger.info(f"  {type_name}: {count}")
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Candidate generation failed: {e}")
        if args.verbose:
            import traceback
            logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    exit(main()) 