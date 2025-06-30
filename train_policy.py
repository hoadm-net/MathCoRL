#!/usr/bin/env python3
"""
MathCoRL - Step 2: Policy Network Training

Train Policy Networks for In-Context Reinforcement Learning using generated candidates.

Usage:
    python train_policy.py --dataset FinQA --epochs 5
    python train_policy.py --dataset GSM8K --epochs 3 --lr 3e-4
    python train_policy.py --help
"""

import argparse
import os
import sys
import logging
import json
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from mint.icrl.trainer import PolicyNetworkTrainer
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
        description='Train Policy Networks for In-Context Reinforcement Learning',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python train_policy.py --dataset FinQA --epochs 5
    python train_policy.py --dataset GSM8K --epochs 3 --lr 3e-4
    python train_policy.py --dataset SVAMP --epochs 4 --samples-per-epoch 50
    python train_policy.py --dataset TabMWP --epochs 6 --verbose
    python train_policy.py --dataset TAT-QA --epochs 5 --save-best

Dataset Configurations:
    - GSM8K: k=2, pool_size=20, default_lr=3e-4
    - SVAMP: k=2, pool_size=15, default_lr=3e-4  
    - TabMWP: k=3, pool_size=25, default_lr=2e-4
    - TAT-QA: k=3, pool_size=25, default_lr=2e-4
    - FinQA: k=3, pool_size=30, default_lr=1e-4
        """
    )
    
    # Required arguments
    parser.add_argument(
        '--dataset', '-d',
        type=str,
        required=True,
        choices=['GSM8K', 'SVAMP', 'TabMWP', 'TAT-QA', 'FinQA'],
        help='Dataset to train policy network on'
    )
    
    # Training parameters
    parser.add_argument(
        '--epochs', '-e',
        type=int,
        default=5,
        help='Number of training epochs (default: 5)'
    )
    
    parser.add_argument(
        '--lr', '--learning-rate',
        type=float,
        default=None,
        help='Learning rate (default: dataset-specific)'
    )
    
    parser.add_argument(
        '--samples-per-epoch',
        type=int,
        default=None,
        help='Samples per epoch (default: use all candidates)'
    )
    
    parser.add_argument(
        '--pool-size',
        type=int,
        default=None,
        help='Candidate pool size for training (default: dataset-specific)'
    )
    
    parser.add_argument(
        '--k',
        type=int,
        default=None,
        help='Number of examples to select (default: dataset-specific)'
    )
    
    # Directories
    parser.add_argument(
        '--candidates-dir',
        type=str,
        default='candidates',
        help='Directory containing candidate files (default: candidates)'
    )
    
    parser.add_argument(
        '--models-dir',
        type=str,
        default='models',
        help='Directory to save trained models (default: models)'
    )
    
    # Options
    parser.add_argument(
        '--save-best',
        action='store_true',
        default=True,
        help='Save best model during training (default: True)'
    )
    
    parser.add_argument(
        '--no-save-best',
        action='store_true',
        help='Disable saving best model'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing model files'
    )
    
    args = parser.parse_args()
    
    # Handle save_best logic
    save_best = args.save_best and not args.no_save_best
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.info("Verbose logging enabled")
    
    # Load configuration
    config = load_config()
    
    # Dataset-specific configurations
    dataset_configs = {
        'GSM8K': {'pool_size': 20, 'k': 2, 'lr': 3e-4},
        'SVAMP': {'pool_size': 15, 'k': 2, 'lr': 3e-4},
        'TabMWP': {'pool_size': 25, 'k': 3, 'lr': 2e-4},
        'TAT-QA': {'pool_size': 25, 'k': 3, 'lr': 2e-4},
        'FinQA': {'pool_size': 30, 'k': 3, 'lr': 1e-4}
    }
    
    dataset_config = dataset_configs[args.dataset]
    
    # Override with command line arguments if provided
    pool_size = args.pool_size or dataset_config['pool_size']
    k = args.k or dataset_config['k']
    lr = args.lr or dataset_config['lr']
    
    # Display configuration
    logger.info("🎓 MathCoRL - Policy Network Training")
    logger.info("=" * 50)
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Learning rate: {lr}")
    logger.info(f"Pool size: {pool_size}")
    logger.info(f"k (examples): {k}")
    logger.info(f"Samples per epoch: {args.samples_per_epoch or 'All'}")
    logger.info(f"Candidates dir: {args.candidates_dir}")
    logger.info(f"Models dir: {args.models_dir}")
    logger.info(f"Save best: {save_best}")
    
    # Check if candidates file exists
    candidates_file = os.path.join(args.candidates_dir, f"{args.dataset}.json")
    if not os.path.exists(candidates_file):
        logger.error(f"❌ Candidates file not found: {candidates_file}")
        logger.error("Please run generate_candidates.py first")
        return 1
    
    # Load and validate candidates
    try:
        with open(candidates_file, 'r', encoding='utf-8') as f:
            candidates = json.load(f)
        
        logger.info(f"📚 Loaded {len(candidates)} candidates")
        
        # Validate candidates have required fields
        required_fields = ['embedding', 'question', 'code']
        missing_fields = []
        for i, candidate in enumerate(candidates[:5]):  # Check first 5
            for field in required_fields:
                if field not in candidate:
                    missing_fields.append(f"candidate[{i}].{field}")
        
        if missing_fields:
            logger.error(f"❌ Invalid candidates file - missing fields: {missing_fields}")
            logger.error("Please regenerate candidates with correct format")
            return 1
            
    except Exception as e:
        logger.error(f"❌ Error loading candidates: {e}")
        return 1
    
    # Check if model already exists
    model_file = os.path.join(args.models_dir, f"{args.dataset}_policy_best.pt")
    if os.path.exists(model_file) and not args.overwrite:
        logger.error(f"❌ Model file already exists: {model_file}")
        logger.error("Use --overwrite to replace existing model")
        return 1
    
    # Ensure directories exist
    os.makedirs(args.models_dir, exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    try:
        # Initialize trainer
        logger.info("🔄 Initializing Policy Network trainer...")
        trainer = PolicyNetworkTrainer(
            dataset_name=args.dataset,
            candidates_dir=args.candidates_dir,
            models_dir=args.models_dir
        )
        
        # Override trainer config with command line arguments
        trainer.config_params.update({
            'pool_size': pool_size,
            'k': k,
            'lr': lr
        })
        
        # Update optimizer with new learning rate
        import torch
        trainer.optimizer = torch.optim.AdamW(
            trainer.policy_net.parameters(),
            lr=lr,
            weight_decay=0.01
        )
        trainer.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            trainer.optimizer, T_max=args.epochs
        )
        
        # Start training
        logger.info(f"🚀 Starting Policy Network training for {args.dataset}...")
        training_history = trainer.train(
            num_epochs=args.epochs,
            samples_per_epoch=args.samples_per_epoch,
            save_best=save_best
        )
        
        # Display results
        logger.info("✅ Policy Network training completed!")
        logger.info("=" * 50)
        logger.info(f"📊 TRAINING SUMMARY - {args.dataset}")
        logger.info(f"Epochs completed: {args.epochs}")
        logger.info(f"Final loss: {training_history['train_loss'][-1]:.4f}")
        logger.info(f"Final reward: {training_history['train_reward'][-1]:.4f}")
        logger.info(f"Final accuracy: {training_history['train_accuracy'][-1]:.4f}")
        
        # Save training history
        history_file = os.path.join('logs', f"{args.dataset}_training_history.json")
        with open(history_file, 'w') as f:
            json.dump(training_history, f, indent=2)
        logger.info(f"📈 Training history saved to: {history_file}")
        
        # Model locations
        best_model = os.path.join(args.models_dir, f"{args.dataset}_policy_best.pt")
        final_model = os.path.join(args.models_dir, f"{args.dataset}_policy_final.pt")
        
        if os.path.exists(best_model):
            logger.info(f"🏆 Best model saved to: {best_model}")
        if os.path.exists(final_model):
            logger.info(f"💾 Final model saved to: {final_model}")
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        if args.verbose:
            import traceback
            logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    exit(main()) 