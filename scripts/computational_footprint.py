#!/usr/bin/env python3
"""
Computational Footprint Analysis for MathCoRL Policy Network.

Measures and compares computational costs of different exemplar selection methods:
- Policy Network (learned selection)
- Random Selection (baseline)
- Similarity-based (embedding cosine similarity)

Metrics tracked:
- Training time per epoch
- Selection time per problem
- Total overhead
- Memory usage
- GPU utilization

Usage:
    python scripts/computational_footprint.py --dataset GSM8K --epochs 2 --samples 50
    python scripts/computational_footprint.py --compare-methods --output-dir results/footprint
"""

import argparse
import json
import logging
import sys
import time
import psutil
import random
from pathlib import Path
from typing import Dict, List, Any, Tuple
from datetime import datetime
from collections import defaultdict

import torch
import numpy as np
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mint.icrl.policy_network import PolicyNetwork
from mint.icrl.trainer import PolicyNetworkTrainer
from mint.icrl.evaluator import PolicyNetworkEvaluator
from openai import OpenAI

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TimingProfiler:
    """Context manager for precise timing measurements."""
    
    def __init__(self, name: str):
        self.name = name
        self.start_time = None
        self.end_time = None
        
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
        
    def __exit__(self, *args):
        self.end_time = time.perf_counter()
        
    @property
    def elapsed(self) -> float:
        """Get elapsed time in seconds."""
        if self.start_time is None or self.end_time is None:
            return 0.0
        return self.end_time - self.start_time


class ComputationalFootprintAnalyzer:
    """
    Analyzer for computational costs of policy network training and inference.
    """
    
    def __init__(self, dataset_name: str, openai_client: OpenAI = None):
        """
        Initialize analyzer.
        
        Args:
            dataset_name: Name of dataset
            openai_client: OpenAI client for evaluation
        """
        self.dataset_name = dataset_name
        self.openai_client = openai_client or OpenAI()
        
        # Load candidates
        candidates_path = Path('candidates') / f'{dataset_name}.json'
        with open(candidates_path, 'r') as f:
            self.candidates = json.load(f)
        
        # Initialize policy network
        self.policy_net = PolicyNetwork()
        self.evaluator = PolicyNetworkEvaluator(openai_client=self.openai_client)
        
        # Configuration
        self.config_params = {
            'pool_size': 20,
            'k': 2
        }
        
        # Process tracking
        self.process = psutil.Process()
        
        logger.info(f"ComputationalFootprintAnalyzer initialized for {dataset_name}")
        logger.info(f"Loaded {len(self.candidates)} candidates")
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage in MB."""
        mem_info = self.process.memory_info()
        return {
            'rss': mem_info.rss / 1024 / 1024,  # Resident Set Size
            'vms': mem_info.vms / 1024 / 1024   # Virtual Memory Size
        }
    
    def get_gpu_memory(self) -> Dict[str, float]:
        """Get GPU memory usage if available."""
        if torch.cuda.is_available():
            return {
                'allocated': torch.cuda.memory_allocated() / 1024 / 1024,
                'cached': torch.cuda.memory_reserved() / 1024 / 1024
            }
        return {'allocated': 0, 'cached': 0}
    
    def policy_selection(self, problem: Dict, candidate_pool: List[Dict]) -> Tuple[List[Dict], Dict[str, float]]:
        """
        Select examples using policy network.
        
        Returns:
            Tuple of (selected_examples, timing_metrics)
        """
        timing = {}
        
        # Convert to tensors
        with TimingProfiler("tensor_conversion") as timer:
            problem_emb = torch.tensor(problem['embedding'], dtype=torch.float32).unsqueeze(0)
            candidate_embs = torch.tensor([c['embedding'] for c in candidate_pool], dtype=torch.float32)
        timing['tensor_conversion'] = timer.elapsed
        
        # Forward pass
        with TimingProfiler("forward_pass") as timer:
            with torch.no_grad():
                probs = self.policy_net(problem_emb, candidate_embs)
        timing['forward_pass'] = timer.elapsed
        
        # Sampling
        with TimingProfiler("sampling") as timer:
            top_k_indices = torch.topk(probs, k=self.config_params['k']).indices
            if top_k_indices.dim() > 1:
                top_k_indices = top_k_indices[0]
            chosen_examples = [candidate_pool[i] for i in top_k_indices]
        timing['sampling'] = timer.elapsed
        
        timing['total'] = sum(timing.values())
        
        return chosen_examples, timing
    
    def random_selection(self, problem: Dict, candidate_pool: List[Dict]) -> Tuple[List[Dict], Dict[str, float]]:
        """
        Select examples randomly (baseline).
        
        Returns:
            Tuple of (selected_examples, timing_metrics)
        """
        with TimingProfiler("random_sampling") as timer:
            chosen_examples = random.sample(candidate_pool, self.config_params['k'])
        
        return chosen_examples, {'total': timer.elapsed, 'random_sampling': timer.elapsed}
    
    def similarity_selection(self, problem: Dict, candidate_pool: List[Dict]) -> Tuple[List[Dict], Dict[str, float]]:
        """
        Select examples by embedding similarity.
        
        Returns:
            Tuple of (selected_examples, timing_metrics)
        """
        timing = {}
        
        # Convert to tensors
        with TimingProfiler("tensor_conversion") as timer:
            problem_emb = torch.tensor(problem['embedding'], dtype=torch.float32)
            candidate_embs = torch.tensor([c['embedding'] for c in candidate_pool], dtype=torch.float32)
        timing['tensor_conversion'] = timer.elapsed
        
        # Calculate similarities
        with TimingProfiler("similarity_computation") as timer:
            similarities = torch.cosine_similarity(
                problem_emb.unsqueeze(0), 
                candidate_embs, 
                dim=1
            )
        timing['similarity_computation'] = timer.elapsed
        
        # Select top-k
        with TimingProfiler("top_k_selection") as timer:
            top_k_indices = torch.topk(similarities, k=self.config_params['k']).indices
            chosen_examples = [candidate_pool[i] for i in top_k_indices]
        timing['top_k_selection'] = timer.elapsed
        
        timing['total'] = sum(timing.values())
        
        return chosen_examples, timing
    
    def measure_selection_overhead(self, n_samples: int = 50) -> Dict[str, Any]:
        """
        Measure selection overhead for different methods.
        
        Args:
            n_samples: Number of problems to test
            
        Returns:
            Dictionary with timing statistics
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"SELECTION OVERHEAD ANALYSIS - {self.dataset_name}")
        logger.info(f"{'='*80}")
        logger.info(f"Testing {n_samples} samples with {self.config_params['k']} examples per problem")
        
        # Sample test problems
        test_problems = random.sample(self.candidates, min(n_samples, len(self.candidates)))
        
        # Storage for results
        results = {
            'policy': [],
            'random': [],
            'similarity': []
        }
        
        # Test each method
        for method_name, method_func in [
            ('policy', self.policy_selection),
            ('random', self.random_selection),
            ('similarity', self.similarity_selection)
        ]:
            logger.info(f"\nTesting {method_name} selection...")
            
            method_timings = []
            
            for problem in tqdm(test_problems, desc=f"{method_name} selection"):
                # Create candidate pool
                available_candidates = [c for c in self.candidates if c != problem]
                candidate_pool = random.sample(available_candidates, self.config_params['pool_size'])
                
                # Measure selection time
                _, timing = method_func(problem, candidate_pool)
                method_timings.append(timing)
            
            results[method_name] = method_timings
        
        # Calculate statistics
        stats = {}
        for method_name, timings in results.items():
            total_times = [t['total'] for t in timings]
            stats[method_name] = {
                'mean': np.mean(total_times) * 1000,  # Convert to ms
                'std': np.std(total_times) * 1000,
                'min': np.min(total_times) * 1000,
                'max': np.max(total_times) * 1000,
                'median': np.median(total_times) * 1000,
                'total': np.sum(total_times),
                'per_problem': np.mean(total_times)
            }
        
        # Print results
        logger.info(f"\n{'='*80}")
        logger.info("SELECTION OVERHEAD COMPARISON")
        logger.info(f"{'='*80}")
        logger.info(f"{'Method':<15} {'Mean (ms)':<12} {'Std (ms)':<12} {'Min (ms)':<12} {'Max (ms)':<12}")
        logger.info(f"{'-'*80}")
        
        for method_name, stat in stats.items():
            logger.info(
                f"{method_name.capitalize():<15} "
                f"{stat['mean']:>10.2f}   "
                f"{stat['std']:>10.2f}   "
                f"{stat['min']:>10.2f}   "
                f"{stat['max']:>10.2f}"
            )
        
        # Calculate overhead ratios
        baseline_time = stats['random']['mean']
        logger.info(f"\n{'='*80}")
        logger.info("OVERHEAD RATIOS (vs Random baseline)")
        logger.info(f"{'='*80}")
        
        for method_name, stat in stats.items():
            if method_name != 'random':
                overhead_ratio = stat['mean'] / baseline_time
                overhead_ms = stat['mean'] - baseline_time
                logger.info(f"{method_name.capitalize()}: {overhead_ratio:.2f}x slower (+{overhead_ms:.2f} ms per problem)")
        
        return {
            'dataset': self.dataset_name,
            'n_samples': n_samples,
            'statistics': stats,
            'raw_results': results
        }
    
    def measure_training_footprint(self, n_epochs: int = 2, n_samples: int = 50) -> Dict[str, Any]:
        """
        Measure computational footprint of policy network training.
        
        Args:
            n_epochs: Number of training epochs
            n_samples: Number of samples per epoch
            
        Returns:
            Dictionary with training metrics
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"TRAINING FOOTPRINT ANALYSIS - {self.dataset_name}")
        logger.info(f"{'='*80}")
        logger.info(f"Training for {n_epochs} epochs with {n_samples} samples per epoch")
        
        # Initialize trainer
        trainer = PolicyNetworkTrainer(
            self.dataset_name,
            openai_client=self.openai_client
        )
        
        # Training metrics
        epoch_metrics = []
        
        # Measure baseline memory
        initial_memory = self.get_memory_usage()
        initial_gpu = self.get_gpu_memory()
        
        logger.info(f"Initial memory: {initial_memory['rss']:.2f} MB RSS")
        logger.info(f"Initial GPU memory: {initial_gpu['allocated']:.2f} MB")
        
        # Training loop
        for epoch in range(1, n_epochs + 1):
            logger.info(f"\nEpoch {epoch}/{n_epochs}")
            
            epoch_start = time.perf_counter()
            
            # Train one epoch
            epoch_result = trainer.train_epoch(epoch, n_samples=n_samples)
            
            epoch_end = time.perf_counter()
            epoch_time = epoch_end - epoch_start
            
            # Measure memory after epoch
            epoch_memory = self.get_memory_usage()
            epoch_gpu = self.get_gpu_memory()
            
            # Collect metrics
            metrics = {
                'epoch': epoch,
                'epoch_time': epoch_time,
                'loss': epoch_result.get('loss', 0),
                'reward': epoch_result.get('reward', 0),
                'accuracy': epoch_result.get('accuracy', 0),
                'memory_rss': epoch_memory['rss'],
                'memory_vms': epoch_memory['vms'],
                'gpu_allocated': epoch_gpu['allocated'],
                'gpu_cached': epoch_gpu['cached'],
                'time_per_sample': epoch_time / n_samples
            }
            
            epoch_metrics.append(metrics)
            
            logger.info(f"Epoch time: {epoch_time:.2f}s ({metrics['time_per_sample']*1000:.2f} ms/sample)")
            logger.info(f"Memory: {epoch_memory['rss']:.2f} MB RSS")
            logger.info(f"Loss: {metrics['loss']:.4f}, Reward: {metrics['reward']:.4f}, Accuracy: {metrics['accuracy']:.2%}")
        
        # Calculate summary statistics
        total_time = sum(m['epoch_time'] for m in epoch_metrics)
        avg_epoch_time = np.mean([m['epoch_time'] for m in epoch_metrics])
        avg_time_per_sample = np.mean([m['time_per_sample'] for m in epoch_metrics])
        
        logger.info(f"\n{'='*80}")
        logger.info("TRAINING SUMMARY")
        logger.info(f"{'='*80}")
        logger.info(f"Total training time: {total_time:.2f}s")
        logger.info(f"Average epoch time: {avg_epoch_time:.2f}s")
        logger.info(f"Average time per sample: {avg_time_per_sample*1000:.2f} ms")
        logger.info(f"Peak memory: {max(m['memory_rss'] for m in epoch_metrics):.2f} MB")
        
        return {
            'dataset': self.dataset_name,
            'n_epochs': n_epochs,
            'n_samples': n_samples,
            'total_time': total_time,
            'avg_epoch_time': avg_epoch_time,
            'avg_time_per_sample': avg_time_per_sample,
            'initial_memory': initial_memory,
            'epoch_metrics': epoch_metrics
        }


def main():
    parser = argparse.ArgumentParser(description='Computational footprint analysis for MathCoRL')
    
    parser.add_argument('--dataset', type=str, default='GSM8K',
                       help='Dataset to analyze (default: GSM8K)')
    parser.add_argument('--epochs', type=int, default=2,
                       help='Number of training epochs (default: 2)')
    parser.add_argument('--samples', type=int, default=50,
                       help='Number of samples per epoch (default: 50)')
    parser.add_argument('--compare-methods', action='store_true',
                       help='Compare different selection methods')
    parser.add_argument('--output-dir', type=str, default='results/footprint',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize analyzer
    analyzer = ComputationalFootprintAnalyzer(args.dataset)
    
    # Run analyses
    all_results = {
        'dataset': args.dataset,
        'timestamp': datetime.now().isoformat(),
        'config': {
            'epochs': args.epochs,
            'samples': args.samples
        }
    }
    
    if args.compare_methods:
        # Selection overhead analysis
        selection_results = analyzer.measure_selection_overhead(n_samples=args.samples)
        all_results['selection_overhead'] = selection_results
        
        # Save selection results
        selection_file = output_dir / f'{args.dataset}_selection_overhead.json'
        with open(selection_file, 'w') as f:
            json.dump(selection_results, f, indent=2)
        logger.info(f"\n✅ Selection overhead results saved to {selection_file}")
    
    # Training footprint analysis
    training_results = analyzer.measure_training_footprint(
        n_epochs=args.epochs,
        n_samples=args.samples
    )
    all_results['training_footprint'] = training_results
    
    # Save training results
    training_file = output_dir / f'{args.dataset}_training_footprint.json'
    with open(training_file, 'w') as f:
        json.dump(training_results, f, indent=2)
    logger.info(f"\n✅ Training footprint results saved to {training_file}")
    
    # Save combined results
    combined_file = output_dir / f'{args.dataset}_footprint_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(combined_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"✅ Combined results saved to {combined_file}")
    
    logger.info(f"\n{'='*80}")
    logger.info("COMPUTATIONAL FOOTPRINT ANALYSIS COMPLETE")
    logger.info(f"{'='*80}")


if __name__ == '__main__':
    main()
