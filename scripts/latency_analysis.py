#!/usr/bin/env python3
"""
Latency Analysis for MathCoRL

Measure and analyze end-to-end latency across different methods:
- Zero-shot baseline
- Random selection
- Policy network selection
- Similarity-based selection (KATE)
- Cluster-based selection (CDS)

Metrics:
- Selection overhead: Time to select examples
- Generation time: LLM inference time
- Total latency: End-to-end time per problem

Usage:
    python scripts/latency_analysis.py --dataset GSM8K --samples 50
    python scripts/latency_analysis.py --dataset SVAMP --all-methods
    python scripts/latency_analysis.py --dataset GSM8K --method policy --samples 100
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any
from datetime import datetime
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mint.core import FunctionPrototypePrompting
from mint.icrl.policy_network import PolicyNetwork
from mint.utils import execute_code, evaluate_result
import torch

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class LatencyMetrics:
    """Latency metrics for a single inference"""
    method: str
    selection_time: float  # Time to select examples (0 for zero-shot)
    generation_time: float  # LLM inference time
    execution_time: float  # Code execution time
    total_time: float  # End-to-end time
    success: bool
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AggregatedMetrics:
    """Aggregated latency statistics"""
    method: str
    num_samples: int
    selection_mean: float
    selection_std: float
    generation_mean: float
    generation_std: float
    execution_mean: float
    execution_std: float
    total_mean: float
    total_std: float
    success_rate: float
    percentiles: Dict[str, float]  # p50, p90, p95, p99
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class LatencyAnalyzer:
    """Analyze latency across different selection methods"""
    
    def __init__(self, dataset_name: str, candidates_dir: str = "candidates", 
                 models_dir: str = "models"):
        """
        Initialize latency analyzer
        
        Args:
            dataset_name: Dataset name (GSM8K, SVAMP, etc.)
            candidates_dir: Directory with candidate files
            models_dir: Directory with trained policy models
        """
        self.dataset_name = dataset_name
        self.candidates_dir = candidates_dir
        self.models_dir = models_dir
        
        # Load candidates
        candidates_file = Path(candidates_dir) / f"{dataset_name}.json"
        if not candidates_file.exists():
            raise FileNotFoundError(f"Candidates file not found: {candidates_file}")
        
        with open(candidates_file, 'r', encoding='utf-8') as f:
            self.candidates = json.load(f)
        
        logger.info(f"Loaded {len(self.candidates)} candidates for {dataset_name}")
        
        # Initialize FPP solver
        self.solver = FunctionPrototypePrompting()
        
        # Load policy network if exists
        self.policy_net = None
        self.policy_available = self._load_policy_network()
    
    def _load_policy_network(self) -> bool:
        """Load policy network if available"""
        model_path = Path(self.models_dir) / f"{self.dataset_name}_policy_best.pt"
        if not model_path.exists():
            logger.warning(f"Policy model not found: {model_path}")
            return False
        
        try:
            self.policy_net = PolicyNetwork(input_dim=1536, hidden_dim=256, output_dim=128)
            state_dict = torch.load(model_path, map_location='cpu')
            self.policy_net.load_state_dict(state_dict)
            self.policy_net.eval()
            logger.info(f"Loaded policy network from {model_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load policy network: {e}")
            return False
    
    def measure_zero_shot(self, problem: Dict[str, Any]) -> LatencyMetrics:
        """
        Measure zero-shot baseline (no example selection)
        
        Args:
            problem: Problem dictionary with question and answer
            
        Returns:
            LatencyMetrics object
        """
        question = problem.get('question', problem.get('original_question', ''))
        expected_answer = problem.get('answer', problem.get('final_answer', ''))
        
        start_total = time.perf_counter()
        
        # No selection overhead for zero-shot
        selection_time = 0.0
        
        # Generation time
        start_gen = time.perf_counter()
        try:
            result_detail = self.solver.solve_detailed(question, context="")
            generation_time = time.perf_counter() - start_gen
            
            # Execution already done in solve_detailed
            result = result_detail.get('result')
            execution_time = 0.0  # Already included in generation_time
            
            # Evaluate
            success = evaluate_result(result, expected_answer)
            error = result_detail.get('error') or None
            
        except Exception as e:
            generation_time = time.perf_counter() - start_gen
            execution_time = 0.0
            success = False
            error = str(e)
        
        total_time = time.perf_counter() - start_total
        
        return LatencyMetrics(
            method="zero_shot",
            selection_time=selection_time,
            generation_time=generation_time,
            execution_time=execution_time,
            total_time=total_time,
            success=success,
            error=error
        )
    
    def measure_random_selection(self, problem: Dict[str, Any], k: int = 3) -> LatencyMetrics:
        """
        Measure random example selection
        
        Args:
            problem: Problem dictionary
            k: Number of examples to select
            
        Returns:
            LatencyMetrics object
        """
        question = problem.get('question', problem.get('original_question', ''))
        expected_answer = problem.get('answer', problem.get('final_answer', ''))
        
        start_total = time.perf_counter()
        
        # Selection time (random sampling)
        start_sel = time.perf_counter()
        selected_indices = np.random.choice(len(self.candidates), size=min(k, len(self.candidates)), replace=False)
        selected_examples = [self.candidates[i] for i in selected_indices]
        selection_time = time.perf_counter() - start_sel
        
        # Build context from selected examples
        context = self._build_context(selected_examples)
        
        # Generation time
        start_gen = time.perf_counter()
        try:
            result_detail = self.solver.solve_detailed(question, context=context)
            generation_time = time.perf_counter() - start_gen
            
            # Execution already done in solve_detailed
            result = result_detail.get('result')
            execution_time = 0.0  # Already included in generation_time
            
            success = evaluate_result(result, expected_answer)
            error = result_detail.get('error') or None
            
        except Exception as e:
            generation_time = time.perf_counter() - start_gen
            execution_time = 0.0
            success = False
            error = str(e)
        
        total_time = time.perf_counter() - start_total
        
        return LatencyMetrics(
            method="random",
            selection_time=selection_time,
            generation_time=generation_time,
            execution_time=execution_time,
            total_time=total_time,
            success=success,
            error=error
        )
    
    def measure_policy_selection(self, problem: Dict[str, Any], k: int = 3) -> LatencyMetrics:
        """
        Measure policy network selection
        
        Args:
            problem: Problem dictionary
            k: Number of examples to select
            
        Returns:
            LatencyMetrics object
        """
        if not self.policy_available:
            raise ValueError("Policy network not available. Train model first.")
        
        question = problem.get('question', problem.get('original_question', ''))
        expected_answer = problem.get('answer', problem.get('final_answer', ''))
        
        # Get question embedding (assume stored in problem or compute)
        question_embedding = problem.get('embedding')
        if question_embedding is None:
            # Compute embedding if not available
            from openai import OpenAI
            client = OpenAI()
            response = client.embeddings.create(
                input=question,
                model="text-embedding-3-small"
            )
            question_embedding = response.data[0].embedding
        
        start_total = time.perf_counter()
        
        # Selection time (policy network inference)
        start_sel = time.perf_counter()
        with torch.no_grad():
            query_tensor = torch.tensor([question_embedding], dtype=torch.float32)
            
            # Get all candidate embeddings
            candidate_embeddings = torch.tensor(
                [c['embedding'] for c in self.candidates],
                dtype=torch.float32
            )
            
            # Policy network scores
            _, scores = self.policy_net(query_tensor, candidate_embeddings)
            
            # Select top-k
            top_k_indices = torch.topk(scores[0], k=min(k, len(self.candidates)))[1]
            selected_examples = [self.candidates[idx] for idx in top_k_indices.tolist()]
        
        selection_time = time.perf_counter() - start_sel
        
        # Build context
        context = self._build_context(selected_examples)
        
        # Generation time
        start_gen = time.perf_counter()
        try:
            result_detail = self.solver.solve_detailed(question, context=context)
            generation_time = time.perf_counter() - start_gen
            
            # Execution already done in solve_detailed
            result = result_detail.get('result')
            execution_time = 0.0  # Already included in generation_time
            
            success = evaluate_result(result, expected_answer)
            error = result_detail.get('error') or None
            
        except Exception as e:
            generation_time = time.perf_counter() - start_gen
            execution_time = 0.0
            success = False
            error = str(e)
        
        total_time = time.perf_counter() - start_total
        
        return LatencyMetrics(
            method="policy",
            selection_time=selection_time,
            generation_time=generation_time,
            execution_time=execution_time,
            total_time=total_time,
            success=success,
            error=error
        )
    
    def measure_similarity_selection(self, problem: Dict[str, Any], k: int = 3) -> LatencyMetrics:
        """
        Measure similarity-based selection (KATE approach)
        
        Args:
            problem: Problem dictionary
            k: Number of examples to select
            
        Returns:
            LatencyMetrics object
        """
        question = problem.get('question', problem.get('original_question', ''))
        expected_answer = problem.get('answer', problem.get('final_answer', ''))
        
        # Get question embedding
        question_embedding = problem.get('embedding')
        if question_embedding is None:
            from openai import OpenAI
            client = OpenAI()
            response = client.embeddings.create(
                input=question,
                model="text-embedding-3-small"
            )
            question_embedding = response.data[0].embedding
        
        start_total = time.perf_counter()
        
        # Selection time (cosine similarity computation)
        start_sel = time.perf_counter()
        query_vec = np.array(question_embedding)
        candidate_vecs = np.array([c['embedding'] for c in self.candidates])
        
        # Cosine similarity
        similarities = np.dot(candidate_vecs, query_vec) / (
            np.linalg.norm(candidate_vecs, axis=1) * np.linalg.norm(query_vec)
        )
        
        # Select top-k most similar
        top_k_indices = np.argsort(similarities)[-k:][::-1]
        selected_examples = [self.candidates[idx] for idx in top_k_indices]
        
        selection_time = time.perf_counter() - start_sel
        
        # Build context
        context = self._build_context(selected_examples)
        
        # Generation time
        start_gen = time.perf_counter()
        try:
            result_detail = self.solver.solve_detailed(question, context=context)
            generation_time = time.perf_counter() - start_gen
            
            # Execution already done in solve_detailed
            result = result_detail.get('result')
            execution_time = 0.0  # Already included in generation_time
            
            success = evaluate_result(result, expected_answer)
            error = result_detail.get('error') or None
            
        except Exception as e:
            generation_time = time.perf_counter() - start_gen
            execution_time = 0.0
            success = False
            error = str(e)
        
        total_time = time.perf_counter() - start_total
        
        return LatencyMetrics(
            method="similarity",
            selection_time=selection_time,
            generation_time=generation_time,
            execution_time=execution_time,
            total_time=total_time,
            success=success,
            error=error
        )
    
    def _build_context(self, examples: List[Dict[str, Any]]) -> str:
        """Build context string from selected examples"""
        context_parts = []
        for i, example in enumerate(examples, 1):
            question = example.get('question', example.get('original_question', ''))
            code = example.get('code', '')
            context_parts.append(f"Example {i}:\nQuestion: {question}\nCode:\n{code}\n")
        return "\n".join(context_parts)
    
    def aggregate_metrics(self, metrics_list: List[LatencyMetrics]) -> AggregatedMetrics:
        """
        Compute aggregated statistics from individual measurements
        
        Args:
            metrics_list: List of LatencyMetrics
            
        Returns:
            AggregatedMetrics with mean, std, percentiles
        """
        if not metrics_list:
            raise ValueError("Empty metrics list")
        
        method = metrics_list[0].method
        
        selection_times = [m.selection_time for m in metrics_list]
        generation_times = [m.generation_time for m in metrics_list]
        execution_times = [m.execution_time for m in metrics_list]
        total_times = [m.total_time for m in metrics_list]
        successes = [m.success for m in metrics_list]
        
        # Compute percentiles for total time
        percentiles = {
            'p50': float(np.percentile(total_times, 50)),
            'p90': float(np.percentile(total_times, 90)),
            'p95': float(np.percentile(total_times, 95)),
            'p99': float(np.percentile(total_times, 99))
        }
        
        return AggregatedMetrics(
            method=method,
            num_samples=len(metrics_list),
            selection_mean=float(np.mean(selection_times)),
            selection_std=float(np.std(selection_times)),
            generation_mean=float(np.mean(generation_times)),
            generation_std=float(np.std(generation_times)),
            execution_mean=float(np.mean(execution_times)),
            execution_std=float(np.std(execution_times)),
            total_mean=float(np.mean(total_times)),
            total_std=float(np.std(total_times)),
            success_rate=float(np.mean(successes)),
            percentiles=percentiles
        )
    
    def run_analysis(self, method: str, num_samples: int, k: int = 3) -> List[LatencyMetrics]:
        """
        Run latency analysis for a specific method
        
        Args:
            method: Method name (zero_shot, random, policy, similarity)
            num_samples: Number of problems to test
            k: Number of examples for in-context learning (ignored for zero_shot)
            
        Returns:
            List of LatencyMetrics
        """
        logger.info(f"Running {method} analysis on {num_samples} samples...")
        
        # Sample problems
        sample_problems = np.random.choice(self.candidates, size=min(num_samples, len(self.candidates)), replace=False)
        
        metrics_list = []
        method_func = {
            'zero_shot': self.measure_zero_shot,
            'random': lambda p: self.measure_random_selection(p, k=k),
            'policy': lambda p: self.measure_policy_selection(p, k=k),
            'similarity': lambda p: self.measure_similarity_selection(p, k=k)
        }.get(method)
        
        if method_func is None:
            raise ValueError(f"Unknown method: {method}")
        
        for i, problem in enumerate(sample_problems, 1):
            try:
                metrics = method_func(problem)
                metrics_list.append(metrics)
                
                if i % 10 == 0:
                    logger.info(f"Processed {i}/{num_samples} samples")
                    
            except Exception as e:
                logger.error(f"Error processing sample {i}: {e}")
                continue
        
        logger.info(f"Completed {method} analysis: {len(metrics_list)}/{num_samples} successful")
        return metrics_list
    
    def save_results(self, results: Dict[str, Any], output_dir: str = "results/latency"):
        """Save analysis results to JSON"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = output_path / f"{self.dataset_name}_latency_{timestamp}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to {filename}")
        return filename


def main():
    parser = argparse.ArgumentParser(
        description='Latency Analysis for MathCoRL',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--dataset', '-d', type=str, required=True,
                       choices=['GSM8K', 'SVAMP', 'TabMWP', 'TAT-QA', 'FinQA'],
                       help='Dataset to analyze')
    
    parser.add_argument('--method', '-m', type=str, default='all',
                       choices=['zero_shot', 'random', 'policy', 'similarity', 'all'],
                       help='Selection method to analyze')
    
    parser.add_argument('--samples', '-n', type=int, default=50,
                       help='Number of samples to test (default: 50)')
    
    parser.add_argument('--k', type=int, default=3,
                       help='Number of examples for ICL (default: 3)')
    
    parser.add_argument('--candidates-dir', type=str, default='candidates',
                       help='Directory with candidate files')
    
    parser.add_argument('--models-dir', type=str, default='models',
                       help='Directory with policy models')
    
    parser.add_argument('--output', '-o', type=str, default='results/latency',
                       help='Output directory for results')
    
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Set seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    logger.info(f"Latency Analysis - {args.dataset}")
    logger.info(f"Samples: {args.samples}, k={args.k}, seed={args.seed}")
    
    # Initialize analyzer
    analyzer = LatencyAnalyzer(
        dataset_name=args.dataset,
        candidates_dir=args.candidates_dir,
        models_dir=args.models_dir
    )
    
    # Determine methods to run
    methods = ['zero_shot', 'random', 'similarity']
    if analyzer.policy_available:
        methods.append('policy')
    
    if args.method != 'all':
        methods = [args.method]
    
    # Run analysis
    all_results = {
        'dataset': args.dataset,
        'num_samples': args.samples,
        'k': args.k,
        'seed': args.seed,
        'timestamp': datetime.now().isoformat(),
        'methods': {}
    }
    
    for method in methods:
        try:
            logger.info(f"\n{'='*60}")
            logger.info(f"Analyzing method: {method}")
            logger.info(f"{'='*60}")
            
            metrics_list = analyzer.run_analysis(method, args.samples, k=args.k)
            aggregated = analyzer.aggregate_metrics(metrics_list)
            
            all_results['methods'][method] = {
                'aggregated': aggregated.to_dict(),
                'individual': [m.to_dict() for m in metrics_list]
            }
            
            # Print summary
            logger.info(f"\n{method.upper()} Results:")
            logger.info(f"  Selection time: {aggregated.selection_mean*1000:.2f}ms ± {aggregated.selection_std*1000:.2f}ms")
            logger.info(f"  Generation time: {aggregated.generation_mean:.3f}s ± {aggregated.generation_std:.3f}s")
            logger.info(f"  Execution time: {aggregated.execution_mean*1000:.2f}ms ± {aggregated.execution_std*1000:.2f}ms")
            logger.info(f"  Total time: {aggregated.total_mean:.3f}s ± {aggregated.total_std:.3f}s")
            logger.info(f"  Success rate: {aggregated.success_rate*100:.1f}%")
            logger.info(f"  Latency percentiles: p50={aggregated.percentiles['p50']:.3f}s, p95={aggregated.percentiles['p95']:.3f}s")
            
        except Exception as e:
            logger.error(f"Failed to analyze {method}: {e}")
            continue
    
    # Save results
    output_file = analyzer.save_results(all_results, args.output)
    logger.info(f"\n{'='*60}")
    logger.info(f"Analysis complete! Results saved to: {output_file}")
    logger.info(f"{'='*60}")


if __name__ == '__main__':
    main()
