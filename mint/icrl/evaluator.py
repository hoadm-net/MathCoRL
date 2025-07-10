import torch
import torch.nn.functional as F
import json
import random
import numpy as np
from typing import Dict, List, Tuple, Any
import logging
from collections import defaultdict
from openai import OpenAI
from ..config import load_config
from ..utils import execute_code, evaluate_result
from mint.prompts import create_policy_evaluation_prompt

logger = logging.getLogger(__name__)


class PolicyNetworkEvaluator:
    """
    Comprehensive evaluation system for Policy Network performance
    
    Measures:
    1. Accuracy comparison: Policy vs Random selection
    2. Selection quality metrics (relevance, diversity, consistency)
    3. Learning curves và progress tracking
    """
    
    def __init__(self, openai_client: OpenAI = None, model: str = None):
        """Initialize evaluator với OpenAI client cho GPT evaluation"""
        self.config = load_config()
        self.openai_client = openai_client or OpenAI(api_key=self.config.get('api_key'))
        self.model = model or self.config.get('model', 'gpt-4o-mini')
        
        # Function prototypes for code generation
        with open('templates/function_prototypes.txt', 'r') as f:
            self.function_prototypes = f.read()
            
        logger.info(f"PolicyNetworkEvaluator initialized with model: {self.model}")

    def gpt_solve_with_examples(self, problem: Dict[str, Any], examples: List[Dict[str, Any]], 
                               dataset_name: str) -> Tuple[bool, float]:
        """
        Use GPT to solve problem với given examples using FPP template và return success + answer
        
        Args:
            problem: Problem dict with context, question, answer
            examples: List of example dicts for few-shot
            dataset_name: Dataset name for specific handling
            
        Returns:
            (success: bool, answer: float)
        """
        try:
            # Get context and question
            context = problem.get('context', '')
            question = problem['question']
            
            # Create prompt using LangChain template
            prompt = create_policy_evaluation_prompt(question, examples, context)

            # Call GPT with tracking
            from ..tracking import track_api_call, count_tokens_openai
            import time
            
            with track_api_call("ICRL-Evaluator", self.model, question, context) as tracker:
                # Estimate input tokens
                input_tokens = count_tokens_openai(prompt, self.model)
                
                start_time = time.time()
                response = self.openai_client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=300
                )
                
                # Extract token counts from response
                usage = response.usage if hasattr(response, 'usage') else None
                if usage:
                    actual_input_tokens = usage.prompt_tokens
                    output_tokens = usage.completion_tokens
                else:
                    actual_input_tokens = input_tokens
                    output_tokens = count_tokens_openai(response.choices[0].message.content, self.model)
                
                tracker.set_tokens(actual_input_tokens, output_tokens)
            
            # Execute generated code
            generated_code = response.choices[0].message.content.strip()
            
            # Clean code (remove markdown blocks if present)
            if "```python" in generated_code:
                generated_code = generated_code.split("```python")[1].split("```")[0].strip()
            elif "```" in generated_code:
                generated_code = generated_code.split("```")[1].split("```")[0].strip()
            
            result, error = execute_code(generated_code)
            
            if error:
                logger.debug(f"Code execution error: {error}")
                return False, 0.0
                
            # Evaluate result
            ground_truth = problem['answer']
            success = evaluate_result(result, ground_truth)
            
            return success, result if result is not None else 0.0
            
        except Exception as e:
            logger.debug(f"GPT solve failed: {e}")
            return False, 0.0

    def select_with_policy(self, policy_net: torch.nn.Module, problem: Dict[str, Any], 
                          candidate_pool: List[Dict[str, Any]], k: int = 2) -> List[Dict[str, Any]]:
        """Select k examples using policy network"""
        try:
            policy_net.eval()
            with torch.no_grad():
                # Convert embeddings to tensors
                problem_emb = torch.tensor(problem['embedding'], dtype=torch.float32).unsqueeze(0)
                candidate_embs = torch.tensor([c['embedding'] for c in candidate_pool], dtype=torch.float32)
                
                # Get probabilities from policy
                probs = policy_net(problem_emb, candidate_embs)
                
                # Sample k examples based on probabilities
                selected_indices = torch.multinomial(probs, k, replacement=False)
                selected_examples = [candidate_pool[i] for i in selected_indices]
                
                return selected_examples
                
        except Exception as e:
            logger.warning(f"Policy selection failed: {e}, falling back to random")
            return random.sample(candidate_pool, k)

    def evaluate_policy_vs_random(self, policy_net: torch.nn.Module, 
                                 dataset_candidates: List[Dict[str, Any]], 
                                 dataset_name: str,
                                 n_trials: int = 50,
                                 pool_size: int = 20,
                                 k: int = 2) -> Dict[str, float]:
        """
        Compare Policy Network vs Random selection accuracy
        
        Args:
            policy_net: Trained policy network
            dataset_candidates: List of candidate examples
            dataset_name: Name of dataset
            n_trials: Number of evaluation trials
            pool_size: Size of candidate pool
            k: Number of examples to select
            
        Returns:
            Dict with accuracy metrics và improvement
        """
        logger.info(f"Evaluating {dataset_name}: Policy vs Random ({n_trials} trials)")
        
        policy_correct = 0
        random_correct = 0
        policy_answers = []
        random_answers = []
        
        for trial in range(n_trials):
            try:
                # Sample problem và candidate pool
                problem = random.choice(dataset_candidates)
                available_candidates = [x for x in dataset_candidates if x != problem]
                
                if len(available_candidates) < pool_size:
                    logger.warning(f"Not enough candidates for trial {trial}, skipping")
                    continue
                    
                candidate_pool = random.sample(available_candidates, pool_size)
                
                # Policy Network selection
                policy_examples = self.select_with_policy(policy_net, problem, candidate_pool, k)
                policy_success, policy_answer = self.gpt_solve_with_examples(problem, policy_examples, dataset_name)
                policy_correct += policy_success
                policy_answers.append(policy_answer)
                
                # Random selection
                random_examples = random.sample(candidate_pool, k)
                random_success, random_answer = self.gpt_solve_with_examples(problem, random_examples, dataset_name)
                random_correct += random_success
                random_answers.append(random_answer)
                
                if (trial + 1) % 10 == 0:
                    logger.info(f"Trial {trial + 1}/{n_trials}: Policy={policy_correct}/{trial+1}, Random={random_correct}/{trial+1}")
                    
            except Exception as e:
                logger.warning(f"Trial {trial} failed: {e}")
                continue
        
        # Calculate metrics
        valid_trials = min(len(policy_answers), len(random_answers))
        if valid_trials == 0:
            return {'error': 'No valid trials completed'}
            
        policy_acc = policy_correct / valid_trials
        random_acc = random_correct / valid_trials
        improvement = ((policy_acc - random_acc) / random_acc * 100) if random_acc > 0 else 0
        
        results = {
            'policy_accuracy': policy_acc,
            'random_accuracy': random_acc,
            'improvement_percent': improvement,
            'valid_trials': valid_trials,
            'policy_wins': policy_correct,
            'random_wins': random_correct
        }
        
        logger.info(f"Evaluation complete: Policy={policy_acc:.3f}, Random={random_acc:.3f}, Improvement={improvement:.1f}%")
        return results

    def analyze_selection_quality(self, policy_net: torch.nn.Module,
                                 dataset_candidates: List[Dict[str, Any]],
                                 n_samples: int = 30,
                                 pool_size: int = 20,
                                 k: int = 2) -> Dict[str, float]:
        """
        Analyze quality of Policy Network selections
        
        Returns:
            Dict với relevance, diversity, và consistency metrics
        """
        logger.info(f"Analyzing selection quality ({n_samples} samples)")
        
        relevance_scores = []
        diversity_scores = []
        consistency_scores = []
        
        for i in range(n_samples):
            try:
                # Sample problem và candidate pool
                problem = random.choice(dataset_candidates)
                available_candidates = [x for x in dataset_candidates if x != problem]
                candidate_pool = random.sample(available_candidates, pool_size)
                
                # Multiple selections for consistency analysis
                selections = []
                for _ in range(5):  # 5 runs for consistency
                    examples = self.select_with_policy(policy_net, problem, candidate_pool, k)
                    selections.append(examples)
                    
                    # Relevance: similarity to problem
                    problem_emb = torch.tensor(problem['embedding'])
                    example_embs = torch.tensor([ex['embedding'] for ex in examples])
                    relevance = F.cosine_similarity(problem_emb.unsqueeze(0), example_embs, dim=1).mean().item()
                    relevance_scores.append(relevance)
                    
                    # Diversity: between selected examples (for k=2)
                    if k == 2:
                        diversity = 1 - F.cosine_similarity(
                            example_embs[0].unsqueeze(0), 
                            example_embs[1].unsqueeze(0)
                        ).item()
                        diversity_scores.append(diversity)
                
                # Consistency: overlap in selections
                consistency = self._measure_selection_consistency(selections)
                consistency_scores.append(consistency)
                
            except Exception as e:
                logger.warning(f"Selection quality analysis failed for sample {i}: {e}")
                continue
        
        return {
            'avg_relevance': np.mean(relevance_scores) if relevance_scores else 0.0,
            'avg_diversity': np.mean(diversity_scores) if diversity_scores else 0.0,
            'avg_consistency': np.mean(consistency_scores) if consistency_scores else 0.0,
            'std_relevance': np.std(relevance_scores) if relevance_scores else 0.0,
            'std_diversity': np.std(diversity_scores) if diversity_scores else 0.0,
            'std_consistency': np.std(consistency_scores) if consistency_scores else 0.0
        }

    def _measure_selection_consistency(self, selections: List[List[Dict[str, Any]]]) -> float:
        """Measure how consistent selections are across multiple runs"""
        if len(selections) < 2:
            return 1.0
            
        # Count frequency of each selected example ID
        selection_counts = defaultdict(int)
        total_selections = 0
        
        for selection in selections:
            for example in selection:
                example_id = example.get('id', str(example))
                selection_counts[example_id] += 1
                total_selections += 1
        
        # Calculate entropy-based consistency (lower entropy = more consistent)
        if total_selections == 0:
            return 0.0
            
        frequencies = [count / total_selections for count in selection_counts.values()]
        entropy = -sum(f * np.log(f + 1e-8) for f in frequencies)
        
        # Normalize to 0-1 scale (1 = completely consistent)
        max_entropy = np.log(len(selection_counts)) if len(selection_counts) > 1 else 0
        consistency = 1 - (entropy / max_entropy) if max_entropy > 0 else 1.0
        
        return consistency

    def comprehensive_evaluation(self, policy_net: torch.nn.Module,
                               dataset_candidates: List[Dict[str, Any]],
                               dataset_name: str,
                               n_trials: int = 50) -> Dict[str, Any]:
        """
        Run comprehensive evaluation combining accuracy và quality metrics
        """
        logger.info(f"Running comprehensive evaluation for {dataset_name}")
        
        # Dataset-specific configurations
        configs = {
            'SVAMP': {'pool_size': 15, 'k': 2},
            'GSM8K': {'pool_size': 20, 'k': 2},
            'TabMWP': {'pool_size': 25, 'k': 3},
            'TAT-QA': {'pool_size': 25, 'k': 3},
            'FinQA': {'pool_size': 30, 'k': 3}
        }
        
        config = configs.get(dataset_name, {'pool_size': 20, 'k': 2})
        
        # Accuracy evaluation
        accuracy_results = self.evaluate_policy_vs_random(
            policy_net, dataset_candidates, dataset_name, 
            n_trials=n_trials, **config
        )
        
        # Quality analysis
        quality_results = self.analyze_selection_quality(
            policy_net, dataset_candidates, 
            n_samples=min(30, len(dataset_candidates)//10), **config
        )
        
        # Combine results
        comprehensive_results = {
            'dataset': dataset_name,
            'accuracy': accuracy_results,
            'quality': quality_results,
            'config': config,
            'n_candidates': len(dataset_candidates)
        }
        
        logger.info(f"Comprehensive evaluation complete for {dataset_name}")
        return comprehensive_results 