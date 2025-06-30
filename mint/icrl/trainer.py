import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import json
import os
import random
import numpy as np
from typing import Dict, List, Any, Optional
import logging
from tqdm import tqdm
from collections import defaultdict
from openai import OpenAI

from .policy_network import PolicyNetwork, ppo_loss, contrastive_loss
from .evaluator import PolicyNetworkEvaluator
from ..config import load_config
from ..utils import execute_code, evaluate_result

logger = logging.getLogger(__name__)


class PolicyNetworkTrainer:
    """
    Comprehensive training system for Policy Network
    
    Features:
    - Load candidates from ICRL Step 1
    - PPO training với multi-objective reward
    - Periodic evaluation during training
    - Model saving và loading
    - Dataset-specific configurations
    """
    
    def __init__(self, dataset_name: str, candidates_dir: str = "candidates", 
                 models_dir: str = "models", openai_client: OpenAI = None):
        """
        Initialize trainer for specific dataset
        
        Args:
            dataset_name: Name of dataset (SVAMP, GSM8K, etc.)
            candidates_dir: Directory containing candidate files
            models_dir: Directory to save trained models
            openai_client: OpenAI client for GPT calls
        """
        self.dataset_name = dataset_name
        self.candidates_dir = candidates_dir
        self.models_dir = models_dir
        
        # Configuration
        self.config = load_config()
        self.openai_client = openai_client or OpenAI(api_key=self.config.get('api_key'))
        
        # Dataset-specific configurations
        self.dataset_configs = {
            'SVAMP': {'pool_size': 15, 'k': 2, 'lr': 3e-4},
            'GSM8K': {'pool_size': 20, 'k': 2, 'lr': 3e-4},
            'TabMWP': {'pool_size': 25, 'k': 3, 'lr': 2e-4},
            'TAT-QA': {'pool_size': 25, 'k': 3, 'lr': 2e-4},
            'FinQA': {'pool_size': 30, 'k': 3, 'lr': 1e-4}
        }
        
        self.config_params = self.dataset_configs.get(dataset_name, {
            'pool_size': 20, 'k': 2, 'lr': 3e-4
        })
        
        # Load candidates data
        self.candidates = self.load_candidates()
        
        # Initialize components
        self.policy_net = PolicyNetwork()
        self.evaluator = PolicyNetworkEvaluator(openai_client=self.openai_client)
        
        # Training hyperparameters
        self.optimizer = torch.optim.AdamW(
            self.policy_net.parameters(), 
            lr=self.config_params['lr'], 
            weight_decay=0.01
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=20
        )
        
        # Ensure models directory exists
        os.makedirs(self.models_dir, exist_ok=True)
        
        logger.info(f"PolicyNetworkTrainer initialized for {dataset_name}")
        logger.info(f"Loaded {len(self.candidates)} candidates")
        logger.info(f"Config: {self.config_params}")

    def load_candidates(self) -> List[Dict[str, Any]]:
        """Load candidates from ICRL Step 1"""
        candidates_file = os.path.join(self.candidates_dir, f"{self.dataset_name}.json")
        
        if not os.path.exists(candidates_file):
            raise FileNotFoundError(f"Candidates file not found: {candidates_file}")
        
        with open(candidates_file, 'r', encoding='utf-8') as f:
            candidates = json.load(f)
        
        logger.info(f"Loaded {len(candidates)} candidates from {candidates_file}")
        return candidates

    def calculate_reward(self, is_correct: bool, problem_emb: torch.Tensor, 
                        example_embs: torch.Tensor) -> float:
        """
        Multi-objective reward function theo paper
        
        Args:
            is_correct: Whether GPT solved problem correctly
            problem_emb: Problem embedding tensor
            example_embs: Selected examples embeddings tensor
            
        Returns:
            Combined reward score
        """
        # Accuracy reward (primary component - 60%)
        accuracy_reward = 1.0 if is_correct else 0.0
        
        # Semantic similarity reward (30%)
        similarity_reward = F.cosine_similarity(
            problem_emb, example_embs.mean(dim=0)
        ).item()
        
        # Diversity reward (10%) - between selected examples
        if example_embs.size(0) >= 2:
            diversity_reward = 1.0 - F.cosine_similarity(
                example_embs[0].unsqueeze(0), 
                example_embs[1].unsqueeze(0)
            ).item()
        else:
            diversity_reward = 0.0
        
        # Weighted combination
        total_reward = (0.6 * accuracy_reward + 
                       0.3 * similarity_reward + 
                       0.1 * diversity_reward)
        
        return total_reward

    def train_epoch(self, epoch: int, n_samples: int = None) -> Dict[str, float]:
        """
        Train policy network for one epoch
        
        Args:
            epoch: Current epoch number
            n_samples: Number of training samples (None = use all)
            
        Returns:
            Training metrics for the epoch
        """
        self.policy_net.train()
        
        # Sample training data
        if n_samples is None:
            training_data = self.candidates.copy()
        else:
            training_data = random.sample(self.candidates, min(n_samples, len(self.candidates)))
        
        random.shuffle(training_data)
        
        # Training metrics
        total_loss = 0.0
        total_reward = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        # Training loop
        progress_bar = tqdm(training_data, desc=f"Epoch {epoch}")
        
        for problem in progress_bar:
            try:
                # Create candidate pool (exclude current problem)
                available_candidates = [c for c in self.candidates if c != problem]
                
                if len(available_candidates) < self.config_params['pool_size']:
                    continue
                    
                candidate_pool = random.sample(available_candidates, self.config_params['pool_size'])
                
                # Convert to tensors
                problem_emb = torch.tensor(problem['embedding'], dtype=torch.float32).unsqueeze(0)
                candidate_embs = torch.tensor([c['embedding'] for c in candidate_pool], dtype=torch.float32)
                
                # Get old policy probabilities (for PPO)
                with torch.no_grad():
                    old_probs = self.policy_net(problem_emb, candidate_embs)
                
                # Sample examples using old policy
                old_dist = torch.distributions.Categorical(old_probs)
                chosen_indices = old_dist.sample(sample_shape=(self.config_params['k'],))
                chosen_examples = [candidate_pool[i] for i in chosen_indices]
                
                # Evaluate with GPT
                is_correct, _ = self.evaluator.gpt_solve_with_examples(
                    problem, chosen_examples, self.dataset_name
                )
                
                # Calculate reward
                example_embs = candidate_embs[chosen_indices]
                reward = self.calculate_reward(is_correct, problem_emb, example_embs)
                
                # Forward pass với current policy
                new_probs = self.policy_net(problem_emb, candidate_embs)
                selected_probs = new_probs[chosen_indices]
                old_selected_probs = old_probs[chosen_indices]
                
                # Calculate advantages (simple baseline)
                advantages = reward - new_probs.mean().item()
                advantages_tensor = torch.tensor([advantages] * len(chosen_indices), dtype=torch.float32)
                
                # PPO loss
                ppo_loss_value = ppo_loss(old_selected_probs, selected_probs, advantages_tensor)
                
                # KL divergence regularization
                kl_div = F.kl_div(
                    F.log_softmax(new_probs, dim=-1),
                    F.softmax(old_probs, dim=-1),
                    reduction='batchmean'
                )
                
                # Contrastive loss (if we have both positive và negative examples)
                contrastive_loss_value = torch.tensor(0.0)
                if is_correct:
                    # Use selected examples as positive, random others as negative
                    negative_indices = [i for i in range(len(candidate_pool)) if i not in chosen_indices]
                    if negative_indices:
                        negative_sample_size = min(3, len(negative_indices))
                        negative_indices = random.sample(negative_indices, negative_sample_size)
                        negative_embs = candidate_embs[negative_indices]
                        contrastive_loss_value = contrastive_loss(problem_emb, example_embs, negative_embs)
                
                # Total loss
                total_loss_value = (ppo_loss_value + 
                                  0.01 * kl_div + 
                                  0.1 * contrastive_loss_value)
                
                # Backward pass
                self.optimizer.zero_grad()
                total_loss_value.backward()
                clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                # Update metrics
                total_loss += total_loss_value.item()
                total_reward += reward
                correct_predictions += 1 if is_correct else 0
                total_predictions += 1
                
                # Update progress bar
                progress_bar.set_postfix({
                    'Loss': f"{total_loss_value.item():.4f}",
                    'Reward': f"{reward:.3f}",
                    'Acc': f"{correct_predictions/total_predictions:.3f}"
                })
                
            except Exception as e:
                logger.warning(f"Training step failed: {e}")
                continue
        
        # Calculate epoch metrics
        epoch_metrics = {
            'loss': total_loss / max(total_predictions, 1),
            'reward': total_reward / max(total_predictions, 1),
            'accuracy': correct_predictions / max(total_predictions, 1),
            'total_samples': total_predictions
        }
        
        return epoch_metrics

    def train(self, num_epochs: int = 10, eval_frequency: int = 3, 
             samples_per_epoch: int = None, save_best: bool = True) -> Dict[str, List[float]]:
        """
        Train Policy Network với internal metrics tracking
        
        Args:
            num_epochs: Number of training epochs
            eval_frequency: Ignored (no evaluation during training)
            samples_per_epoch: Number of training samples per epoch
            save_best: Whether to save best model
            
        Returns:
            Training metrics history
        """
        logger.info(f"Starting training for {num_epochs} epochs")
        
        # Tracking metrics
        train_losses = []
        train_rewards = []
        train_accuracies = []
        
        best_reward = -float('inf')
        
        for epoch in range(num_epochs):
            logger.info(f"\n=== Epoch {epoch+1}/{num_epochs} ===")
            
            # Train one epoch
            epoch_metrics = self.train_epoch(epoch, samples_per_epoch)
            
            # Record metrics
            train_losses.append(epoch_metrics['loss'])
            train_rewards.append(epoch_metrics['reward'])
            train_accuracies.append(epoch_metrics['accuracy'])
            
            # Log progress
            logger.info(f"Epoch {epoch+1} - Loss: {epoch_metrics['loss']:.4f}, "
                       f"Reward: {epoch_metrics['reward']:.4f}, "
                       f"Accuracy: {epoch_metrics['accuracy']:.4f}")
            
            # Save best model based on reward
            if save_best and epoch_metrics['reward'] > best_reward:
                best_reward = epoch_metrics['reward']
                self.save_model('best')
                logger.info(f"New best model saved! Reward: {best_reward:.4f}")
            
            # Update learning rate
            self.scheduler.step()
        
        # Save final model
        self.save_model('final')
        logger.info(f"Training completed! Final model saved.")
        
        return {
            'train_loss': train_losses,
            'train_reward': train_rewards, 
            'train_accuracy': train_accuracies
        }

    def save_model(self, suffix: str = 'final'):
        """Save trained model"""
        model_path = os.path.join(self.models_dir, f"{self.dataset_name}_policy_{suffix}.pt")
        
        checkpoint = {
            'model_state_dict': self.policy_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'dataset_name': self.dataset_name,
            'config': self.config_params
        }
        
        torch.save(checkpoint, model_path)
        logger.info(f"Model saved to {model_path}")

    def load_model(self, suffix: str = 'best') -> bool:
        """Load trained model"""
        model_path = os.path.join(self.models_dir, f"{self.dataset_name}_policy_{suffix}.pt")
        
        if not os.path.exists(model_path):
            logger.warning(f"Model file not found: {model_path}")
            return False
        
        try:
            checkpoint = torch.load(model_path, map_location='cpu')
            self.policy_net.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Model loaded from {model_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False

    def evaluate_final_model(self, n_trials: int = 100) -> Dict[str, Any]:
        """Run comprehensive evaluation on trained model"""
        logger.info(f"Running final evaluation với {n_trials} trials...")
        
        # Load best model
        if not self.load_model('best'):
            logger.warning("Could not load best model, using current model")
        
        # Comprehensive evaluation
        results = self.evaluator.comprehensive_evaluation(
            self.policy_net, self.candidates, self.dataset_name, n_trials=n_trials
        )
        
        return results 