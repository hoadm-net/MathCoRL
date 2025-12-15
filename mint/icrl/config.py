"""
Configuration classes for ICRL (In-Context Reinforcement Learning) components.

This module defines configuration dataclasses for policy network training,
including reward weight configurations and hyperparameter settings.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RewardConfig:
    """
    Configuration for multi-objective reward calculation in policy network training.
    
    The policy network optimizes a weighted combination of three reward components:
    - Accuracy: Binary reward based on correctness (0 or 1)
    - Similarity: Semantic relevance of selected examples to target problem
    - Diversity: Variety of approaches in selected examples
    
    Attributes:
        accuracy_weight: Weight for accuracy reward component (λ_acc)
        similarity_weight: Weight for similarity reward component (λ_sim)  
        diversity_weight: Weight for diversity reward component (λ_div)
        
    Default weights (0.6, 0.3, 0.1) balance correctness with example quality.
    
    Examples:
        >>> # Default balanced configuration
        >>> config = RewardConfig()
        >>> 
        >>> # Accuracy-focused configuration
        >>> config = RewardConfig(accuracy_weight=0.9, similarity_weight=0.05, diversity_weight=0.05)
        >>> 
        >>> # Diversity-focused configuration  
        >>> config = RewardConfig(accuracy_weight=0.4, similarity_weight=0.5, diversity_weight=0.1)
    """
    
    accuracy_weight: float = 0.6
    similarity_weight: float = 0.3
    diversity_weight: float = 0.1
    
    def __post_init__(self):
        """Validate reward weights sum to approximately 1.0."""
        total = self.accuracy_weight + self.similarity_weight + self.diversity_weight
        if not (0.99 <= total <= 1.01):
            raise ValueError(
                f"Reward weights must sum to 1.0, got {total:.3f}. "
                f"(accuracy={self.accuracy_weight}, similarity={self.similarity_weight}, "
                f"diversity={self.diversity_weight})"
            )
        
        # Validate individual weights are non-negative
        if self.accuracy_weight < 0 or self.similarity_weight < 0 or self.diversity_weight < 0:
            raise ValueError("All reward weights must be non-negative")
    
    def __repr__(self) -> str:
        """String representation for logging."""
        return (
            f"RewardConfig(λ_acc={self.accuracy_weight:.2f}, "
            f"λ_sim={self.similarity_weight:.2f}, "
            f"λ_div={self.diversity_weight:.2f})"
        )
    
    @classmethod
    def from_string(cls, weights_str: str) -> 'RewardConfig':
        """
        Create RewardConfig from comma-separated string.
        
        Args:
            weights_str: String like "0.6,0.3,0.1" or "0.9,0.05,0.05"
            
        Returns:
            RewardConfig instance
            
        Raises:
            ValueError: If string format is invalid or weights don't sum to 1.0
            
        Examples:
            >>> config = RewardConfig.from_string("0.6,0.3,0.1")
            >>> config = RewardConfig.from_string("0.9, 0.05, 0.05")  # Spaces OK
        """
        try:
            parts = [float(x.strip()) for x in weights_str.split(',')]
            if len(parts) != 3:
                raise ValueError(
                    f"Expected 3 comma-separated values, got {len(parts)}. "
                    f"Format: 'accuracy,similarity,diversity'"
                )
            return cls(
                accuracy_weight=parts[0],
                similarity_weight=parts[1],
                diversity_weight=parts[2]
            )
        except (ValueError, IndexError) as e:
            raise ValueError(
                f"Invalid reward weights string: '{weights_str}'. "
                f"Expected format: '0.6,0.3,0.1'. Error: {str(e)}"
            )
    
    @classmethod
    def accuracy_focused(cls) -> 'RewardConfig':
        """Preset: Maximize correctness (90% accuracy, 5% similarity, 5% diversity)."""
        return cls(accuracy_weight=0.9, similarity_weight=0.05, diversity_weight=0.05)
    
    @classmethod
    def diversity_focused(cls) -> 'RewardConfig':
        """Preset: Emphasize varied examples (40% accuracy, 50% similarity, 10% diversity)."""
        return cls(accuracy_weight=0.4, similarity_weight=0.5, diversity_weight=0.1)
    
    @classmethod
    def balanced(cls) -> 'RewardConfig':
        """Preset: Equal importance (50% accuracy, 25% similarity, 25% diversity)."""
        return cls(accuracy_weight=0.5, similarity_weight=0.25, diversity_weight=0.25)


@dataclass
class TrainingConfig:
    """
    Configuration for policy network training hyperparameters.
    
    Attributes:
        learning_rate: Optimizer learning rate
        weight_decay: L2 regularization coefficient
        max_epochs: Maximum training epochs
        batch_size: Number of samples per training batch
        grad_clip: Gradient clipping threshold (None to disable)
        scheduler_t_max: CosineAnnealingLR scheduler period
        reward_config: Reward weight configuration
    """
    
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    max_epochs: int = 20
    batch_size: int = 32
    grad_clip: Optional[float] = 1.0
    scheduler_t_max: int = 20
    reward_config: RewardConfig = field(default_factory=RewardConfig)
    
    def __repr__(self) -> str:
        """String representation for logging."""
        return (
            f"TrainingConfig(lr={self.learning_rate}, wd={self.weight_decay}, "
            f"epochs={self.max_epochs}, {self.reward_config})"
        )
