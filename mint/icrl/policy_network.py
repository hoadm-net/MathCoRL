import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class PolicyNetwork(nn.Module):
    """
    Enhanced Policy Network for In-Context Example Selection
    
    Based on recent advances in RL for few-shot learning (2024-2025):
    - Multi-head attention for better representation learning
    - Contrastive learning components
    - Adaptive gating mechanisms
    
    Input:
    - problem_emb: embedding vector from OpenAI text-embedding-3-small (1536-D)
    - candidate_embs: tensor containing embeddings of candidate examples (N x 1536)
    
    Output:
    - probs: probability distribution over candidates for selection
    """

    def __init__(self, emb_dim: Optional[int] = None, hidden_dim: Optional[int] = None, 
                 num_heads: Optional[int] = None, dropout: Optional[float] = None):
        super().__init__()
        
        # Load defaults from config if not provided
        from ..config import get_policy_network_config
        config = get_policy_network_config()
        
        self.emb_dim = emb_dim or config.get('emb_dim', 1536)
        self.hidden_dim = hidden_dim or config.get('hidden_dim', 768)
        self.num_heads = num_heads or config.get('num_heads', 8)
        dropout_rate = dropout if dropout is not None else config.get('dropout', 0.1)
        dropout_rate = dropout if dropout is not None else config.get('dropout', 0.1)
        
        # Input projection
        self.input_projection = nn.Linear(self.emb_dim, self.hidden_dim)
        
        # Multi-head attention for better representation learning
        self.attention = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=self.num_heads,
            dropout=dropout_rate,
            batch_first=True
        )
        
        # Layer normalization
        self.layer_norm1 = nn.LayerNorm(self.hidden_dim)
        self.layer_norm2 = nn.LayerNorm(self.hidden_dim)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.Dropout(dropout_rate)
        )
        
        # Output projection for scoring
        self.score_projection = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.GELU(),
            nn.LayerNorm(self.hidden_dim // 2),
            nn.Dropout(dropout_rate),
            nn.Linear(self.hidden_dim // 2, 1)
        )
        
        # Adaptive temperature for softmax
        self.temperature = nn.Parameter(torch.ones(1))
        
        logger.info(f"PolicyNetwork initialized: {emb_dim}-D → {hidden_dim}-D, {num_heads} heads")

    def forward(self, problem_emb, candidate_embs):
        """
        Forward pass: Score each candidate's relevance to the problem
        
        Selection Process (see docs/policy-selection-rules.md for details):
        1. Normalize embeddings to unit sphere
        2. Project to hidden dimension (1536-D → 768-D)
        3. Multi-head attention to capture problem-candidate relationships
        4. Score each candidate via:
           - Interaction score: dot product with problem representation
           - Projection score: learned non-linear scoring function
        5. Apply adaptive temperature and softmax to get probability distribution
        
        Args:
            problem_emb: shape [1, 1536] - target problem embedding
            candidate_embs: shape [N, 1536] - pool of N candidate example embeddings
            
        Returns:
            probs: shape [N] - probability distribution over candidates
                   - Training: sample from this distribution (stochastic)
                   - Inference: take top-k (greedy)
        """
        batch_size = candidate_embs.size(0)
        
        # Step 1: Normalize embeddings to unit sphere
        # Rationale: Makes cosine similarity equivalent to dot product
        problem_emb = F.normalize(problem_emb, p=2, dim=-1)
        candidate_embs = F.normalize(candidate_embs, p=2, dim=-1)
        
        # Step 2: Project to hidden dimension for efficient computation
        problem_h = self.input_projection(problem_emb)  # [1, 768]
        candidate_h = self.input_projection(candidate_embs)  # [N, 768]
        
        # Step 3: Combine for self-attention
        # Attention learns: which candidates are relevant? which pairs work well together?
        combined = torch.cat([problem_h, candidate_h], dim=0)  # [N+1, 768]
        
        # Multi-head attention: 8 heads, each can focus on different aspects
        # (e.g., mathematical concepts, difficulty, solution structure)
        attn_out, attn_weights = self.attention(
            query=combined,
            key=combined, 
            value=combined
        )
        
        # Residual connection + layer norm (standard Transformer technique)
        combined = self.layer_norm1(combined + attn_out)
        
        # Feed-forward network for non-linear transformation
        ffn_out = self.ffn(combined)
        combined = self.layer_norm2(combined + ffn_out)
        
        # Step 4: Extract learned representations
        candidate_repr = combined[1:]  # [N, 768] - candidate representations
        problem_repr = combined[0:1]   # [1, 768] - problem representation
        
        # Step 5: Calculate relevance scores (two components)
        
        # 5a. Interaction score: How well does candidate align with problem?
        interaction_scores = torch.matmul(candidate_repr, problem_repr.T).squeeze(-1)  # [N]
        
        # 5b. Projection score: Additional learned scoring (captures non-linear patterns)
        projected_scores = self.score_projection(candidate_repr).squeeze(-1)  # [N]
        
        # Combine both scoring mechanisms
        final_scores = interaction_scores + projected_scores
        
        # Step 6: Apply adaptive temperature softmax
        # Temperature τ is learned: high τ = more uniform (explore), low τ = sharper (exploit)
        temperature = torch.clamp(self.temperature, min=0.1, max=2.0)
        probs = F.softmax(final_scores / temperature, dim=0)
        
        return probs

    def get_attention_weights(self, problem_emb, candidate_embs):
        """Get attention weights for interpretability"""
        with torch.no_grad():
            problem_emb = F.normalize(problem_emb, p=2, dim=-1)
            candidate_embs = F.normalize(candidate_embs, p=2, dim=-1)
            
            problem_h = self.input_projection(problem_emb)
            candidate_h = self.input_projection(candidate_embs)
            combined = torch.cat([problem_h, candidate_h], dim=0)
            
            _, attn_weights = self.attention(
                query=combined,
                key=combined,
                value=combined
            )
            
            return attn_weights


def contrastive_loss(problem_emb, positive_embs, negative_embs, temperature=0.1):
    """
    Contrastive loss to pull positive examples closer, push negative ones away
    
    Args:
        problem_emb: [1, emb_dim] - problem embedding
        positive_embs: [K, emb_dim] - selected examples (positive)
        negative_embs: [M, emb_dim] - rejected examples (negative) 
        temperature: softmax temperature
        
    Returns:
        loss: contrastive loss value
    """
    try:
        # Calculate similarities
        pos_sim = F.cosine_similarity(
            problem_emb.unsqueeze(1), 
            positive_embs.unsqueeze(0), 
            dim=-1
        ).mean()  # Average over positive examples
        
        neg_sim = F.cosine_similarity(
            problem_emb.unsqueeze(1),
            negative_embs.unsqueeze(0),
            dim=-1
        ).mean()  # Average over negative examples
        
        # Contrastive logits
        logits = torch.stack([pos_sim, neg_sim]) / temperature
        labels = torch.tensor([0], device=logits.device)  # Positive is index 0
        
        loss = F.cross_entropy(logits.unsqueeze(0), labels)
        return loss
        
    except Exception as e:
        logger.warning(f"Contrastive loss calculation failed: {e}")
        return torch.tensor(0.0, requires_grad=True)


def ppo_loss(old_probs, new_probs, advantages, epsilon=0.2):
    """
    Proximal Policy Optimization loss
    
    Args:
        old_probs: probabilities from old policy
        new_probs: probabilities from current policy  
        advantages: advantage values
        epsilon: clipping parameter
        
    Returns:
        ppo_loss: clipped policy loss
    """
    ratio = new_probs / (old_probs.detach() + 1e-8)
    clipped_ratio = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)
    
    policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
    return policy_loss 