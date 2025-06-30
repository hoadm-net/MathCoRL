import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class PolicyNetwork(nn.Module):
    """
    Enhanced Policy Network for In-Context Example Selection
    
    Based on recent advances in RL for few-shot learning (2024-2025):
    - Multi-head attention for better representation learning
    - Contrastive learning components
    - Adaptive gating mechanisms
    
    Input:
    - problem_emb: embedding vector từ OpenAI text-embedding-3-small (1536-D)
    - candidate_embs: tensor chứa embedding của các ví dụ candidate (N x 1536)
    
    Output:
    - probs: probability distribution over candidates for selection
    """

    def __init__(self, emb_dim=1536, hidden_dim=768, num_heads=8, dropout=0.1):
        super().__init__()
        
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # Input projection
        self.input_projection = nn.Linear(emb_dim, hidden_dim)
        
        # Multi-head attention for better representation learning
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer normalization
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout)
        )
        
        # Output projection for scoring
        self.score_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.LayerNorm(hidden_dim // 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Adaptive temperature for softmax
        self.temperature = nn.Parameter(torch.ones(1))
        
        logger.info(f"PolicyNetwork initialized: {emb_dim}-D → {hidden_dim}-D, {num_heads} heads")

    def forward(self, problem_emb, candidate_embs):
        """
        Forward pass với multi-head attention và adaptive scoring
        
        Args:
            problem_emb: shape [1, 1536] - problem embedding
            candidate_embs: shape [N, 1536] - candidate embeddings
            
        Returns:
            probs: shape [N] - probability distribution over candidates
        """
        batch_size = candidate_embs.size(0)
        
        # Normalize embeddings
        problem_emb = F.normalize(problem_emb, p=2, dim=-1)
        candidate_embs = F.normalize(candidate_embs, p=2, dim=-1)
        
        # Project to hidden dimension
        problem_h = self.input_projection(problem_emb)  # [1, hidden_dim]
        candidate_h = self.input_projection(candidate_embs)  # [N, hidden_dim]
        
        # Combine for attention (problem as query, candidates as key/value)
        combined = torch.cat([problem_h, candidate_h], dim=0)  # [N+1, hidden_dim]
        
        # Multi-head attention
        attn_out, attn_weights = self.attention(
            query=combined,
            key=combined, 
            value=combined
        )
        
        # Residual connection và layer norm
        combined = self.layer_norm1(combined + attn_out)
        
        # Feed-forward network
        ffn_out = self.ffn(combined)
        combined = self.layer_norm2(combined + ffn_out)
        
        # Extract candidate representations (skip problem at index 0)
        candidate_repr = combined[1:]  # [N, hidden_dim]
        problem_repr = combined[0:1]   # [1, hidden_dim]
        
        # Calculate interaction scores
        interaction_scores = torch.matmul(candidate_repr, problem_repr.T).squeeze(-1)  # [N]
        
        # Additional scoring through projection
        projected_scores = self.score_projection(candidate_repr).squeeze(-1)  # [N]
        
        # Combine scores
        final_scores = interaction_scores + projected_scores
        
        # Apply adaptive temperature và softmax
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
    Contrastive loss để pull positive examples closer, push negative ones away
    
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
        old_probs: probabilities từ old policy
        new_probs: probabilities từ current policy  
        advantages: advantage values
        epsilon: clipping parameter
        
    Returns:
        ppo_loss: clipped policy loss
    """
    ratio = new_probs / (old_probs.detach() + 1e-8)
    clipped_ratio = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)
    
    policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
    return policy_loss 