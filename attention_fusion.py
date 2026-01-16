"""
Attention-Based Fusion Module
Project StyleForge: Hybrid Fashion Recommendation System

This module implements the attention mechanism for dynamically combining
visual and collaborative signals based on user context.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import numpy as np


class AttentionFusion(nn.Module):
    """
    Attention-based fusion of visual and collaborative embeddings.
    
    Learns to dynamically weight visual versus behavioural signals based on
    the characteristics of the user and item being considered.
    """
    
    def __init__(
        self,
        visual_dim: int,
        collab_dim: int,
        hidden_dim: int = 128,
        num_heads: int = 4
    ):
        """
        Initialise attention fusion.
        
        Args:
            visual_dim: Dimensionality of visual embeddings
            collab_dim: Dimensionality of collaborative embeddings
            hidden_dim: Hidden layer dimensionality
            num_heads: Number of attention heads
        """
        super().__init__()
        
        self.visual_dim = visual_dim
        self.collab_dim = collab_dim
        self.hidden_dim = hidden_dim
        
        # Project both modalities to common space
        self.visual_proj = nn.Linear(visual_dim, hidden_dim)
        self.collab_proj = nn.Linear(collab_dim, hidden_dim)
        
        # Multi-head attention for modality weighting
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, num_heads * 2),  # 2 modalities
        )
        self.num_heads = num_heads
        
        # Gated fusion
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
    
    def forward(
        self,
        visual_emb: torch.Tensor,
        collab_emb: torch.Tensor,
        return_weights: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Fuse visual and collaborative embeddings.
        
        Args:
            visual_emb: Visual embeddings [B, visual_dim]
            collab_emb: Collaborative embeddings [B, collab_dim]
            return_weights: Whether to return attention weights
            
        Returns:
            Fused embeddings [B, hidden_dim]
            Attention weights [B, 2] if return_weights=True
        """
        batch_size = visual_emb.size(0)
        
        # Project to common space
        visual_h = F.relu(self.visual_proj(visual_emb))
        collab_h = F.relu(self.collab_proj(collab_emb))
        
        # Compute attention weights
        concat = torch.cat([visual_h, collab_h], dim=-1)
        attn_logits = self.attention(concat)
        attn_logits = attn_logits.view(batch_size, self.num_heads, 2)
        weights = F.softmax(attn_logits, dim=-1)
        
        # Average across heads
        weights = weights.mean(dim=1)  # [B, 2]
        
        # Weighted combination
        visual_weight = weights[:, 0:1]
        collab_weight = weights[:, 1:2]
        
        weighted_visual = visual_weight * visual_h
        weighted_collab = collab_weight * collab_h
        
        # Gated fusion
        gate_input = torch.cat([weighted_visual, weighted_collab], dim=-1)
        gate = self.gate(gate_input)
        
        fused = gate * weighted_visual + (1 - gate) * weighted_collab
        
        # Output projection
        output = self.output_proj(fused)
        
        if return_weights:
            return output, weights
        return output, None


class ContextAwareAttention(nn.Module):
    """
    Context-aware attention that considers user history length and item popularity.
    """
    
    def __init__(
        self,
        visual_dim: int,
        collab_dim: int,
        hidden_dim: int = 128,
        context_dim: int = 16
    ):
        """
        Initialise context-aware attention.
        
        Args:
            visual_dim: Visual embedding dimension
            collab_dim: Collaborative embedding dimension
            hidden_dim: Hidden layer dimension
            context_dim: Context feature dimension
        """
        super().__init__()
        
        # Projections
        self.visual_proj = nn.Linear(visual_dim, hidden_dim)
        self.collab_proj = nn.Linear(collab_dim, hidden_dim)
        
        # Context encoder
        self.context_encoder = nn.Sequential(
            nn.Linear(context_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim)
        )
        
        # Context-conditioned attention
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 2),
            nn.Softmax(dim=-1)
        )
        
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(
        self,
        visual_emb: torch.Tensor,
        collab_emb: torch.Tensor,
        context_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Fuse with context awareness.
        
        Args:
            visual_emb: Visual embeddings
            collab_emb: Collaborative embeddings
            context_features: Context features (user history length, item popularity, etc.)
            
        Returns:
            Fused embeddings and attention weights
        """
        # Project embeddings
        visual_h = F.relu(self.visual_proj(visual_emb))
        collab_h = F.relu(self.collab_proj(collab_emb))
        context_h = self.context_encoder(context_features)
        
        # Compute context-conditioned attention
        concat = torch.cat([visual_h, collab_h, context_h], dim=-1)
        weights = self.attention(concat)
        
        # Weighted combination
        fused = weights[:, 0:1] * visual_h + weights[:, 1:2] * collab_h
        output = self.output_proj(fused)
        
        return output, weights


class HybridRecommendationHead(nn.Module):
    """
    Complete recommendation head combining fusion with prediction.
    """
    
    def __init__(
        self,
        visual_dim: int,
        collab_dim: int,
        hidden_dim: int = 128
    ):
        """
        Initialise recommendation head.
        
        Args:
            visual_dim: Visual embedding dimension
            collab_dim: Collaborative embedding dimension
            hidden_dim: Hidden layer dimension
        """
        super().__init__()
        
        self.fusion = AttentionFusion(visual_dim, collab_dim, hidden_dim)
        
        # User and item towers
        self.user_tower = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )
        
        self.item_tower = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )
        
        # Prediction head
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(
        self,
        user_visual: torch.Tensor,
        user_collab: torch.Tensor,
        item_visual: torch.Tensor,
        item_collab: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute recommendation scores.
        
        Args:
            user_visual: User visual preferences
            user_collab: User collaborative embedding
            item_visual: Item visual embedding
            item_collab: Item collaborative embedding
            
        Returns:
            Prediction scores and attention weights
        """
        # Fuse modalities
        user_fused, user_weights = self.fusion(user_visual, user_collab, return_weights=True)
        item_fused, item_weights = self.fusion(item_visual, item_collab, return_weights=True)
        
        # Apply towers
        user_repr = self.user_tower(user_fused)
        item_repr = self.item_tower(item_fused)
        
        # Predict
        interaction = torch.cat([user_repr, item_repr], dim=-1)
        score = self.predictor(interaction)
        
        # Average weights for interpretability
        weights = (user_weights + item_weights) / 2
        
        return score.squeeze(-1), weights


def analyse_attention_patterns(
    model: AttentionFusion,
    visual_embeddings: torch.Tensor,
    collab_embeddings: torch.Tensor,
    user_history_lengths: np.ndarray
) -> dict:
    """
    Analyse learned attention patterns across user segments.
    
    Args:
        model: Trained fusion model
        visual_embeddings: Visual embeddings for test set
        collab_embeddings: Collaborative embeddings for test set
        user_history_lengths: Number of interactions per user
        
    Returns:
        Dictionary with attention pattern analysis
    """
    model.eval()
    
    with torch.no_grad():
        _, weights = model(visual_embeddings, collab_embeddings, return_weights=True)
        weights = weights.cpu().numpy()
    
    # Segment users
    segments = {
        "new_users": user_history_lengths < 5,
        "regular_users": (user_history_lengths >= 5) & (user_history_lengths < 50),
        "power_users": user_history_lengths >= 50
    }
    
    analysis = {}
    for segment_name, mask in segments.items():
        if mask.sum() > 0:
            segment_weights = weights[mask]
            analysis[segment_name] = {
                "mean_visual_weight": float(segment_weights[:, 0].mean()),
                "mean_collab_weight": float(segment_weights[:, 1].mean()),
                "std_visual_weight": float(segment_weights[:, 0].std()),
                "count": int(mask.sum())
            }
    
    return analysis


if __name__ == "__main__":
    # Example usage
    fusion = AttentionFusion(visual_dim=256, collab_dim=128, hidden_dim=128)
    print(f"Fusion parameters: {sum(p.numel() for p in fusion.parameters()):,}")
    
    # Test forward pass
    visual = torch.randn(32, 256)
    collab = torch.randn(32, 128)
    
    fused, weights = fusion(visual, collab, return_weights=True)
    print(f"Fused shape: {fused.shape}")
    print(f"Weights shape: {weights.shape}")
    print(f"Mean weights: Visual={weights[:, 0].mean():.3f}, Collab={weights[:, 1].mean():.3f}")
