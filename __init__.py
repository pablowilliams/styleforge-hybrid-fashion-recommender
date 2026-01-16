"""
Project StyleForge: Hybrid Visual-Collaborative Fashion Recommendation System
==============================================================================

A comprehensive recommendation system combining visual similarity with
collaborative filtering for personalised fashion recommendations.

Modules:
    visual_encoder: ResNet50-based visual feature extraction
    ncf_model: Neural Collaborative Filtering
    attention_fusion: Attention-based modality fusion
    recommender: Complete recommendation pipeline
    evaluation: Metrics and diversity measures
"""

from .visual_encoder import VisualEncoder, TripletLoss, FashionImageDataset
from .ncf_model import NeuralCollaborativeFilter, NCFTrainer
from .attention_fusion import AttentionFusion, ContextAwareAttention
from .recommender import HybridRecommender
from .evaluation import (
    RecommenderEvaluator,
    hit_rate_at_k,
    ndcg_at_k,
    mmr_rerank,
    intra_list_diversity,
    catalogue_coverage
)

__version__ = "1.0.0"
__author__ = "Pablo Williams"

__all__ = [
    "VisualEncoder",
    "TripletLoss",
    "FashionImageDataset",
    "NeuralCollaborativeFilter",
    "NCFTrainer",
    "AttentionFusion",
    "ContextAwareAttention",
    "HybridRecommender",
    "RecommenderEvaluator",
    "hit_rate_at_k",
    "ndcg_at_k",
    "mmr_rerank",
    "intra_list_diversity",
    "catalogue_coverage"
]
