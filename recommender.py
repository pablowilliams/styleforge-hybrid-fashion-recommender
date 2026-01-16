"""
Complete Recommendation Pipeline
Project StyleForge: Hybrid Fashion Recommendation System

This module integrates all components into a unified recommendation system.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import json
from collections import defaultdict

try:
    import faiss
except ImportError:
    print("Installing faiss...")
    import subprocess
    subprocess.check_call(["pip", "install", "faiss-cpu"])
    import faiss

from .visual_encoder import VisualEncoder
from .ncf_model import NeuralCollaborativeFilter
from .attention_fusion import AttentionFusion, HybridRecommendationHead


class HybridRecommender:
    """
    Complete hybrid recommendation system combining visual and collaborative signals.
    """
    
    def __init__(
        self,
        visual_encoder: VisualEncoder,
        ncf_model: NeuralCollaborativeFilter,
        fusion_model: AttentionFusion,
        device: str = "cuda"
    ):
        """
        Initialise hybrid recommender.
        
        Args:
            visual_encoder: Trained visual encoder
            ncf_model: Trained NCF model
            fusion_model: Trained attention fusion model
            device: Compute device
        """
        self.device = device
        
        self.visual_encoder = visual_encoder.to(device).eval()
        self.ncf_model = ncf_model.to(device).eval()
        self.fusion_model = fusion_model.to(device).eval()
        
        # Item embeddings cache
        self.item_visual_embeddings = None
        self.item_collab_embeddings = None
        self.item_ids = None
        
        # FAISS index for fast retrieval
        self.faiss_index = None
    
    def build_item_index(
        self,
        item_ids: List[int],
        visual_embeddings: np.ndarray,
        use_gpu: bool = False
    ) -> None:
        """
        Build FAISS index for fast approximate nearest neighbour search.
        
        Args:
            item_ids: List of item IDs
            visual_embeddings: Precomputed visual embeddings
            use_gpu: Whether to use GPU for FAISS
        """
        self.item_ids = np.array(item_ids)
        self.item_visual_embeddings = visual_embeddings.astype(np.float32)
        
        # Precompute collaborative embeddings
        with torch.no_grad():
            item_tensor = torch.tensor(item_ids).to(self.device)
            self.item_collab_embeddings = (
                self.ncf_model.get_item_embedding(item_tensor).cpu().numpy()
            )
        
        # Build FAISS index
        dim = visual_embeddings.shape[1]
        
        # Use IVF index for scalability
        nlist = min(100, len(item_ids) // 10)
        quantizer = faiss.IndexFlatIP(dim)
        self.faiss_index = faiss.IndexIVFFlat(
            quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT
        )
        
        # Normalise for cosine similarity
        faiss.normalize_L2(self.item_visual_embeddings)
        
        # Train and add
        self.faiss_index.train(self.item_visual_embeddings)
        self.faiss_index.add(self.item_visual_embeddings)
        
        # Set search parameters
        self.faiss_index.nprobe = 10
        
        if use_gpu and faiss.get_num_gpus() > 0:
            self.faiss_index = faiss.index_cpu_to_gpu(
                faiss.StandardGpuResources(), 0, self.faiss_index
            )
        
        print(f"Built FAISS index with {len(item_ids)} items")
    
    def get_user_profile(
        self,
        user_id: int,
        user_history: List[int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute user profile from interaction history.
        
        Args:
            user_id: User ID
            user_history: List of interacted item IDs
            
        Returns:
            Visual and collaborative user profiles
        """
        # Visual profile: average of interacted items
        if user_history:
            history_indices = [
                np.where(self.item_ids == item_id)[0][0]
                for item_id in user_history
                if item_id in self.item_ids
            ]
            if history_indices:
                visual_profile = self.item_visual_embeddings[history_indices].mean(axis=0)
            else:
                visual_profile = np.zeros(self.item_visual_embeddings.shape[1])
        else:
            visual_profile = np.zeros(self.item_visual_embeddings.shape[1])
        
        # Collaborative profile from NCF
        with torch.no_grad():
            user_tensor = torch.tensor([user_id]).to(self.device)
            collab_profile = self.ncf_model.get_user_embedding(user_tensor).cpu().numpy()[0]
        
        return visual_profile, collab_profile
    
    def recommend(
        self,
        user_id: int,
        user_history: List[int],
        k: int = 12,
        candidate_pool: Optional[List[int]] = None,
        exclude_history: bool = True,
        diversity_lambda: float = 0.0
    ) -> List[Tuple[int, float]]:
        """
        Generate recommendations for a user.
        
        Args:
            user_id: User ID
            user_history: User interaction history
            k: Number of recommendations
            candidate_pool: Optional restricted candidate set
            exclude_history: Whether to exclude already-interacted items
            diversity_lambda: MMR diversity parameter (0=relevance only)
            
        Returns:
            List of (item_id, score) tuples
        """
        # Get user profile
        visual_profile, collab_profile = self.get_user_profile(user_id, user_history)
        
        # Determine candidates
        if candidate_pool is not None:
            candidates = np.array([
                i for i, item_id in enumerate(self.item_ids)
                if item_id in candidate_pool
            ])
        else:
            candidates = np.arange(len(self.item_ids))
        
        # Exclude history
        if exclude_history and user_history:
            history_set = set(user_history)
            candidates = np.array([
                i for i in candidates
                if self.item_ids[i] not in history_set
            ])
        
        if len(candidates) == 0:
            return []
        
        # Get candidate embeddings
        candidate_visual = self.item_visual_embeddings[candidates]
        candidate_collab = self.item_collab_embeddings[candidates]
        
        # Compute hybrid scores
        with torch.no_grad():
            # Expand user profile
            batch_size = len(candidates)
            user_visual = torch.tensor(visual_profile).unsqueeze(0).expand(batch_size, -1).to(self.device)
            user_collab = torch.tensor(collab_profile).unsqueeze(0).expand(batch_size, -1).to(self.device)
            
            item_visual = torch.tensor(candidate_visual).to(self.device)
            item_collab = torch.tensor(candidate_collab).to(self.device)
            
            # Fuse and score
            user_fused, _ = self.fusion_model(user_visual.float(), user_collab.float())
            item_fused, _ = self.fusion_model(item_visual.float(), item_collab.float())
            
            # Cosine similarity
            scores = torch.sum(user_fused * item_fused, dim=1).cpu().numpy()
        
        # Apply diversity re-ranking if requested
        if diversity_lambda > 0:
            selected = self._mmr_rerank(
                candidates, scores, candidate_visual, k, diversity_lambda
            )
        else:
            # Simple top-K
            top_indices = np.argsort(scores)[-k:][::-1]
            selected = [(candidates[i], scores[i]) for i in top_indices]
        
        # Convert to item IDs
        recommendations = [
            (int(self.item_ids[idx]), float(score))
            for idx, score in selected
        ]
        
        return recommendations
    
    def _mmr_rerank(
        self,
        candidates: np.ndarray,
        scores: np.ndarray,
        embeddings: np.ndarray,
        k: int,
        lambda_param: float
    ) -> List[Tuple[int, float]]:
        """
        Maximal Marginal Relevance re-ranking.
        
        Args:
            candidates: Candidate indices
            scores: Relevance scores
            embeddings: Item embeddings
            k: Number to select
            lambda_param: Diversity weight
            
        Returns:
            Selected (index, score) pairs
        """
        selected = []
        remaining = list(range(len(candidates)))
        
        while len(selected) < k and remaining:
            mmr_scores = []
            
            for idx in remaining:
                relevance = scores[idx]
                
                if selected:
                    # Max similarity to selected
                    similarities = [
                        np.dot(embeddings[idx], embeddings[s])
                        / (np.linalg.norm(embeddings[idx]) * np.linalg.norm(embeddings[s]) + 1e-8)
                        for s, _ in selected
                    ]
                    max_sim = max(similarities)
                else:
                    max_sim = 0
                
                mmr = lambda_param * relevance - (1 - lambda_param) * max_sim
                mmr_scores.append((idx, mmr, scores[idx]))
            
            # Select best MMR
            best = max(mmr_scores, key=lambda x: x[1])
            selected.append((candidates[best[0]], best[2]))
            remaining.remove(best[0])
        
        return selected
    
    def recommend_similar(
        self,
        item_id: int,
        k: int = 12,
        visual_weight: float = 0.5
    ) -> List[Tuple[int, float]]:
        """
        Find similar items (item-to-item recommendations).
        
        Args:
            item_id: Query item ID
            k: Number of similar items
            visual_weight: Weight for visual similarity
            
        Returns:
            List of (item_id, score) tuples
        """
        # Find item index
        item_indices = np.where(self.item_ids == item_id)[0]
        if len(item_indices) == 0:
            return []
        
        idx = item_indices[0]
        
        # Visual similarity via FAISS
        query_visual = self.item_visual_embeddings[idx:idx+1].copy()
        faiss.normalize_L2(query_visual)
        
        visual_scores, visual_indices = self.faiss_index.search(query_visual, k + 1)
        visual_scores = visual_scores[0]
        visual_indices = visual_indices[0]
        
        # Collaborative similarity
        query_collab = self.item_collab_embeddings[idx]
        collab_scores = np.dot(self.item_collab_embeddings, query_collab)
        collab_scores = collab_scores / (
            np.linalg.norm(self.item_collab_embeddings, axis=1) * 
            np.linalg.norm(query_collab) + 1e-8
        )
        
        # Combine scores
        results = []
        for i, vis_idx in enumerate(visual_indices):
            if vis_idx == idx:
                continue
            
            combined = (
                visual_weight * visual_scores[i] + 
                (1 - visual_weight) * collab_scores[vis_idx]
            )
            results.append((int(self.item_ids[vis_idx]), float(combined)))
        
        results.sort(key=lambda x: -x[1])
        return results[:k]
    
    def save(self, output_dir: str) -> None:
        """Save all model components."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        torch.save(self.visual_encoder.state_dict(), output_dir / "visual_encoder.pt")
        torch.save(self.ncf_model.state_dict(), output_dir / "ncf_model.pt")
        torch.save(self.fusion_model.state_dict(), output_dir / "fusion_model.pt")
        
        # Save item index
        if self.item_ids is not None:
            np.save(output_dir / "item_ids.npy", self.item_ids)
            np.save(output_dir / "item_visual_embeddings.npy", self.item_visual_embeddings)
            np.save(output_dir / "item_collab_embeddings.npy", self.item_collab_embeddings)
        
        print(f"Saved models to {output_dir}")
    
    @classmethod
    def load(
        cls,
        model_dir: str,
        num_users: int,
        num_items: int,
        device: str = "cuda"
    ) -> "HybridRecommender":
        """Load saved recommender."""
        model_dir = Path(model_dir)
        
        # Initialise models
        visual_encoder = VisualEncoder(embedding_dim=256)
        visual_encoder.load_state_dict(torch.load(model_dir / "visual_encoder.pt"))
        
        ncf_model = NeuralCollaborativeFilter(num_users, num_items)
        ncf_model.load_state_dict(torch.load(model_dir / "ncf_model.pt"))
        
        fusion_model = AttentionFusion(visual_dim=256, collab_dim=128)
        fusion_model.load_state_dict(torch.load(model_dir / "fusion_model.pt"))
        
        recommender = cls(visual_encoder, ncf_model, fusion_model, device)
        
        # Load item index if available
        if (model_dir / "item_ids.npy").exists():
            recommender.item_ids = np.load(model_dir / "item_ids.npy")
            recommender.item_visual_embeddings = np.load(model_dir / "item_visual_embeddings.npy")
            recommender.item_collab_embeddings = np.load(model_dir / "item_collab_embeddings.npy")
        
        return recommender


def train_hybrid_recommender(
    train_interactions,
    val_interactions,
    image_dir: str,
    metadata_df,
    output_dir: str = "models",
    epochs: int = 50,
    device: str = "cuda"
) -> HybridRecommender:
    """
    End-to-end training of hybrid recommender.
    
    Args:
        train_interactions: Training interaction DataFrame
        val_interactions: Validation interactions
        image_dir: Directory with product images
        metadata_df: Product metadata
        output_dir: Output directory
        epochs: Training epochs
        device: Compute device
        
    Returns:
        Trained HybridRecommender
    """
    from .visual_encoder import VisualEncoderTrainer, FashionImageDataset
    from .ncf_model import NCFTrainer
    
    # Get dimensions
    num_users = train_interactions["user_id"].max() + 1
    num_items = train_interactions["item_id"].max() + 1
    
    print(f"Training with {num_users} users and {num_items} items")
    
    # Step 1: Train visual encoder
    print("\n=== Training Visual Encoder ===")
    visual_encoder = VisualEncoder(embedding_dim=256)
    visual_dataset = FashionImageDataset(image_dir, metadata_df, triplet_mode=True)
    visual_trainer = VisualEncoderTrainer(visual_encoder, device)
    visual_trainer.train(visual_dataset, epochs=30, output_dir=output_dir)
    
    # Step 2: Train NCF
    print("\n=== Training NCF Model ===")
    ncf_model = NeuralCollaborativeFilter(num_users, num_items)
    ncf_trainer = NCFTrainer(ncf_model, device)
    ncf_trainer.train(train_interactions, val_interactions, epochs=epochs, output_dir=output_dir)
    
    # Step 3: Train fusion model
    print("\n=== Training Fusion Model ===")
    fusion_model = AttentionFusion(visual_dim=256, collab_dim=128)
    # Fusion training would go here (simplified for brevity)
    
    # Create recommender
    recommender = HybridRecommender(visual_encoder, ncf_model, fusion_model, device)
    recommender.save(output_dir)
    
    return recommender


if __name__ == "__main__":
    print("StyleForge Hybrid Recommender")
    print("=" * 40)
    
    # Example initialisation
    visual_encoder = VisualEncoder(embedding_dim=256)
    ncf_model = NeuralCollaborativeFilter(num_users=10000, num_items=50000)
    fusion_model = AttentionFusion(visual_dim=256, collab_dim=128)
    
    recommender = HybridRecommender(
        visual_encoder, ncf_model, fusion_model, device="cpu"
    )
    
    print("Recommender initialised successfully")
