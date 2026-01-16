"""
Neural Collaborative Filtering Module
Project StyleForge: Hybrid Fashion Recommendation System

This module implements the NCF model combining GMF and MLP pathways
for learning user-item interaction patterns.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Optional, Tuple
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict


class NeuralCollaborativeFilter(nn.Module):
    """
    NCF model combining Generalised Matrix Factorisation (GMF) and 
    Multi-Layer Perceptron (MLP) pathways.
    
    Reference: He et al., "Neural Collaborative Filtering", WWW 2017.
    """
    
    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 64,
        mlp_layers: List[int] = None
    ):
        """
        Initialise NCF model.
        
        Args:
            num_users: Number of users
            num_items: Number of items
            embedding_dim: Embedding dimensionality
            mlp_layers: MLP hidden layer dimensions
        """
        super().__init__()
        
        if mlp_layers is None:
            mlp_layers = [256, 128, 64]
        
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        
        # GMF embeddings
        self.user_gmf = nn.Embedding(num_users, embedding_dim)
        self.item_gmf = nn.Embedding(num_items, embedding_dim)
        
        # MLP embeddings
        self.user_mlp = nn.Embedding(num_users, embedding_dim)
        self.item_mlp = nn.Embedding(num_items, embedding_dim)
        
        # MLP layers
        mlp_modules = []
        input_dim = 2 * embedding_dim
        for layer_dim in mlp_layers:
            mlp_modules.append(nn.Linear(input_dim, layer_dim))
            mlp_modules.append(nn.ReLU())
            mlp_modules.append(nn.BatchNorm1d(layer_dim))
            mlp_modules.append(nn.Dropout(0.2))
            input_dim = layer_dim
        self.mlp = nn.Sequential(*mlp_modules)
        
        # NeuMF fusion layer
        self.fusion = nn.Linear(embedding_dim + mlp_layers[-1], 1)
        
        # Initialisation
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Initialise embeddings with small random values."""
        nn.init.normal_(self.user_gmf.weight, std=0.01)
        nn.init.normal_(self.item_gmf.weight, std=0.01)
        nn.init.normal_(self.user_mlp.weight, std=0.01)
        nn.init.normal_(self.item_mlp.weight, std=0.01)
    
    def forward(
        self,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            user_ids: User indices [B]
            item_ids: Item indices [B]
            
        Returns:
            Predicted scores [B]
        """
        # GMF pathway: element-wise product
        user_gmf_emb = self.user_gmf(user_ids)
        item_gmf_emb = self.item_gmf(item_ids)
        gmf_output = user_gmf_emb * item_gmf_emb
        
        # MLP pathway: concatenation + MLP
        user_mlp_emb = self.user_mlp(user_ids)
        item_mlp_emb = self.item_mlp(item_ids)
        mlp_input = torch.cat([user_mlp_emb, item_mlp_emb], dim=-1)
        mlp_output = self.mlp(mlp_input)
        
        # NeuMF fusion
        concat = torch.cat([gmf_output, mlp_output], dim=-1)
        prediction = torch.sigmoid(self.fusion(concat))
        
        return prediction.squeeze(-1)
    
    def get_user_embedding(self, user_ids: torch.Tensor) -> torch.Tensor:
        """Get combined user embedding for downstream tasks."""
        gmf_emb = self.user_gmf(user_ids)
        mlp_emb = self.user_mlp(user_ids)
        return torch.cat([gmf_emb, mlp_emb], dim=-1)
    
    def get_item_embedding(self, item_ids: torch.Tensor) -> torch.Tensor:
        """Get combined item embedding for downstream tasks."""
        gmf_emb = self.item_gmf(item_ids)
        mlp_emb = self.item_mlp(item_ids)
        return torch.cat([gmf_emb, mlp_emb], dim=-1)


class InteractionDataset(Dataset):
    """
    Dataset for user-item interactions with negative sampling.
    """
    
    def __init__(
        self,
        interactions: pd.DataFrame,
        num_users: int,
        num_items: int,
        num_negatives: int = 4
    ):
        """
        Initialise dataset.
        
        Args:
            interactions: DataFrame with user_id, item_id columns
            num_users: Total number of users
            num_items: Total number of items
            num_negatives: Number of negatives per positive
        """
        self.interactions = interactions
        self.num_users = num_users
        self.num_items = num_items
        self.num_negatives = num_negatives
        
        # Build user interaction sets for negative sampling
        self.user_items = defaultdict(set)
        for _, row in interactions.iterrows():
            self.user_items[row["user_id"]].add(row["item_id"])
        
        self.all_items = set(range(num_items))
    
    def __len__(self) -> int:
        return len(self.interactions) * (1 + self.num_negatives)
    
    def __getitem__(self, idx: int) -> Tuple[int, int, float]:
        """
        Get training sample.
        
        Returns:
            Tuple of (user_id, item_id, label)
        """
        pos_idx = idx // (1 + self.num_negatives)
        is_negative = idx % (1 + self.num_negatives) > 0
        
        row = self.interactions.iloc[pos_idx]
        user_id = row["user_id"]
        
        if is_negative:
            # Sample negative item
            user_pos_items = self.user_items[user_id]
            neg_candidates = list(self.all_items - user_pos_items)
            item_id = np.random.choice(neg_candidates) if neg_candidates else 0
            label = 0.0
        else:
            item_id = row["item_id"]
            label = 1.0
        
        return user_id, item_id, label


class NCFTrainer:
    """
    Trainer for Neural Collaborative Filtering.
    """
    
    def __init__(
        self,
        model: NeuralCollaborativeFilter,
        device: str = "cuda"
    ):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.BCELoss()
    
    def train(
        self,
        train_data: pd.DataFrame,
        val_data: Optional[pd.DataFrame] = None,
        epochs: int = 50,
        batch_size: int = 1024,
        learning_rate: float = 0.001,
        num_negatives: int = 4,
        output_dir: str = "models"
    ) -> Dict:
        """
        Train NCF model.
        
        Args:
            train_data: Training interactions DataFrame
            val_data: Validation interactions
            epochs: Number of epochs
            batch_size: Batch size
            learning_rate: Learning rate
            num_negatives: Negatives per positive
            output_dir: Output directory
            
        Returns:
            Training history
        """
        train_dataset = InteractionDataset(
            train_data,
            self.model.num_users,
            self.model.num_items,
            num_negatives
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4
        )
        
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=1e-5
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=3
        )
        
        history = {"train_loss": [], "val_ndcg": []}
        best_ndcg = 0
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_losses = []
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
            for user_ids, item_ids, labels in pbar:
                user_ids = user_ids.to(self.device)
                item_ids = item_ids.to(self.device)
                labels = labels.float().to(self.device)
                
                optimizer.zero_grad()
                predictions = self.model(user_ids, item_ids)
                loss = self.criterion(predictions, labels)
                
                loss.backward()
                optimizer.step()
                
                train_losses.append(loss.item())
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
            avg_loss = np.mean(train_losses)
            history["train_loss"].append(avg_loss)
            
            # Validation
            if val_data is not None:
                val_ndcg = self._evaluate(val_data)
                history["val_ndcg"].append(val_ndcg)
                scheduler.step(val_ndcg)
                
                if val_ndcg > best_ndcg:
                    best_ndcg = val_ndcg
                    self._save_model(output_dir, "best_ncf.pt")
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                print(f"Epoch {epoch + 1}: Loss = {avg_loss:.4f}, Val NDCG@12 = {val_ndcg:.4f}")
                
                if patience_counter >= 5:
                    print("Early stopping triggered")
                    break
            else:
                print(f"Epoch {epoch + 1}: Loss = {avg_loss:.4f}")
        
        self._save_model(output_dir, "final_ncf.pt")
        return history
    
    def _evaluate(self, val_data: pd.DataFrame, k: int = 12) -> float:
        """
        Evaluate model on validation set.
        
        Args:
            val_data: Validation interactions
            k: Top-K for NDCG computation
            
        Returns:
            NDCG@K score
        """
        self.model.eval()
        
        # Group by user
        user_items = val_data.groupby("user_id")["item_id"].apply(set).to_dict()
        
        ndcg_scores = []
        
        with torch.no_grad():
            for user_id, ground_truth in user_items.items():
                if user_id >= self.model.num_users:
                    continue
                
                # Score all items
                user_tensor = torch.tensor([user_id] * self.model.num_items).to(self.device)
                item_tensor = torch.arange(self.model.num_items).to(self.device)
                
                scores = self.model(user_tensor, item_tensor).cpu().numpy()
                
                # Get top-K
                top_k = np.argsort(scores)[-k:][::-1]
                
                # Compute NDCG
                dcg = sum(
                    1.0 / np.log2(i + 2) 
                    for i, item in enumerate(top_k) 
                    if item in ground_truth
                )
                idcg = sum(
                    1.0 / np.log2(i + 2) 
                    for i in range(min(k, len(ground_truth)))
                )
                
                ndcg = dcg / idcg if idcg > 0 else 0
                ndcg_scores.append(ndcg)
        
        return np.mean(ndcg_scores) if ndcg_scores else 0.0
    
    def _save_model(self, output_dir: str, filename: str) -> None:
        """Save model checkpoint."""
        from pathlib import Path
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), output_dir / filename)
    
    def predict(
        self,
        user_ids: List[int],
        candidate_items: Optional[List[int]] = None,
        k: int = 12
    ) -> Dict[int, List[int]]:
        """
        Generate recommendations for users.
        
        Args:
            user_ids: List of user IDs
            candidate_items: Optional candidate item pool
            k: Number of recommendations
            
        Returns:
            Dictionary mapping user_id to recommended item list
        """
        self.model.eval()
        recommendations = {}
        
        candidates = candidate_items or list(range(self.model.num_items))
        
        with torch.no_grad():
            for user_id in user_ids:
                if user_id >= self.model.num_users:
                    recommendations[user_id] = []
                    continue
                
                user_tensor = torch.tensor([user_id] * len(candidates)).to(self.device)
                item_tensor = torch.tensor(candidates).to(self.device)
                
                scores = self.model(user_tensor, item_tensor).cpu().numpy()
                
                top_k_idx = np.argsort(scores)[-k:][::-1]
                top_k_items = [candidates[i] for i in top_k_idx]
                
                recommendations[user_id] = top_k_items
        
        return recommendations


if __name__ == "__main__":
    # Example usage
    model = NeuralCollaborativeFilter(
        num_users=10000,
        num_items=50000,
        embedding_dim=64,
        mlp_layers=[256, 128, 64]
    )
    
    print(f"NCF parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    users = torch.randint(0, 10000, (32,))
    items = torch.randint(0, 50000, (32,))
    scores = model(users, items)
    print(f"Output shape: {scores.shape}")
    print(f"Score range: [{scores.min():.3f}, {scores.max():.3f}]")
