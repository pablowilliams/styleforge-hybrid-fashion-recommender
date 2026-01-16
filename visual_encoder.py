"""
Visual Encoder Module
Project StyleForge: Hybrid Fashion Recommendation System

This module implements the ResNet50-based visual feature extractor
for fashion product images.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np
from tqdm import tqdm


class VisualEncoder(nn.Module):
    """
    ResNet50-based visual feature extractor for fashion images.
    
    Extracts dense embeddings that capture stylistic attributes including
    colour, texture, silhouette, and design patterns.
    """
    
    def __init__(
        self,
        embedding_dim: int = 256,
        freeze_backbone: bool = False,
        pretrained: bool = True
    ):
        """
        Initialise visual encoder.
        
        Args:
            embedding_dim: Output embedding dimensionality
            freeze_backbone: Whether to freeze ResNet backbone
            pretrained: Use ImageNet pretrained weights
        """
        super().__init__()
        
        # Load pretrained ResNet50
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        resnet = models.resnet50(weights=weights)
        
        # Remove classification head
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        
        # Projection head for embedding
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, embedding_dim),
            nn.BatchNorm1d(embedding_dim)
        )
        
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        self.embedding_dim = embedding_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract visual embeddings.
        
        Args:
            x: Input images [B, 3, H, W]
            
        Returns:
            L2-normalised embeddings [B, embedding_dim]
        """
        features = self.backbone(x)
        embeddings = self.projection(features)
        return F.normalize(embeddings, p=2, dim=1)
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract raw features before projection (for analysis).
        
        Args:
            x: Input images
            
        Returns:
            Raw 2048-dim features
        """
        with torch.no_grad():
            features = self.backbone(x)
        return features.flatten(1)


class TripletLoss(nn.Module):
    """
    Triplet margin loss with hard negative mining.
    """
    
    def __init__(self, margin: float = 0.3):
        super().__init__()
        self.margin = margin
    
    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute triplet loss.
        
        Args:
            anchor: Anchor embeddings [B, D]
            positive: Positive embeddings [B, D]
            negative: Negative embeddings [B, D]
            
        Returns:
            Triplet loss value
        """
        pos_dist = F.pairwise_distance(anchor, positive)
        neg_dist = F.pairwise_distance(anchor, negative)
        
        loss = F.relu(pos_dist - neg_dist + self.margin)
        return loss.mean()


class FashionImageDataset(Dataset):
    """
    Dataset for fashion product images with triplet sampling.
    """
    
    def __init__(
        self,
        image_dir: str,
        metadata_df,
        transform=None,
        triplet_mode: bool = True
    ):
        """
        Initialise dataset.
        
        Args:
            image_dir: Directory containing images
            metadata_df: DataFrame with article_id and product_type_name columns
            transform: Image transforms
            triplet_mode: Whether to return triplets
        """
        self.image_dir = Path(image_dir)
        self.metadata = metadata_df
        self.triplet_mode = triplet_mode
        
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # Build category index for triplet sampling
        self.category_to_items = {}
        for idx, row in self.metadata.iterrows():
            cat = row["product_type_name"]
            if cat not in self.category_to_items:
                self.category_to_items[cat] = []
            self.category_to_items[cat].append(idx)
        
        self.categories = list(self.category_to_items.keys())
        self.indices = list(range(len(self.metadata)))
    
    def __len__(self) -> int:
        return len(self.metadata)
    
    def __getitem__(self, idx: int):
        row = self.metadata.iloc[idx]
        article_id = str(row["article_id"]).zfill(10)
        
        # Load anchor image
        img_path = self.image_dir / f"{article_id[:3]}" / f"{article_id}.jpg"
        
        try:
            anchor_img = Image.open(img_path).convert("RGB")
        except:
            # Return placeholder if image not found
            anchor_img = Image.new("RGB", (224, 224), color=(128, 128, 128))
        
        anchor_img = self.transform(anchor_img)
        
        if not self.triplet_mode:
            return anchor_img, idx
        
        # Sample positive (same category)
        anchor_cat = row["product_type_name"]
        pos_candidates = self.category_to_items.get(anchor_cat, [idx])
        pos_idx = np.random.choice([i for i in pos_candidates if i != idx] or [idx])
        
        # Sample negative (different category)
        neg_cat = np.random.choice([c for c in self.categories if c != anchor_cat] or self.categories)
        neg_idx = np.random.choice(self.category_to_items[neg_cat])
        
        # Load positive and negative images
        pos_row = self.metadata.iloc[pos_idx]
        neg_row = self.metadata.iloc[neg_idx]
        
        pos_id = str(pos_row["article_id"]).zfill(10)
        neg_id = str(neg_row["article_id"]).zfill(10)
        
        try:
            pos_img = Image.open(self.image_dir / f"{pos_id[:3]}" / f"{pos_id}.jpg").convert("RGB")
        except:
            pos_img = Image.new("RGB", (224, 224), color=(128, 128, 128))
        
        try:
            neg_img = Image.open(self.image_dir / f"{neg_id[:3]}" / f"{neg_id}.jpg").convert("RGB")
        except:
            neg_img = Image.new("RGB", (224, 224), color=(128, 128, 128))
        
        pos_img = self.transform(pos_img)
        neg_img = self.transform(neg_img)
        
        return anchor_img, pos_img, neg_img, idx


class VisualEncoderTrainer:
    """
    Trainer for visual encoder with triplet loss.
    """
    
    def __init__(
        self,
        model: VisualEncoder,
        device: str = "cuda"
    ):
        self.model = model.to(device)
        self.device = device
        self.criterion = TripletLoss(margin=0.3)
    
    def train(
        self,
        train_dataset: FashionImageDataset,
        val_dataset: Optional[FashionImageDataset] = None,
        epochs: int = 30,
        batch_size: int = 64,
        learning_rate: float = 1e-4,
        output_dir: str = "models"
    ) -> Dict:
        """
        Train visual encoder.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            epochs: Number of epochs
            batch_size: Batch size
            learning_rate: Learning rate
            output_dir: Output directory
            
        Returns:
            Training history
        """
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        # Different learning rates for backbone and projection
        optimizer = torch.optim.Adam([
            {"params": self.model.backbone.parameters(), "lr": learning_rate * 0.1},
            {"params": self.model.projection.parameters(), "lr": learning_rate}
        ])
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=1e-6
        )
        
        history = {"train_loss": [], "val_loss": []}
        best_loss = float("inf")
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_losses = []
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
            for anchor, positive, negative, _ in pbar:
                anchor = anchor.to(self.device)
                positive = positive.to(self.device)
                negative = negative.to(self.device)
                
                optimizer.zero_grad()
                
                anchor_emb = self.model(anchor)
                pos_emb = self.model(positive)
                neg_emb = self.model(negative)
                
                loss = self.criterion(anchor_emb, pos_emb, neg_emb)
                
                loss.backward()
                optimizer.step()
                
                train_losses.append(loss.item())
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
            scheduler.step()
            
            avg_train_loss = np.mean(train_losses)
            history["train_loss"].append(avg_train_loss)
            
            # Validation
            if val_dataset:
                val_loss = self._validate(val_dataset, batch_size)
                history["val_loss"].append(val_loss)
                
                if val_loss < best_loss:
                    best_loss = val_loss
                    self._save_model(output_dir, "best_visual_encoder.pt")
            
            print(f"Epoch {epoch + 1}: Train Loss = {avg_train_loss:.4f}")
        
        self._save_model(output_dir, "final_visual_encoder.pt")
        return history
    
    def _validate(self, dataset: FashionImageDataset, batch_size: int) -> float:
        """Validate model."""
        self.model.eval()
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        losses = []
        
        with torch.no_grad():
            for anchor, positive, negative, _ in loader:
                anchor = anchor.to(self.device)
                positive = positive.to(self.device)
                negative = negative.to(self.device)
                
                anchor_emb = self.model(anchor)
                pos_emb = self.model(positive)
                neg_emb = self.model(negative)
                
                loss = self.criterion(anchor_emb, pos_emb, neg_emb)
                losses.append(loss.item())
        
        return np.mean(losses)
    
    def _save_model(self, output_dir: str, filename: str) -> None:
        """Save model checkpoint."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), output_dir / filename)


def extract_embeddings(
    model: VisualEncoder,
    image_dir: str,
    metadata_df,
    batch_size: int = 64,
    device: str = "cuda"
) -> Dict[str, np.ndarray]:
    """
    Extract embeddings for all images.
    
    Args:
        model: Trained visual encoder
        image_dir: Image directory
        metadata_df: Metadata DataFrame
        batch_size: Batch size
        device: Compute device
        
    Returns:
        Dictionary mapping article_id to embedding
    """
    model = model.to(device)
    model.eval()
    
    dataset = FashionImageDataset(
        image_dir,
        metadata_df,
        triplet_mode=False
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    embeddings = {}
    
    with torch.no_grad():
        for images, indices in tqdm(loader, desc="Extracting embeddings"):
            images = images.to(device)
            embs = model(images).cpu().numpy()
            
            for emb, idx in zip(embs, indices):
                article_id = str(metadata_df.iloc[idx.item()]["article_id"])
                embeddings[article_id] = emb
    
    return embeddings


if __name__ == "__main__":
    # Example usage
    model = VisualEncoder(embedding_dim=256)
    print(f"Visual Encoder parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    x = torch.randn(4, 3, 224, 224)
    embeddings = model(x)
    print(f"Output shape: {embeddings.shape}")
    print(f"Embedding norm: {embeddings.norm(dim=1)}")  # Should be ~1.0
