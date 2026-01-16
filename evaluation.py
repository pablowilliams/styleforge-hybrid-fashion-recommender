"""
Evaluation and Diversity Metrics Module
Project StyleForge: Hybrid Fashion Recommendation System

This module implements recommendation evaluation metrics and diversity measures.
"""

import numpy as np
from typing import List, Dict, Set, Optional, Tuple
from collections import defaultdict
import pandas as pd
from tqdm import tqdm


# =============================================================================
# DIVERSITY METRICS
# =============================================================================

def intra_list_diversity(
    recommendations: List[int],
    embeddings: np.ndarray,
    item_to_idx: Dict[int, int]
) -> float:
    """
    Compute Intra-List Diversity (ILD).
    
    ILD measures how different items within a recommendation list are.
    Higher ILD indicates more diverse recommendations.
    
    Args:
        recommendations: List of recommended item IDs
        embeddings: Item embedding matrix
        item_to_idx: Mapping from item ID to embedding index
        
    Returns:
        ILD score in [0, 1]
    """
    if len(recommendations) < 2:
        return 0.0
    
    # Get embeddings for recommended items
    indices = [item_to_idx.get(item_id) for item_id in recommendations if item_id in item_to_idx]
    if len(indices) < 2:
        return 0.0
    
    rec_embeddings = embeddings[indices]
    
    # Normalise
    norms = np.linalg.norm(rec_embeddings, axis=1, keepdims=True)
    rec_embeddings = rec_embeddings / (norms + 1e-8)
    
    # Compute pairwise similarities
    similarity_matrix = np.dot(rec_embeddings, rec_embeddings.T)
    
    # ILD = 1 - average pairwise similarity (excluding diagonal)
    n = len(indices)
    total_sim = (similarity_matrix.sum() - np.trace(similarity_matrix)) / (n * (n - 1))
    
    return 1.0 - total_sim


def catalogue_coverage(
    all_recommendations: List[List[int]],
    total_items: int
) -> float:
    """
    Compute catalogue coverage.
    
    Coverage measures what fraction of the catalogue appears in recommendations.
    
    Args:
        all_recommendations: List of recommendation lists (one per user)
        total_items: Total number of items in catalogue
        
    Returns:
        Coverage ratio in [0, 1]
    """
    recommended_items = set()
    for recs in all_recommendations:
        recommended_items.update(recs)
    
    return len(recommended_items) / total_items


def novelty(
    recommendations: List[int],
    item_popularity: Dict[int, int],
    total_interactions: int
) -> float:
    """
    Compute recommendation novelty.
    
    Novelty measures how unexpected/long-tail recommendations are.
    Uses self-information: -log2(popularity).
    
    Args:
        recommendations: List of recommended items
        item_popularity: Dict mapping item_id to interaction count
        total_interactions: Total number of interactions
        
    Returns:
        Average novelty score
    """
    if not recommendations:
        return 0.0
    
    novelty_scores = []
    for item_id in recommendations:
        pop = item_popularity.get(item_id, 1)
        prob = pop / total_interactions
        novelty_scores.append(-np.log2(prob + 1e-10))
    
    return np.mean(novelty_scores)


def mmr_rerank(
    candidates: List[int],
    scores: np.ndarray,
    embeddings: np.ndarray,
    item_to_idx: Dict[int, int],
    k: int = 12,
    lambda_param: float = 0.5
) -> List[int]:
    """
    Maximal Marginal Relevance re-ranking for diversity.
    
    Balances relevance with diversity by iteratively selecting items
    that are both relevant and different from already selected items.
    
    Args:
        candidates: Candidate item IDs
        scores: Relevance scores for candidates
        embeddings: Item embeddings
        item_to_idx: Mapping from item ID to embedding index
        k: Number of items to select
        lambda_param: Tradeoff parameter (1=relevance, 0=diversity)
        
    Returns:
        Re-ranked list of k item IDs
    """
    # Get embeddings
    valid_candidates = []
    valid_scores = []
    valid_embeddings = []
    
    for i, item_id in enumerate(candidates):
        if item_id in item_to_idx:
            valid_candidates.append(item_id)
            valid_scores.append(scores[i])
            valid_embeddings.append(embeddings[item_to_idx[item_id]])
    
    if not valid_candidates:
        return []
    
    valid_embeddings = np.array(valid_embeddings)
    valid_scores = np.array(valid_scores)
    
    # Normalise embeddings
    norms = np.linalg.norm(valid_embeddings, axis=1, keepdims=True)
    valid_embeddings = valid_embeddings / (norms + 1e-8)
    
    # Normalise scores to [0, 1]
    if valid_scores.max() > valid_scores.min():
        valid_scores = (valid_scores - valid_scores.min()) / (valid_scores.max() - valid_scores.min())
    
    selected = []
    selected_indices = []
    remaining = list(range(len(valid_candidates)))
    
    while len(selected) < k and remaining:
        mmr_scores = []
        
        for idx in remaining:
            relevance = valid_scores[idx]
            
            if selected_indices:
                # Max similarity to already selected
                selected_embs = valid_embeddings[selected_indices]
                similarities = np.dot(selected_embs, valid_embeddings[idx])
                max_sim = similarities.max()
            else:
                max_sim = 0
            
            mmr = lambda_param * relevance - (1 - lambda_param) * max_sim
            mmr_scores.append((idx, mmr))
        
        # Select highest MMR
        best_idx = max(mmr_scores, key=lambda x: x[1])[0]
        selected.append(valid_candidates[best_idx])
        selected_indices.append(best_idx)
        remaining.remove(best_idx)
    
    return selected


# =============================================================================
# RANKING METRICS
# =============================================================================

def hit_rate_at_k(
    recommendations: List[int],
    ground_truth: Set[int],
    k: int = 12
) -> float:
    """
    Compute Hit Rate @ K.
    
    Hit Rate measures whether any ground truth item appears in top-K.
    
    Args:
        recommendations: Ranked list of recommended items
        ground_truth: Set of relevant items
        k: Cutoff position
        
    Returns:
        1.0 if hit, 0.0 otherwise
    """
    top_k = set(recommendations[:k])
    return 1.0 if top_k & ground_truth else 0.0


def precision_at_k(
    recommendations: List[int],
    ground_truth: Set[int],
    k: int = 12
) -> float:
    """
    Compute Precision @ K.
    
    Args:
        recommendations: Ranked recommendation list
        ground_truth: Set of relevant items
        k: Cutoff position
        
    Returns:
        Precision score
    """
    top_k = recommendations[:k]
    hits = sum(1 for item in top_k if item in ground_truth)
    return hits / k


def recall_at_k(
    recommendations: List[int],
    ground_truth: Set[int],
    k: int = 12
) -> float:
    """
    Compute Recall @ K.
    
    Args:
        recommendations: Ranked recommendation list
        ground_truth: Set of relevant items
        k: Cutoff position
        
    Returns:
        Recall score
    """
    if not ground_truth:
        return 0.0
    
    top_k = set(recommendations[:k])
    hits = len(top_k & ground_truth)
    return hits / len(ground_truth)


def ndcg_at_k(
    recommendations: List[int],
    ground_truth: Set[int],
    k: int = 12
) -> float:
    """
    Compute Normalised Discounted Cumulative Gain @ K.
    
    NDCG accounts for position of relevant items in the ranking.
    
    Args:
        recommendations: Ranked recommendation list
        ground_truth: Set of relevant items
        k: Cutoff position
        
    Returns:
        NDCG score in [0, 1]
    """
    # DCG
    dcg = 0.0
    for i, item in enumerate(recommendations[:k]):
        if item in ground_truth:
            dcg += 1.0 / np.log2(i + 2)
    
    # IDCG (ideal DCG)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(k, len(ground_truth))))
    
    return dcg / idcg if idcg > 0 else 0.0


def mean_reciprocal_rank(
    recommendations: List[int],
    ground_truth: Set[int]
) -> float:
    """
    Compute Mean Reciprocal Rank.
    
    MRR is the reciprocal of the rank of the first relevant item.
    
    Args:
        recommendations: Ranked recommendation list
        ground_truth: Set of relevant items
        
    Returns:
        Reciprocal rank (0 if no hit)
    """
    for i, item in enumerate(recommendations):
        if item in ground_truth:
            return 1.0 / (i + 1)
    return 0.0


# =============================================================================
# EVALUATION FRAMEWORK
# =============================================================================

class RecommenderEvaluator:
    """
    Comprehensive evaluation framework for recommendation systems.
    """
    
    def __init__(
        self,
        embeddings: Optional[np.ndarray] = None,
        item_to_idx: Optional[Dict[int, int]] = None,
        item_popularity: Optional[Dict[int, int]] = None
    ):
        """
        Initialise evaluator.
        
        Args:
            embeddings: Item embeddings for diversity metrics
            item_to_idx: Item ID to embedding index mapping
            item_popularity: Item popularity counts
        """
        self.embeddings = embeddings
        self.item_to_idx = item_to_idx or {}
        self.item_popularity = item_popularity or {}
        self.total_interactions = sum(item_popularity.values()) if item_popularity else 1
    
    def evaluate_user(
        self,
        recommendations: List[int],
        ground_truth: Set[int],
        k: int = 12
    ) -> Dict[str, float]:
        """
        Evaluate recommendations for a single user.
        
        Args:
            recommendations: Ranked recommendation list
            ground_truth: Set of relevant items
            k: Cutoff position
            
        Returns:
            Dictionary of metric scores
        """
        metrics = {
            f"hit@{k}": hit_rate_at_k(recommendations, ground_truth, k),
            f"precision@{k}": precision_at_k(recommendations, ground_truth, k),
            f"recall@{k}": recall_at_k(recommendations, ground_truth, k),
            f"ndcg@{k}": ndcg_at_k(recommendations, ground_truth, k),
            "mrr": mean_reciprocal_rank(recommendations, ground_truth)
        }
        
        # Diversity metrics
        if self.embeddings is not None:
            metrics["ild"] = intra_list_diversity(
                recommendations[:k], self.embeddings, self.item_to_idx
            )
        
        if self.item_popularity:
            metrics["novelty"] = novelty(
                recommendations[:k], self.item_popularity, self.total_interactions
            )
        
        return metrics
    
    def evaluate_all(
        self,
        user_recommendations: Dict[int, List[int]],
        user_ground_truth: Dict[int, Set[int]],
        k: int = 12
    ) -> Dict[str, float]:
        """
        Evaluate recommendations for all users.
        
        Args:
            user_recommendations: Dict mapping user_id to recommendations
            user_ground_truth: Dict mapping user_id to ground truth items
            k: Cutoff position
            
        Returns:
            Aggregated metrics
        """
        all_metrics = defaultdict(list)
        all_recommendations = []
        
        for user_id, recs in tqdm(user_recommendations.items(), desc="Evaluating"):
            gt = user_ground_truth.get(user_id, set())
            if not gt:
                continue
            
            user_metrics = self.evaluate_user(recs, gt, k)
            for metric, value in user_metrics.items():
                all_metrics[metric].append(value)
            
            all_recommendations.append(recs[:k])
        
        # Aggregate
        results = {
            metric: np.mean(values)
            for metric, values in all_metrics.items()
        }
        
        # Coverage
        if self.item_to_idx:
            results[f"coverage@{k}"] = catalogue_coverage(
                all_recommendations, len(self.item_to_idx)
            )
        
        return results
    
    def evaluate_cold_start(
        self,
        user_recommendations: Dict[int, List[int]],
        user_ground_truth: Dict[int, Set[int]],
        user_history_lengths: Dict[int, int],
        k: int = 12,
        cold_threshold: int = 5
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate separately for cold-start and warm users.
        
        Args:
            user_recommendations: Recommendations per user
            user_ground_truth: Ground truth per user
            user_history_lengths: Number of prior interactions per user
            k: Cutoff position
            cold_threshold: Max interactions to be considered cold-start
            
        Returns:
            Separate metrics for cold and warm users
        """
        cold_users = {
            uid for uid, length in user_history_lengths.items()
            if length <= cold_threshold
        }
        warm_users = set(user_history_lengths.keys()) - cold_users
        
        cold_recs = {uid: recs for uid, recs in user_recommendations.items() if uid in cold_users}
        warm_recs = {uid: recs for uid, recs in user_recommendations.items() if uid in warm_users}
        
        cold_gt = {uid: gt for uid, gt in user_ground_truth.items() if uid in cold_users}
        warm_gt = {uid: gt for uid, gt in user_ground_truth.items() if uid in warm_users}
        
        return {
            "cold_start": self.evaluate_all(cold_recs, cold_gt, k),
            "warm_users": self.evaluate_all(warm_recs, warm_gt, k)
        }


def generate_evaluation_report(
    results: Dict[str, float],
    output_path: str
) -> None:
    """
    Generate formatted evaluation report.
    
    Args:
        results: Dictionary of metric results
        output_path: Output file path
    """
    report = []
    report.append("=" * 60)
    report.append("STYLEFORGE EVALUATION REPORT")
    report.append("=" * 60)
    report.append("")
    
    # Ranking metrics
    report.append("RANKING METRICS")
    report.append("-" * 40)
    for metric in ["hit@12", "precision@12", "recall@12", "ndcg@12", "mrr"]:
        if metric in results:
            report.append(f"  {metric.upper()}: {results[metric]:.4f}")
    
    report.append("")
    report.append("DIVERSITY METRICS")
    report.append("-" * 40)
    for metric in ["ild", "novelty", "coverage@12"]:
        if metric in results:
            report.append(f"  {metric.upper()}: {results[metric]:.4f}")
    
    report.append("")
    report.append("=" * 60)
    
    with open(output_path, "w") as f:
        f.write("\n".join(report))
    
    print("\n".join(report))


if __name__ == "__main__":
    # Example usage
    evaluator = RecommenderEvaluator()
    
    # Sample data
    recommendations = [1, 5, 3, 8, 2, 9, 4, 7, 6, 10, 11, 12]
    ground_truth = {3, 7, 15}
    
    metrics = evaluator.evaluate_user(recommendations, ground_truth, k=12)
    print("Single user metrics:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
