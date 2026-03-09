"""
Supervised Contrastive Loss (Khosla et al., NeurIPS 2020)
Adapted for EEG classification.

Usage:
    loss_fn = SupConLoss(temperature=0.07)
    loss = loss_fn(projected_embeddings, labels)

Args:
    features : (N, proj_dim) — L2-normalised projection vectors, one per sample.
    labels   : (N,)          — integer class labels (0 … num_classes-1).

The loss pulls all same-class samples together in the hypersphere and
pushes different-class samples apart. It is computed over every ordered
pair (i, j) where j shares the label of i, using all other samples in the
batch as negatives.

Temperature: lower → sharper decision boundary but harder optimisation.
Start with 0.07 and tune if needed.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (N, D) — raw or already-normalised embeddings.
            labels:   (N,)   — integer class labels.
        Returns:
            Scalar loss.
        """
        device = features.device
        N = features.shape[0]

        # L2-normalise so the dot product = cosine similarity
        features = F.normalize(features, dim=1)  # (N, D)

        # --- similarity matrix ---
        sim = torch.matmul(features, features.T) / self.temperature  # (N, N)

        # Numerical stability: subtract row-wise max (like log-sum-exp trick)
        sim_max, _ = sim.max(dim=1, keepdim=True)
        sim = sim - sim_max.detach()

        # --- masks ---
        labels = labels.view(-1, 1)                              # (N, 1)
        pos_mask = (labels == labels.T).float().to(device)       # 1 where same class
        pos_mask.fill_diagonal_(0)                               # exclude self

        # Denominator mask: all pairs except self
        denom_mask = torch.ones(N, N, device=device)
        denom_mask.fill_diagonal_(0)

        # --- log probability ---
        exp_sim = torch.exp(sim) * denom_mask                    # exclude self from sum
        # safe log: add epsilon to avoid log(0)
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)  # (N, N)

        # --- mean over positives for each anchor ---
        pos_count = pos_mask.sum(dim=1)  # (N,)
        # Only compute loss for anchors that have at least one positive
        # (matters for very small batches or unbalanced sampling)
        has_positive = pos_count > 0

        loss_per_anchor = -(pos_mask * log_prob).sum(dim=1) / (pos_count + 1e-8)  # (N,)
        loss = loss_per_anchor[has_positive].mean()

        return loss
