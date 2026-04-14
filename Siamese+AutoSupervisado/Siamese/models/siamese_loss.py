import torch
import torch.nn as nn
import torch.nn.functional as F  # <--- 1. ADD THIS IMPORT

class TripletLoss(nn.Module):
    """Online hard triplet loss for EEG classification.

    For each anchor in the batch:
      - Hardest positive: same-class sample with the LARGEST Euclidean distance.
      - Hardest negative: different-class sample with the SMALLEST Euclidean distance.

    Operates in raw Euclidean space on get_embedding() output (NOT L2-normalised).
    Anchors without a valid same-class partner in the batch are excluded from the mean.

    Args:
        margin (float): Margin for the triplet inequality. Default: 1.0.

    Inputs:
        embeddings (Tensor): Raw encoder output, shape (N, d_model).
        labels     (Tensor): Class indices, shape (N,).

    Returns:
        Scalar loss (mean over valid anchors).
    """

    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        N = embeddings.size(0)
        device = embeddings.device

        embeddings = F.normalize(embeddings, p=2, dim=1)

        # --- Pairwise Euclidean distances (N x N) ---
        # ||a - b||^2 = ||a||^2 + ||b||^2 - 2 a·b
        dot = torch.mm(embeddings, embeddings.t())                   # (N, N)
        sq_norm = dot.diag().unsqueeze(1)                            # (N, 1)
        sq_dist = sq_norm + sq_norm.t() - 2.0 * dot                 # (N, N)
        sq_dist = sq_dist.clamp(min=1e-12)                          # numerical safety
        dist = sq_dist.sqrt()                                        # (N, N)

        # --- Boolean masks ---
        labels_row = labels.unsqueeze(1)                             # (N, 1)
        same_class = labels_row.eq(labels_row.t())                   # (N, N)
        eye = torch.eye(N, dtype=torch.bool, device=device)

        pos_mask = same_class & ~eye     # same class, exclude self
        neg_mask = ~same_class           # different class (diagonal already False)

        # --- Hardest positive: max distance within same class ---
        # Zero out non-positives before taking max
        pos_dist = dist * pos_mask.float()
        hardest_pos, _ = pos_dist.max(dim=1)                        # (N,)

        # --- Hardest negative: min distance across different classes ---
        # Replace non-negatives with large sentinel
        neg_dist = dist + (~neg_mask).float() * 1e9
        hardest_neg, _ = neg_dist.min(dim=1)                        # (N,)

        # --- Triplet loss (hinge) ---
        triplet = (hardest_pos - hardest_neg + self.margin).clamp(min=0.0)  # (N,)

        # --- Exclude anchors with no same-class partner in the batch ---
        valid = pos_mask.any(dim=1)                                  # (N,) bool
        if valid.sum() == 0:
            # Degenerate batch (all unique classes) — return zero with grad
            return (embeddings * 0.0).sum()

        return triplet[valid].mean()
