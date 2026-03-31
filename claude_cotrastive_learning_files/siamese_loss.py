import torch
import torch.nn as nn
import torch.nn.functional as F


class SiameseLoss(nn.Module):
    """
    Online Contrastive Loss for Siamese Network training.

    Within each batch, all possible pairs are mined automatically.
    Positive pairs: two trials from the same class (y = 1).
    Negative pairs: two trials from different classes (y = 0).

    Loss formula per pair:
        L = y * d^2 + (1 - y) * max(0, margin - d)^2

    where d is the Euclidean distance between L2-normalised embeddings.

    Args:
        margin (float): minimum distance enforced between negative pairs.
                        Pairs already further apart than this contribute
                        zero gradient. Default: 1.0.
    """

    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings: (B, D) raw encoder output — normalised inside this fn.
            labels:     (B,)  integer class labels.

        Returns:
            Scalar loss averaged over all valid pairs in the batch.
        """
        # L2-normalise so distances live in [0, 2]
        emb = F.normalize(embeddings, p=2, dim=1)   # (B, D)

        # Pairwise squared Euclidean distance matrix  (B, B)
        # ||a - b||^2 = ||a||^2 + ||b||^2 - 2 a·b  = 2 - 2 a·b  (unit vectors)
        dot = torch.mm(emb, emb.t())                 # (B, B)
        dist_sq = (2.0 - 2.0 * dot).clamp(min=0.0)  # numerical safety

        # Binary same-class matrix  (B, B),  1 = positive pair
        labels_row = labels.unsqueeze(1)             # (B, 1)
        labels_col = labels.unsqueeze(0)             # (1, B)
        same = (labels_row == labels_col).float()    # (B, B)

        # Exclude diagonal (a sample paired with itself is trivially distance 0)
        B = embeddings.size(0)
        diag_mask = 1.0 - torch.eye(B, device=embeddings.device)
        same = same * diag_mask

        # Contrastive loss components
        pos_loss = same * dist_sq                                           # pull positives together
        neg_loss = (1.0 - same) * diag_mask * \
                   F.relu(self.margin - dist_sq.sqrt()).pow(2)              # push negatives apart

        # Average over all off-diagonal pairs
        n_pairs = diag_mask.sum().clamp(min=1.0)
        loss = (pos_loss + neg_loss).sum() / n_pairs

        return loss
