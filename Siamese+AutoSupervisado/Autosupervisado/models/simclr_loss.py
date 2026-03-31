import torch
import torch.nn as nn
import torch.nn.functional as F


class NTXentLoss(nn.Module):
    """
    Normalised Temperature-scaled Cross-Entropy Loss (NT-Xent).
    Used in SimCLR (Chen et al., ICML 2020) for self-supervised contrastive learning.

    Given two augmented views of the same batch (z_i, z_j), each of shape (N, dim),
    the loss treats the pair (z_i[k], z_j[k]) as the positive pair for sample k,
    and all 2(N-1) other views in the concatenated batch as negatives.

    No labels are used — positive pairs are defined purely by sample identity.

    Args:
        temperature (float): Softmax temperature τ. Standard SimCLR default is 0.5.
                             Lower values produce sharper distributions.
        device (torch.device): Device on which to create the negative mask.
    """

    def __init__(self, temperature=0.5, device=torch.device("cuda")):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature
        self.device = device

    def forward(self, z_i, z_j):
        """
        Compute NT-Xent loss over a batch of two augmented views.

        Args:
            z_i (Tensor): L2-normalised projections of view 1, shape (N, dim).
            z_j (Tensor): L2-normalised projections of view 2, shape (N, dim).
                          z_i and z_j must correspond to the same samples in the
                          same order — i.e. z_i[k] and z_j[k] are the positive pair.

        Returns:
            Tensor: Scalar NT-Xent loss averaged over all 2N samples.
        """
        N = z_i.size(0)

        # Concatenate both views into a single 2N x dim matrix.
        # Layout: [z_i[0], ..., z_i[N-1], z_j[0], ..., z_j[N-1]]
        z = torch.cat([z_i, z_j], dim=0)  # (2N, dim)

        # Compute pairwise cosine similarity matrix.
        # Since z_i and z_j are already L2-normalised, this is just z @ z^T.
        # Shape: (2N, 2N). Entry [a, b] = cos_sim(z[a], z[b]).
        sim = torch.mm(z, z.t()) / self.temperature  # (2N, 2N)

        # Build the positive pair index mapping.
        # For sample k in [0, N): its positive is at index k+N (the j-view).
        # For sample k in [N, 2N): its positive is at index k-N (the i-view).
        pos_idx = torch.cat([
            torch.arange(N, 2 * N, device=self.device),  # i-view → j-view
            torch.arange(0, N, device=self.device)        # j-view → i-view
        ])  # (2N,)

        # Mask out self-similarity (diagonal) so a sample is never its own negative.
        # This replaces sim[k, k] with -inf before the softmax, effectively removing
        # the self-pair from the denominator.
        self_mask = torch.eye(2 * N, dtype=torch.bool, device=self.device)
        sim = sim.masked_fill(self_mask, float('-inf'))

        # Cross-entropy over the 2N-1 remaining entries per row.
        # F.cross_entropy expects logits (2N, 2N) and target indices (2N,).
        # The target for each row is the index of its positive partner.
        loss = F.cross_entropy(sim, pos_idx)

        return loss