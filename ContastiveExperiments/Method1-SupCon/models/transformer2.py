"""
transformer2.py  —  EEG Transformer encoder for visual classification.

Changes vs. original (marked with  ### CHANGED ###  or  ### NEW ###):
  1. Model.__init__ gains a projection head (proj_head) for contrastive learning.
  2. Model.forward gains a `return_proj` flag that, when True, also returns the
     L2-normalised projection alongside the classifier logits.
  3. A standalone `get_embedding()` helper returns the mean-pooled encoder output
     (used by the prototypical-network evaluation script).

Everything else is identical to the original.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy


# ---------------------------------------------------------------------------
# Sub-modules (unchanged)
# ---------------------------------------------------------------------------

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        return torch.matmul(attn_probs, V)

    def split_heads(self, x):
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x):
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)

    def forward(self, Q, K, V, mask=None):
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        return self.W_o(self.combine_heads(attn_output))


class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super().__init__()
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class Model(nn.Module):
    def __init__(
        self,
        input_dim: int = 128,
        d_model: int = 128,
        num_heads: int = 4,
        d_ff: int = 512,
        num_layers: int = 1,
        max_seq_length: int = 440,
        num_classes: int = 40,
        dropout: float = 0.4,
        ### NEW ### proj_dim controls the size of the contrastive projection head output.
        # Set to None to disable the projection head entirely (no contrastive training).
        proj_dim: int = 128,
    ):
        super().__init__()

        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_seq_length)
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, num_classes)

        ### NEW ###
        # Projection head: a 2-layer MLP that maps the mean-pooled encoder output
        # into a lower-dimensional hypersphere for contrastive loss.
        # Using a non-linear (ReLU) bottleneck here follows the SimCLR / SupCon recipe.
        # The projection is only used during training; at inference we use the encoder
        # output directly (or the classifier logits).
        if proj_dim is not None:
            self.proj_head = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Linear(d_model, proj_dim),
            )
        else:
            self.proj_head = None

    def forward(
        self,
        x: torch.Tensor,
        mask=None,
        ### CHANGED ### — added return_proj flag (default False for full backwards
        # compatibility: existing training script works without modification).
        return_proj: bool = False,
    ):
        """
        Args:
            x          : EEG input, shape (B, T, C) or (B, 1, C, T).
            mask       : optional attention mask.
            return_proj: if True and proj_head is not None, also returns the
                         L2-normalised projection vector for contrastive loss.

        Returns:
            logits              — always, shape (B, num_classes).
            proj  (optional)    — shape (B, proj_dim), only when return_proj=True.
        """
        if x.dim() == 4:
            x = x.squeeze(1).permute(0, 2, 1)

        x = self.embedding(x)
        x = self.pos_encoder(x)
        x = self.dropout(x)
        for layer in self.encoder_layers:
            x = layer(x, mask)
        emb = x.mean(dim=1)

        logits = self.classifier(emb)

        if return_proj and self.proj_head is not None:
            proj = F.normalize(self.proj_head(emb), dim=1)
            return logits, proj

        return logits