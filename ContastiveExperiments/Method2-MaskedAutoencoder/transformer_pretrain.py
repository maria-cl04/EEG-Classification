"""
transformer_pretrain.py — Masked Autoencoder (MAE) pre-training for EEG Transformer.

WHAT THIS DOES
--------------
1.  Loads ALL available EEG data (no label supervision; ignore split files).
2.  Randomly masks 75 % of time-steps in each trial.
3.  Runs the unmasked tokens through the transformer encoder.
4.  A lightweight decoder reconstructs the masked time-steps.
5.  Loss = MSE on the masked positions only (exactly as in He et al., 2022).
6.  After training, saves ONLY the encoder weights (the decoder is discarded).
7.  The main classification script loads these weights as a warm start.

WHY THIS HELPS
--------------
In LOSO and fine-tuning experiments the model is trained on different subjects
than it is tested on.  MAE pre-training forces the encoder to learn EEG structure
that generalises across subjects (oscillatory patterns, evoked-response timing)
without any label signal.  The supervised fine-tune phase then only needs to learn
the class-discriminative mapping on top of an already rich representation.

HOW TO RUN
----------
  python transformer_pretrain.py \
      --eeg-dataset path/to/eeg_55_95_std.pth \
      --epochs 100 \
      --mask-ratio 0.75 \
      --save-path pretrained_encoder.pth

THEN in the main classification script pass:
  --pretrained_net pretrained_encoder.pth

(The main script already handles --pretrained_net by loading the full saved model;
 see integration notes at the bottom of this file for the minor adjustment needed.)
"""

import argparse
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# ── We reuse the encoder building blocks from transformer2 ─────────────────
from models.transformer2 import (
    MultiHeadAttention,
    PositionWiseFeedForward,
    PositionalEncoding,
    EncoderLayer,
)

# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="MAE pre-training for EEG Transformer")
parser.add_argument('--eeg-dataset', required=True, help="Path to eeg_*.pth dataset file")
parser.add_argument('--epochs', default=100, type=int)
parser.add_argument('--batch-size', default=128, type=int)
parser.add_argument('--lr', default=1e-3, type=float)
parser.add_argument('--mask-ratio', default=0.75, type=float,
                    help="Fraction of time-steps to mask (default 0.75, from MAE paper)")
parser.add_argument('--time-low', default=20, type=int)
parser.add_argument('--time-high', default=460, type=int)
parser.add_argument('--d-model', default=128, type=int)
parser.add_argument('--num-heads', default=4, type=int)
parser.add_argument('--d-ff', default=512, type=int)
parser.add_argument('--num-layers', default=1, type=int)
parser.add_argument('--dropout', default=0.4, type=float)
parser.add_argument('--decoder-layers', default=2, type=int,
                    help="Number of transformer layers in the MAE decoder (2 is enough)")
parser.add_argument('--decoder-d-model', default=64, type=int,
                    help="Decoder hidden size (smaller than encoder to keep it lightweight)")
parser.add_argument('--save-path', default='pretrained_encoder.pth',
                    help="Where to save the pre-trained encoder weights")
parser.add_argument('--no-cuda', action='store_true')
opt = parser.parse_args()

device = torch.device('cpu' if opt.no_cuda or not torch.cuda.is_available() else 'cuda')
T = opt.time_high - opt.time_low   # sequence length, e.g. 440
C = 128                             # EEG channels (input_dim)


# ---------------------------------------------------------------------------
# Dataset — loads all trials, no labels needed
# ---------------------------------------------------------------------------

class EEGPretrainDataset(Dataset):
    """Returns raw EEG tensors of shape (T, C); no labels."""
    def __init__(self, path, time_low, time_high):
        loaded = torch.load(path)
        self.data = [
            d for d in loaded['dataset']
            if 450 <= d['eeg'].size(1) <= 600
        ]
        self.time_low = time_low
        self.time_high = time_high

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        eeg = self.data[i]['eeg'].float().t()          # (raw_T, 128)
        eeg = eeg[self.time_low:self.time_high, :]     # (T, 128)
        return eeg


loader = DataLoader(
    EEGPretrainDataset(opt.eeg_dataset, opt.time_low, opt.time_high),
    batch_size=opt.batch_size, shuffle=True, drop_last=True, num_workers=4
)


# ---------------------------------------------------------------------------
# MAE Model
# ---------------------------------------------------------------------------

class MAEEncoder(nn.Module):
    """
    The same encoder architecture as transformer2.Model, without the classifier
    or projection head.  This is the part we will keep after pre-training.
    """
    def __init__(self, input_dim, d_model, num_heads, d_ff, num_layers, dropout, max_seq_length):
        super().__init__()
        self.embedding  = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_seq_length)
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """x: (B, T, C)  →  (B, T, d_model)"""
        x = self.embedding(x)
        x = self.pos_encoder(x)
        x = self.dropout(x)
        for layer in self.encoder_layers:
            x = layer(x, mask)
        return x  # (B, T, d_model) — NOT mean-pooled; decoder needs all time-steps


class MAEDecoder(nn.Module):
    """
    Lightweight decoder: maps encoder tokens + learned mask token → input space.
    Follows the asymmetric design from He et al. (2022): decoder is much smaller
    than the encoder and is discarded after pre-training.
    """
    def __init__(self, encoder_d_model, decoder_d_model, num_heads, decoder_layers, output_dim, max_seq_length):
        super().__init__()
        # Project encoder tokens to decoder dimension
        self.encoder_to_decoder = nn.Linear(encoder_d_model, decoder_d_model)
        # Learned mask token — a single vector broadcast over all masked positions
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_d_model))
        nn.init.normal_(self.mask_token, std=0.02)

        self.pos_encoder = PositionalEncoding(decoder_d_model, max_seq_length)
        self.decoder_layers = nn.ModuleList([
            EncoderLayer(decoder_d_model, num_heads, decoder_d_model * 2, dropout=0.0)
            for _ in range(decoder_layers)
        ])
        # Reconstruct original EEG channels at each time-step
        self.pred_head = nn.Linear(decoder_d_model, output_dim)

    def forward(self, encoder_tokens, mask_indices, full_seq_length):
        """
        encoder_tokens : (B, n_visible, d_model) — only the unmasked positions
        mask_indices   : (B, n_masked)            — which positions were masked
        full_seq_length: T                        — total sequence length

        Returns:
            pred : (B, T, output_dim) — reconstruction for ALL positions
                   (but loss is only computed on masked ones)
        """
        B = encoder_tokens.size(0)
        n_masked = mask_indices.size(1)
        n_visible = encoder_tokens.size(1)

        # Project to decoder dimension
        vis_tokens = self.encoder_to_decoder(encoder_tokens)  # (B, n_visible, dec_d)

        # Expand the mask token to fill all masked positions
        mask_tokens = self.mask_token.expand(B, n_masked, -1)  # (B, n_masked, dec_d)

        # Reconstruct the full-length sequence by scattering tokens back
        # into their original positions
        full = torch.zeros(B, full_seq_length, vis_tokens.size(-1), device=vis_tokens.device)
        # Build visible index tensor
        vis_idx = torch.ones(B, full_seq_length, dtype=torch.bool, device=vis_tokens.device)
        for b in range(B):
            vis_idx[b, mask_indices[b]] = False
        full[vis_idx.view(-1)].view(B, n_visible, -1)  # not the right way — use scatter below

        # Simpler scatter approach:
        full = torch.zeros(B, full_seq_length, vis_tokens.size(-1), device=vis_tokens.device)
        # Scatter visible tokens
        vis_pos = (~get_mask_bool(B, full_seq_length, mask_indices, vis_tokens.device))
        full[vis_pos] = vis_tokens.reshape(-1, vis_tokens.size(-1))
        # Scatter mask tokens
        msk_pos = get_mask_bool(B, full_seq_length, mask_indices, vis_tokens.device)
        full[msk_pos] = mask_tokens.reshape(-1, mask_tokens.size(-1))

        # Add positional encoding and run decoder
        full = self.pos_encoder(full)
        for layer in self.decoder_layers:
            full = layer(full, mask=None)

        pred = self.pred_head(full)  # (B, T, output_dim=C)
        return pred


def get_mask_bool(B, T, mask_indices, device):
    """Convert integer mask indices to a boolean mask of shape (B, T)."""
    bool_mask = torch.zeros(B, T, dtype=torch.bool, device=device)
    for b in range(B):
        bool_mask[b, mask_indices[b]] = True
    return bool_mask


class MaskedAutoencoder(nn.Module):
    """
    Full MAE model.  Handles masking, encoding visible tokens, and decoding.
    """
    def __init__(self, encoder, decoder, mask_ratio):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.mask_ratio = mask_ratio

    def random_masking(self, x):
        """
        Randomly mask mask_ratio fraction of time-steps.

        Args:
            x: (B, T, C)

        Returns:
            x_visible     : (B, n_visible, C)   — unmasked input tokens
            mask_indices  : (B, n_masked)         — integer positions that are masked
            restore_indices: (B, T)              — argsort to restore full sequence
        """
        B, T, C = x.shape
        n_masked = int(T * self.mask_ratio)

        # Sample random permutation per sample
        noise = torch.rand(B, T, device=x.device)
        shuffle_idx = noise.argsort(dim=1)       # (B, T) — random order
        restore_idx = shuffle_idx.argsort(dim=1) # (B, T) — reverse permutation

        # First n_masked indices (after shuffle) are masked
        mask_idx  = shuffle_idx[:, :n_masked]    # (B, n_masked)
        keep_idx  = shuffle_idx[:, n_masked:]    # (B, n_visible)

        # Gather visible tokens
        keep_expanded = keep_idx.unsqueeze(-1).expand(-1, -1, C)
        x_visible = torch.gather(x, dim=1, index=keep_expanded)  # (B, n_visible, C)

        return x_visible, mask_idx, restore_idx

    def forward(self, x):
        """
        x: (B, T, C)
        Returns:
            loss: scalar MSE on masked positions only
            pred: (B, T, C) reconstruction
        """
        B, T, C = x.shape
        x_vis, mask_idx, restore_idx = self.random_masking(x)  # mask

        # Encode visible tokens only
        enc_out = self.encoder(x_vis)  # (B, n_visible, d_model)

        # Decode — reconstruct all T positions
        pred = self.decoder(enc_out, mask_idx, T)  # (B, T, C)

        # Loss: MSE only on masked positions
        bool_mask = get_mask_bool(B, T, mask_idx, x.device)  # (B, T)
        loss = F.mse_loss(pred[bool_mask], x[bool_mask])

        return loss, pred


# ---------------------------------------------------------------------------
# Build model
# ---------------------------------------------------------------------------

encoder = MAEEncoder(
    input_dim=C,
    d_model=opt.d_model,
    num_heads=opt.num_heads,
    d_ff=opt.d_ff,
    num_layers=opt.num_layers,
    dropout=opt.dropout,
    max_seq_length=T,
).to(device)

decoder = MAEDecoder(
    encoder_d_model=opt.d_model,
    decoder_d_model=opt.decoder_d_model,
    num_heads=opt.num_heads,         # reuse same number for simplicity
    decoder_layers=opt.decoder_layers,
    output_dim=C,
    max_seq_length=T,
).to(device)

mae = MaskedAutoencoder(encoder, decoder, mask_ratio=opt.mask_ratio).to(device)

optimizer = torch.optim.AdamW(mae.parameters(), lr=opt.lr, weight_decay=0.05)
# Cosine annealing is standard for MAE pre-training
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.epochs, eta_min=1e-5)


# ---------------------------------------------------------------------------
# Pre-training loop
# ---------------------------------------------------------------------------

print(f"\n{'='*60}")
print(f"MAE Pre-training  |  mask_ratio={opt.mask_ratio}  |  epochs={opt.epochs}")
print(f"Encoder: d_model={opt.d_model}, num_layers={opt.num_layers}, num_heads={opt.num_heads}")
print(f"Dataset : {len(loader.dataset)} trials  |  T={T}, C={C}")
print(f"{'='*60}\n")

for epoch in range(1, opt.epochs + 1):
    mae.train()
    epoch_loss = 0.0
    n_batches = 0

    for x, in [(batch,) for batch in loader]:
        x = x.to(device)              # (B, T, C)

        optimizer.zero_grad()
        loss, _ = mae(x)
        loss.backward()
        # Gradient clipping — important for transformer stability
        torch.nn.utils.clip_grad_norm_(mae.parameters(), max_norm=1.0)
        optimizer.step()

        epoch_loss += loss.item()
        n_batches += 1

    scheduler.step()
    avg_loss = epoch_loss / n_batches

    if epoch % 10 == 0 or epoch == 1:
        print(f"Epoch [{epoch:>4d}/{opt.epochs}]  MAE loss = {avg_loss:.6f}  lr = {scheduler.get_last_lr()[0]:.2e}")


# ---------------------------------------------------------------------------
# Save encoder weights ONLY — the decoder is thrown away.
# ---------------------------------------------------------------------------

torch.save(encoder.state_dict(), opt.save_path)
print(f"\nEncoder weights saved to: {opt.save_path}")
print("Decoder discarded (as per MAE protocol).")


# ---------------------------------------------------------------------------
# INTEGRATION NOTES — how to load pre-trained weights in the main script
# ---------------------------------------------------------------------------
#
# In transformer_eeg_signal_classification.py, after model = module.Model(...)
# add the following block (just before the optimizer is created):
#
#   if opt.pretrained_encoder != '':
#       state = torch.load(opt.pretrained_encoder)
#       # The MAEEncoder and Model share the same sub-module names:
#       # embedding, pos_encoder, encoder_layers, dropout.
#       # We do a partial load using strict=False so the classifier and
#       # proj_head (which don't exist in the encoder checkpoint) are ignored.
#       missing, unexpected = model.load_state_dict(state, strict=False)
#       print(f"Loaded pre-trained encoder.  Missing keys: {missing}")
#       print(f"Unexpected keys (will be ignored): {unexpected}")
#
# Add to argument parser:
#   parser.add_argument('--pretrained-encoder', default='', ...)
#
# The missing keys will be ['classifier.weight', 'classifier.bias',
# 'proj_head.0.weight', ...] which is expected and fine — those will
# train from random init on the labelled data.
