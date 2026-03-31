"""
transformer_pretrain_mae.py — Masked Autoencoder (MAE) self-supervised
pre-training for the EEG Transformer.

WHAT THIS DOES
--------------
1.  Loads ALL available EEG data (no labels needed; no train/val/test split).
2.  Randomly masks 75% of time-steps in each trial.
3.  Runs only the visible 25% through the transformer encoder.
4.  A lightweight decoder reconstructs ALL time-steps from the visible tokens
    plus a learned [MASK] token placed at each masked position.
5.  Loss = MSE on the masked positions only.
6.  After training, saves ONLY the encoder weights — the decoder is discarded.
7.  The main classification script loads these weights as a warm initialisation
    using --pretrained-encoder.

WHY THIS HELPS
--------------
The encoder is forced to learn structural properties of EEG — oscillatory
patterns, evoked-response timing, channel correlations — without any label
signal.  When you then fine-tune on the labelled multi-subject data, the
encoder already understands EEG, so the classifier only needs to learn the
class-discriminative mapping on top.

The 75% masking ratio is deliberately aggressive.  At 15% the task is too easy
(nearest-neighbour interpolation suffices).  At 75% the model must learn true
temporal structure rather than local smoothing.

HOW TO RUN — PHASE 1 (pre-training, no labels)
-----------------------------------------------
    python transformer_pretrain_mae.py \\
        --eeg-dataset /path/to/eeg_55_95_std.pth \\
        --epochs 100 \\
        --mask-ratio 0.75 \\
        --save-path pretrained_encoder.pth

HOW TO RUN — PHASE 2 (supervised fine-tuning)
----------------------------------------------
Pass the saved checkpoint to the main training script:

    python transformer_eeg_signal_classification.py \\
        --eeg-dataset /path/to/eeg_55_95_std.pth \\
        --splits-path /path/to/splits.pth \\
        --pretrained-encoder pretrained_encoder.pth \\
        [other args as normal]

The main script will load the encoder weights and randomly initialise the
classifier head, then train everything end-to-end.
"""

import argparse
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# Re-use the encoder sub-modules from transformer2 so the architecture
# is guaranteed to be identical to the classification model.
from models.transformer2 import (
    MultiHeadAttention,
    PositionWiseFeedForward,
    PositionalEncoding,
    EncoderLayer,
)

# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="MAE self-supervised pre-training for EEG Transformer")
parser.add_argument('--eeg-dataset',     required=True,          help="Path to eeg_*.pth dataset file")
parser.add_argument('--epochs',          default=100, type=int)
parser.add_argument('--batch-size',      default=128, type=int)
parser.add_argument('--lr',              default=1e-3, type=float)
parser.add_argument('--mask-ratio',      default=0.75, type=float,
                    help="Fraction of time-steps to mask (default 0.75)")
parser.add_argument('--time-low',        default=20,  type=int)
parser.add_argument('--time-high',       default=460, type=int)
parser.add_argument('--d-model',         default=128, type=int)
parser.add_argument('--num-heads',       default=4,   type=int)
parser.add_argument('--d-ff',            default=512, type=int)
parser.add_argument('--num-layers',      default=1,   type=int)
parser.add_argument('--dropout',         default=0.4, type=float)
parser.add_argument('--decoder-layers',  default=2,   type=int,
                    help="Transformer layers in the MAE decoder (2 is sufficient)")
parser.add_argument('--decoder-d-model', default=64,  type=int,
                    help="Decoder hidden size — smaller than encoder by design")
parser.add_argument('--save-path',       default='pretrained_encoder.pth',
                    help="Where to save the pre-trained encoder state_dict")
parser.add_argument('--no-cuda',         action='store_true')
opt = parser.parse_args()

device = torch.device('cpu' if opt.no_cuda or not torch.cuda.is_available() else 'cuda')
T = opt.time_high - opt.time_low   # sequence length, e.g. 440
C = 128                             # EEG channels (input_dim)


# ---------------------------------------------------------------------------
# Dataset  —  loads every trial, ignores labels
# ---------------------------------------------------------------------------

class EEGPretrainDataset(Dataset):
    """Returns raw EEG tensors of shape (T, C).  No labels."""

    def __init__(self, path, time_low, time_high):
        loaded = torch.load(path)
        self.data = [
            d for d in loaded['dataset']
            if 450 <= d['eeg'].size(1) <= 600
        ]
        self.time_low  = time_low
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
# MAE Encoder  —  identical architecture to transformer2.Model's encoder
# ---------------------------------------------------------------------------

class MAEEncoder(nn.Module):
    """
    Transformer encoder — same architecture as transformer2.Model, but without
    the classifier or projection head.  This is the part kept after pre-training.
    """
    def __init__(self, input_dim, d_model, num_heads, d_ff, num_layers, dropout, max_seq_length):
        super().__init__()
        self.embedding     = nn.Linear(input_dim, d_model)
        self.pos_encoder   = PositionalEncoding(d_model, max_seq_length)
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """x : (B, T_visible, C)  →  (B, T_visible, d_model)"""
        x = self.embedding(x)
        x = self.pos_encoder(x)
        x = self.dropout(x)
        for layer in self.encoder_layers:
            x = layer(x, mask)
        return x  # NOT mean-pooled — decoder needs all time-steps


# ---------------------------------------------------------------------------
# MAE Decoder  —  lightweight, discarded after pre-training
# ---------------------------------------------------------------------------

class MAEDecoder(nn.Module):
    """
    Small transformer decoder.  Takes visible encoder tokens + a shared
    learned [MASK] token and reconstructs the full T-length sequence.

    Design follows He et al. (2022): decoder is much smaller than the encoder
    so that representational capacity is concentrated in the encoder.
    """
    def __init__(self, encoder_d_model, decoder_d_model, num_heads,
                 decoder_layers, output_dim, max_seq_length):
        super().__init__()
        # Project encoder tokens into decoder's smaller dimension
        self.encoder_to_decoder = nn.Linear(encoder_d_model, decoder_d_model)
        # Single shared [MASK] token, broadcast over all masked positions
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_d_model))
        nn.init.normal_(self.mask_token, std=0.02)

        self.pos_encoder = PositionalEncoding(decoder_d_model, max_seq_length)
        self.decoder_layers = nn.ModuleList([
            EncoderLayer(decoder_d_model, num_heads,
                         decoder_d_model * 2, dropout=0.0)
            for _ in range(decoder_layers)
        ])
        # Predict original EEG channels at each time-step
        self.pred_head = nn.Linear(decoder_d_model, output_dim)

    def forward(self, encoder_tokens, keep_idx, mask_idx, T):
        """
        Args:
            encoder_tokens : (B, n_visible, encoder_d_model)
            keep_idx       : (B, n_visible) — which positions were NOT masked
            mask_idx       : (B, n_masked)  — which positions WERE masked
            T              : total sequence length
        Returns:
            pred : (B, T, output_dim) — reconstruction for all positions
        """
        B          = encoder_tokens.size(0)
        n_visible  = keep_idx.size(1)
        n_masked   = mask_idx.size(1)
        dec_d      = self.encoder_to_decoder.out_features

        # Project visible tokens to decoder dimension
        vis_tokens = self.encoder_to_decoder(encoder_tokens)  # (B, n_vis, dec_d)

        # Expand the shared mask token to fill all masked positions
        mask_tokens = self.mask_token.expand(B, n_masked, -1)  # (B, n_masked, dec_d)

        # Reconstruct the full sequence by scattering tokens back into their
        # original positions using torch.scatter_.
        full = torch.zeros(B, T, dec_d, device=encoder_tokens.device)

        vis_idx_exp  = keep_idx.unsqueeze(-1).expand(-1, -1, dec_d)  # (B, n_vis,   dec_d)
        msk_idx_exp  = mask_idx.unsqueeze(-1).expand(-1, -1, dec_d)  # (B, n_masked, dec_d)

        full.scatter_(1, vis_idx_exp, vis_tokens)    # place visible tokens
        full.scatter_(1, msk_idx_exp, mask_tokens)   # place mask tokens

        # Add positional encoding and run decoder transformer
        full = self.pos_encoder(full)
        for layer in self.decoder_layers:
            full = layer(full, mask=None)

        pred = self.pred_head(full)  # (B, T, C)
        return pred


# ---------------------------------------------------------------------------
# Full MAE  —  handles masking, encoding, decoding, and loss
# ---------------------------------------------------------------------------

class MaskedAutoencoder(nn.Module):
    def __init__(self, encoder, decoder, mask_ratio):
        super().__init__()
        self.encoder    = encoder
        self.decoder    = decoder
        self.mask_ratio = mask_ratio

    def random_masking(self, x):
        """
        Randomly mask mask_ratio fraction of time-steps per trial.

        Args:
            x : (B, T, C)
        Returns:
            x_visible : (B, n_visible, C) — only the unmasked input tokens
            keep_idx  : (B, n_visible)    — positions of unmasked tokens
            mask_idx  : (B, n_masked)     — positions of masked tokens
        """
        B, T, C = x.shape
        n_masked  = int(T * self.mask_ratio)
        n_visible = T - n_masked

        # Random permutation per sample in the batch
        noise = torch.rand(B, T, device=x.device)
        shuffled = noise.argsort(dim=1)           # (B, T)

        # First n_masked positions after shuffle are masked
        mask_idx = shuffled[:, :n_masked]         # (B, n_masked)
        keep_idx = shuffled[:, n_masked:]         # (B, n_visible)

        # Sort keep_idx so visible tokens are in temporal order
        # (helps positional encoding; not strictly required)
        keep_idx, _ = keep_idx.sort(dim=1)

        # Gather the visible tokens
        keep_expanded = keep_idx.unsqueeze(-1).expand(-1, -1, C)  # (B, n_vis, C)
        x_visible = torch.gather(x, dim=1, index=keep_expanded)   # (B, n_vis, C)

        return x_visible, keep_idx, mask_idx

    def forward(self, x):
        """
        Args:
            x : (B, T, C)
        Returns:
            loss : scalar MSE on masked positions only
            pred : (B, T, C) full reconstruction (for inspection)
        """
        B, T, C = x.shape

        x_visible, keep_idx, mask_idx = self.random_masking(x)

        # Encode only visible tokens
        enc_out = self.encoder(x_visible)                               # (B, n_vis, d_model)

        # Reconstruct all T positions
        pred = self.decoder(enc_out, keep_idx, mask_idx, T)             # (B, T, C)

        # Boolean mask for masked positions
        bool_mask = torch.zeros(B, T, dtype=torch.bool, device=x.device)
        bool_mask.scatter_(1, mask_idx, True)                           # (B, T)

        # MSE loss only on masked positions
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
    num_heads=opt.num_heads,
    decoder_layers=opt.decoder_layers,
    output_dim=C,
    max_seq_length=T,
).to(device)

mae = MaskedAutoencoder(encoder, decoder, mask_ratio=opt.mask_ratio).to(device)

optimizer = torch.optim.AdamW(mae.parameters(), lr=opt.lr, weight_decay=0.05)
# Cosine annealing is standard for MAE pre-training
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=opt.epochs, eta_min=1e-5
)

n_params_enc = sum(p.numel() for p in encoder.parameters())
n_params_dec = sum(p.numel() for p in decoder.parameters())

print(f"\n{'='*60}")
print(f"MAE Pre-training")
print(f"  mask_ratio   = {opt.mask_ratio}  ({int(opt.mask_ratio*100)}% of {T} time-steps masked)")
print(f"  epochs       = {opt.epochs}")
print(f"  encoder      = {n_params_enc:,} params  (d_model={opt.d_model}, layers={opt.num_layers})")
print(f"  decoder      = {n_params_dec:,} params  (d_model={opt.decoder_d_model}, layers={opt.decoder_layers})")
print(f"  dataset      = {len(loader.dataset):,} trials  |  T={T}, C={C}")
print(f"  save_path    = {opt.save_path}")
print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Pre-training loop
# ---------------------------------------------------------------------------

for epoch in range(1, opt.epochs + 1):
    mae.train()
    epoch_loss = 0.0
    n_batches  = 0

    for x in loader:
        x = x.to(device)        # (B, T, C)

        optimizer.zero_grad()
        loss, _ = mae(x)
        loss.backward()
        # Gradient clipping prevents rare instability in the transformer layers
        torch.nn.utils.clip_grad_norm_(mae.parameters(), max_norm=1.0)
        optimizer.step()

        epoch_loss += loss.item()
        n_batches  += 1

    scheduler.step()
    avg_loss = epoch_loss / n_batches

    if epoch % 10 == 0 or epoch == 1:
        lr_now = scheduler.get_last_lr()[0]
        print(f"Epoch [{epoch:>4d}/{opt.epochs}]  "
              f"MAE loss = {avg_loss:.6f}  "
              f"lr = {lr_now:.2e}")


# ---------------------------------------------------------------------------
# Save encoder weights only — the decoder is discarded
# ---------------------------------------------------------------------------

torch.save(encoder.state_dict(), opt.save_path)
print(f"\nEncoder weights saved → {opt.save_path}")
print("Decoder discarded (MAE protocol: decoder is a training tool, not a result).")
print("\nNext step: pass this file to the main training script with:")
print(f"  --pretrained-encoder {opt.save_path}")
