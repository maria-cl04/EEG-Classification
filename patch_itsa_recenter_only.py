"""
patch_itsa_recenter_only.py
---------------------------
Replaces the back-projected recolouring filter for the held-out LOSO subject
with a recentering-only filter (M_inv_sqrt_[subject]).

This is the correct fallback when the 40-class SVD rotation is too noisy to
produce a reliable recolouring direction. The recentering step alone (Adaptive M)
is stable and gives most of ITSA's benefit in high-class-count LOSO settings.

Usage:
    python patch_itsa_recenter_only.py \
        --input   itsa_loso_subject1_light_patched.pth \
        --output  itsa_loso_subject1_light_recenter.pth \
        --subject 1
"""

import argparse
import numpy as np
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--input",   required=True, help="Path to saved ITSA .pth file")
parser.add_argument("--output",  required=True, help="Where to save the patched file")
parser.add_argument("--subject", required=True, type=int, help="Held-out subject ID (e.g. 1)")
args = parser.parse_args()

# ── Load ──────────────────────────────────────────────────────────────────────
print(f"Loading {args.input} ...")
itsa_core = torch.load(args.input, weights_only=False)

# ── Check M_inv_sqrt_ exists for this subject ─────────────────────────────────
if args.subject not in itsa_core.M_inv_sqrt_:
    raise KeyError(
        f"M_inv_sqrt_ does not contain subject {args.subject}. "
        f"Keys present: {list(itsa_core.M_inv_sqrt_.keys())}"
    )

M_inv_sqrt = itsa_core.M_inv_sqrt_[args.subject]

# ── Report before state ───────────────────────────────────────────────────────
A_old = itsa_core.A_filters_.get(args.subject)
if A_old is not None:
    print(f"Old A_filters_[{args.subject}] norm : {np.linalg.norm(A_old):.4f}")
else:
    print(f"A_filters_[{args.subject}] was not set — will be created.")

print(f"M_inv_sqrt_[{args.subject}] norm  : {np.linalg.norm(M_inv_sqrt):.4f}  (identity scale = {np.sqrt(M_inv_sqrt.shape[0]):.2f})")

# ── Patch: replace with recentering-only filter ───────────────────────────────
if itsa_core.A_filters_ is None:
    itsa_core.A_filters_ = {}

itsa_core.A_filters_[args.subject] = M_inv_sqrt.copy()
itsa_core._filters_cache.clear()

print(f"Replaced A_filters_[{args.subject}] with M_inv_sqrt_[{args.subject}]  ✓")

# ── Save ──────────────────────────────────────────────────────────────────────
torch.save(itsa_core, args.output)
print(f"Saved → {args.output}")
print("Done. Point --pretrained_itsa at the new file and rerun training.")
