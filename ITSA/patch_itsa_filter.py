"""
patch_itsa_filter.py
--------------------
Fixes the exploded spatial filter norm for the held-out LOSO subject
in a saved ITSA .pth file, without rerunning the 2-hour fit.

Usage:
    python patch_itsa_filter.py \
        --input  itsa_loso_subject1_light.pth \
        --output itsa_loso_subject1_light_patched.pth \
        --subject 1

The script:
  1. Loads the saved ITSA object.
  2. Reads A_filters_[subject] and reports its current norm.
  3. Normalises it to identity scale (sqrt(n_channels)).
  4. Saves the patched object to --output (original is never overwritten).
"""

import argparse
import numpy as np
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--input",   required=True,  help="Path to the saved ITSA .pth file")
parser.add_argument("--output",  required=True,  help="Where to save the patched file")
parser.add_argument("--subject", required=True,  type=int, help="Held-out subject ID (e.g. 1)")
args = parser.parse_args()

# ── Load ──────────────────────────────────────────────────────────────────────
print(f"Loading {args.input} ...")
itsa_core = torch.load(args.input, weights_only=False)

A_filters = itsa_core.A_filters_
if A_filters is None or args.subject not in A_filters:
    raise KeyError(
        f"Subject {args.subject} not found in A_filters_. "
        f"Keys present: {list(A_filters.keys()) if A_filters else 'None'}"
    )

# ── Diagnose ──────────────────────────────────────────────────────────────────
A = A_filters[args.subject]
n_ch = A.shape[0]
current_norm  = np.linalg.norm(A)
identity_norm = np.sqrt(n_ch)

print(f"Subject {args.subject} filter shape : {A.shape}")
print(f"Current norm   : {current_norm:.4f}")
print(f"Identity scale : {identity_norm:.4f}  (= sqrt({n_ch}))")
print(f"Ratio          : {current_norm / identity_norm:.1f}x")

# ── Patch ─────────────────────────────────────────────────────────────────────
if current_norm <= identity_norm * 2:
    print("Norm is already within acceptable range — no patch needed.")
else:
    A_patched = A * (identity_norm / current_norm)
    new_norm  = np.linalg.norm(A_patched)
    itsa_core.A_filters_[args.subject] = A_patched
    itsa_core._filters_cache.clear()          # clear any cached tensors
    print(f"Patched norm   : {new_norm:.4f}  ✓")

# ── Save ──────────────────────────────────────────────────────────────────────
torch.save(itsa_core, args.output)
print(f"Saved patched file → {args.output}")
print("Done. Point --pretrained_itsa at the new file and rerun training.")
