"""
patch_itsa_recolour_global.py
-----------------------------
Patches the spatial filter for the held-out LOSO subject in a saved ITSA .pth
file, replacing it with one computed by recolouring toward reference_G_ (the
global Riemannian mean of all recentered training covariances).

This avoids the two failure modes seen with simpler approaches:
  - Recentering-only (M^{-1/2}):  whitens signals → chance-level accuracy
  - Noisy 40-class rotation:       exploded norm / wrong direction → TeL diverges

reference_G_ is already stored in the ITSA file. The only things computed fresh
here are M_inv_sqrt_[subject] (if absent from an old export_light file) and
Cbar_rec (the Riemannian mean of the test subject's recentered covariances).

Usage:
    python patch_itsa_recolour_global.py \
        --itsa     itsa_loso_subject1_light.pth \
        --output   itsa_loso_subject1_light_recolour.pth \
        --dataset  path/to/eeg_55_95_std.pth \
        --splits   path/to/block_splits_LOSO_subject1.pth \
        --subject  1 \
        --time_low  20 \
        --time_high 460
"""

import argparse
import numpy as np
import torch

from pyriemann.utils.mean import mean_riemann, mean_logeuclid
from pyriemann.utils.base import invsqrtm, sqrtm

parser = argparse.ArgumentParser()
parser.add_argument("--itsa",     required=True,  help="Path to saved ITSA .pth file")
parser.add_argument("--output",   required=True,  help="Where to save the patched file")
parser.add_argument("--dataset",  required=True,  help="Path to EEG dataset .pth file")
parser.add_argument("--splits",   required=True,  help="Path to LOSO splits .pth file")
parser.add_argument("--subject",  required=True,  type=int, help="Held-out subject ID (e.g. 1)")
parser.add_argument("--split_num", default=0,     type=int, help="Split index (default 0)")
parser.add_argument("--time_low",  default=20,    type=int, help="Time window start (default 20)")
parser.add_argument("--time_high", default=460,   type=int, help="Time window end (default 460)")
args = parser.parse_args()

# ── Helpers (mirrors EEGDataset / Splitter / cov_from_signal_torch) ───────────

def to_spd_np(A, eps=1e-10):
    A = 0.5 * (A + A.T)
    w, V = np.linalg.eigh(A)
    w = np.clip(w, eps, None)
    return (V * w) @ V.T

def cov_from_signal_np(eeg_2d, eps=1e-4):
    """eeg_2d: (T, C) numpy array → (C, C) SPD covariance"""
    x = eeg_2d - eeg_2d.mean(axis=0, keepdims=True)
    T = max(1, x.shape[0] - 1)
    C = (x.T @ x) / T
    C += eps * np.eye(C.shape[1])
    return to_spd_np(C)

def load_test_covs(dataset_path, splits_path, subject_id, split_num,
                   time_low, time_high):
    """Load covariance matrices for the held-out test subject."""
    print("Loading dataset ...")
    raw = torch.load(dataset_path, weights_only=False)
    data = raw["dataset"]

    print("Loading splits ...")
    splits = torch.load(splits_path, weights_only=False)
    test_idx = splits["splits"][split_num]["test"]

    # Same length filter as Splitter
    test_idx = [i for i in test_idx if 450 <= data[i]["eeg"].size(1) <= 600]

    covs = []
    for i in test_idx:
        subj = int(data[i]["subject"])
        if subj != subject_id:
            continue
        eeg = data[i]["eeg"].float().numpy().T          # (T_raw, C)
        eeg = eeg[time_low:time_high, :]                # (T_window, C)
        covs.append(cov_from_signal_np(eeg))

    if len(covs) == 0:
        raise ValueError(
            f"No test samples found for subject {subject_id}. "
            f"Check --subject, --splits, and length filter."
        )
    print(f"Loaded {len(covs)} covariance matrices for subject {subject_id}.")
    return np.stack(covs, axis=0)   # (N, C, C)

# ── Load ITSA ─────────────────────────────────────────────────────────────────
print(f"\nLoading ITSA from {args.itsa} ...")
itsa = torch.load(args.itsa, weights_only=False)

assert itsa.reference_G_ is not None, \
    "reference_G_ is None in the saved file — was fit() called properly?"

eps = getattr(itsa, "subject_eps", 1e-10)
mean_tol = getattr(itsa, "mean_tol", 1e-6)
mean_maxiter = getattr(itsa, "mean_maxiter", 50)
unit_trace = getattr(itsa, "unit_trace_per_subject", True)

# ── Load test covariances ─────────────────────────────────────────────────────
covs_test = load_test_covs(
    args.dataset, args.splits, args.subject, args.split_num,
    args.time_low, args.time_high,
)

# ── Compute (or recover) M_inv_sqrt_ for the test subject ─────────────────────
if args.subject in itsa.M_inv_sqrt_:
    print(f"M_inv_sqrt_[{args.subject}] found in saved file — reusing.")
    M_inv_sqrt = itsa.M_inv_sqrt_[args.subject]
else:
    print(f"M_inv_sqrt_[{args.subject}] not in saved file — computing from test covs.")
    covs_norm = covs_test.copy()
    if unit_trace:
        tr = np.trace(covs_norm, axis1=1, axis2=2).reshape(-1, 1, 1)
        covs_norm = covs_norm / np.maximum(tr, 1e-12)
    M_init = mean_logeuclid(covs_norm)
    M = mean_riemann(covs_norm, init=M_init, tol=mean_tol, maxiter=mean_maxiter)
    M_inv_sqrt = invsqrtm(M)
    itsa.M_inv_sqrt_[args.subject] = M_inv_sqrt
    print(f"  Computed and stored M_inv_sqrt_[{args.subject}].")

# ── Recenter test covariances ─────────────────────────────────────────────────
print("Recentering test covariances ...")
covs_norm = covs_test.copy()
if unit_trace:
    tr = np.trace(covs_norm, axis1=1, axis2=2).reshape(-1, 1, 1)
    covs_norm = covs_norm / np.maximum(tr, 1e-12)

covs_rec = np.stack([
    to_spd_np(M_inv_sqrt @ to_spd_np(C, eps) @ M_inv_sqrt, eps)
    for C in covs_norm
])

# ── Compute Cbar_rec ──────────────────────────────────────────────────────────
print("Computing Riemannian mean of recentered covariances ...")
Cbar_rec = to_spd_np(
    mean_riemann(
        covs_rec,
        init=mean_logeuclid(covs_rec),
        tol=mean_tol,
        maxiter=mean_maxiter,
    ),
    eps=eps,
)

# ── Build the new filter ──────────────────────────────────────────────────────
G_target = to_spd_np(itsa.reference_G_, eps=eps)

Cbar_rec_isqrt = invsqrtm(Cbar_rec)
G_target_sqrt  = sqrtm(G_target)
W = Cbar_rec_isqrt @ G_target_sqrt
A_new = M_inv_sqrt @ W

# ── Diagnostics ───────────────────────────────────────────────────────────────
n_ch = A_new.shape[0]
identity_norm = np.sqrt(n_ch)
new_norm = np.linalg.norm(A_new)

old_A = itsa.A_filters_.get(args.subject) if itsa.A_filters_ else None
if old_A is not None:
    print(f"\nOld filter norm : {np.linalg.norm(old_A):.4f}")
print(f"New filter norm : {new_norm:.4f}  (identity scale = {identity_norm:.4f})")

if new_norm > identity_norm * 5:
    print("WARNING: new filter norm is still >5× identity scale. "
          "Check that reference_G_ is well-conditioned.")
else:
    print("Filter norm looks healthy ✓")

# ── Patch and save ────────────────────────────────────────────────────────────
if itsa.A_filters_ is None:
    itsa.A_filters_ = {}
itsa.A_filters_[args.subject] = A_new
itsa._filters_cache.clear()

torch.save(itsa, args.output)
print(f"\nSaved patched file → {args.output}")
print("Done. Point --pretrained_itsa at the new file and rerun training.")
