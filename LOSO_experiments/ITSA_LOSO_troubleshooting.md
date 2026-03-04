# Troubleshooting: ITSA LOSO Alignment Fixes

**Date:** 2026-03-04
**Files changed:** `ITSA.py`, `transformer_eeg_signal_classification.py`

---

## Background

The experiment uses ITSA (Individual Tangent Space Alignment, arXiv:2508.08216) as a
pre-alignment strategy in a Leave-One-Subject-Out (LOSO) setup. Subject 1 is held out for
testing; subjects 2–6 are used for training.

ITSA aligns EEG covariance matrices across subjects through three sequential steps:
1. **Recentering** — each subject's covariance matrices are recentered to the identity using their own Log-Euclidean mean.
2. **Rescaling** — tangent-space features are normalised so the average Frobenius norm equals 1.
3. **Rotation** — a supervised SVD-based rotation aligns the test subject's class-wise anchor points to the training space, using a nested 2-fold CV to prevent data leakage.

Several bugs meant the held-out subject was receiving none of this alignment at test time, and test accuracy was being reported incorrectly.

---

## Bug 1 — Held-out subject had no spatial filter (identity fallback)

### File: `ITSA.py`

### What was wrong

`_derive_subject_filters()` — the method that builds the signal-domain spatial filter
`A_filters_[s]` for each subject — iterates only over **training subjects**. Subject 1 (the
held-out test subject) was never processed, so `A_filters_.get(1)` returned `None`.

In `transform_signals()`, a `None` filter silently falls back to the identity matrix:

```python
if A_np is None:
    A_base = torch.eye(C, device=device, dtype=torch.float32)
```

This means **ITSA did nothing for the test set**. The model saw raw, unaligned signals
from subject 1 even when ITSA was enabled.

### Fix

Added a new method `adapt_loso_test_subject()` that must be called after `fit()`, passing
only the test subject's data. It:

1. Computes `M_inv_sqrt_[test_s]` if not already present (recentering basis).
2. Calls the corrected `transform_test_loso()` to obtain fully aligned tangent features (all 3 steps).
3. Back-projects the mean rotated feature to SPD space to obtain `G_target`.
4. Derives the spatial filter using the same recolouring logic as training subjects:
   `A = M^{-1/2} @ W`, where `W` maps the subject's recentered mean covariance toward `G_target`.
5. Stores the result in `A_filters_[test_s]` and clears the filter cache.

After this call, `transform_signals()` works correctly for the test subject.

---

## Bug 2 — Wrong rotation algorithm (Procrustes instead of SVD)

### File: `ITSA.py`, method `transform_test_loso()`

### What was wrong

The paper (Eq. 9–11) defines the rotation via SVD on a cross-product matrix:

```
C_TC = C̄_train^T @ C̄_calib        # cross-product matrix
C_TC = U D V^T                      # SVD decomposition
Ĉ_ROT = Ũ Ṽ^T @ C̃_SC_eval        # rotation applied to evaluation subset
```

The code used `scipy.linalg.orthogonal_procrustes` instead, which solves a different
optimisation problem (minimising Frobenius distance between two matrices). These are not
equivalent, and Procrustes does not implement the paper's truncated-variance SVD step.

Additionally, the anchor matrices were built from **all K training classes** vs only the
classes present in the calibration fold — mismatched row counts that caused the Procrustes
solver to silently produce a non-square result and fall back to the identity.

### Fix

Replaced the Procrustes calls in both folds with the correct SVD pipeline:

```python
# Only use classes present in BOTH the training set and this calib fold
valid_keys = sorted(
    set(self.mu_global_.keys()) &
    {int(k) for k in np.unique(y_calib)}
)

train_anchors = np.stack([self.mu_global_[k] for k in valid_keys])            # (K', d)
calib_anchors = np.stack([Z_calib[y_calib == k].mean(0) for k in valid_keys]) # (K', d)

C_TC = train_anchors.T @ calib_anchors          # (d, d)
U, sv, Vt = np.linalg.svd(C_TC, full_matrices=False)

# Truncate to retain 99.9% of variance (paper's N_v selection)
cum_var = np.cumsum(sv ** 2) / (np.sum(sv ** 2) + 1e-12)
n_v = int(np.searchsorted(cum_var, 0.999) + 1)
R = U[:, :n_v] @ Vt[:n_v, :]                   # (d, d)

Z_rot_combined[eval_slice] = (R @ Z_eval.T).T
```

The class intersection ensures anchor matrices always have the same number of rows.
If fewer than 2 classes appear in a calibration fold, the fold falls back to no rotation
rather than crashing or silently using a garbage matrix.

---

## Bug 3 — No `from_dataset_loso()` constructor on `ITSAIntegrator`

### File: `ITSA.py`

### What was wrong

`ITSAIntegrator.from_dataset()` calls `ITSA.fit()`, which only processes training subjects.
There was no constructor that subsequently called `adapt_loso_test_subject()`. The main
script was already calling `from_dataset_loso()` (line 177), which did not exist, causing an
`AttributeError` at startup.

### Fix

Added `ITSAIntegrator.from_dataset_loso()`, which:

1. Loads all splits identically to `from_dataset()`.
2. Calls `ITSA.fit()` on training subjects.
3. Calls `itsa.adapt_loso_test_subject()` on the test-split indices, passing their true labels for the supervised rotation step.

This is the correct entry point for all LOSO experiments.

---

## Bug 4 — Test accuracy silently wrong every epoch

### File: `transformer_eeg_signal_classification.py`, lines 317–319

### What was wrong

After the post-loop test block computed predictions, it calculated accuracy using the wrong
variable:

```python
_, pref = output.data.max(1)
correct = pred.eq(test_target_full.data).sum().item()  # ← pred, not pref
```

`pred` is defined inside the `train`/`val` batch loop and holds predictions from the **last
training or validation batch**, not from the test output. Every epoch the reported test
accuracy was actually the accuracy of the final training batch re-evaluated against the test
labels — a meaningless number.

### Fix

This entire post-loop accumulation block was removed (see Bug 5 below), which eliminates
the variable entirely. Test batches are now evaluated inline like train and val, using `pred`
consistently.

---

## Bug 5 — Unnecessary test-batch accumulation loop

### File: `transformer_eeg_signal_classification.py`, lines 255–321

### What was wrong

The original design accumulated all test batches into memory before applying a single
"LOSO rotation" pass at the end. This was based on an earlier (incorrect) assumption that
the rotation needed to see all test data at once.

After the ITSA.py fixes, the rotation is computed once during `from_dataset_loso()` and
stored as a spatial filter in `A_filters_[test_subject]`. At inference time, `transform_batch()`
applies this pre-computed filter per sample — no different from how training subjects are
handled. There is no reason to accumulate batches.

The accumulation code also introduced a `continue` statement that skipped the model
forward pass for test batches inside the main loop, meaning `counts["test"]` was 0 until
after the post-loop block ran. If ITSA was disabled (`--itsa_off`), the post-loop block was
skipped entirely, `counts["test"]` stayed 0, and the epoch-end print divided by zero.

### Fix

- Removed the `test_inputs_batch / test_targets_batch / test_subjects_batch` initialisation.
- Removed the `continue` and the entire post-loop block.
- Replaced the 3-branch ITSA condition (`train` / `test` / `val`) with a 2-branch one:
  `train` uses augmented mode; `val` and `test` both use deterministic mode via `transform_batch()`.

The loop is now uniform across all splits. LOSO alignment is fully pre-computed at fit time.

---

## Summary table

| # | Bug | File | Lines | Impact |
|---|-----|------|-------|--------|
| 1 | Held-out subject gets identity filter (no alignment) | `ITSA.py` | `_derive_subject_filters`, new `adapt_loso_test_subject` | ITSA has zero effect on test set |
| 2 | Procrustes used instead of SVD; anchor row mismatch | `ITSA.py` | `transform_test_loso` lines 250–306 | Wrong rotation, often falls back to identity |
| 3 | `from_dataset_loso()` missing, causes `AttributeError` | `ITSA.py` | New classmethod on `ITSAIntegrator` | Script crashes on startup in LOSO mode |
| 4 | `pred` used instead of `pref` in test accuracy | `transformer_eeg_signal_classification.py` | Line 318 | Reported test accuracy is last train-batch accuracy |
| 5 | Test accumulation loop unnecessary and fragile | `transformer_eeg_signal_classification.py` | Lines 255–321 | Division-by-zero when `itsa_off=True`; stale `pred` variable |
