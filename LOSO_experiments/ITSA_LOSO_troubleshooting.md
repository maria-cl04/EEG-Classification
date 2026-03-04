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

## Bug 6 — Spatial filter norm explosion (~640× identity scale)

### File: `ITSA.py`, method `adapt_loso_test_subject()`

### What was wrong

After the fixes in Bugs 1–5, subject 1 received a spatial filter for the first time.
However, the filter norm was **7220.95** against an identity scale of **11.31**
(= √128 for 128 EEG channels) — a factor of ~640× too large.

The explosion originated in the back-projection pipeline inside `adapt_loso_test_subject()`.
The method called `transform_test_loso()` to obtain rotated tangent-space features, took
their mean, and inverted the StandardScaler and TangentSpace transforms to recover a target
SPD matrix `G_target`. With 40 classes and only `N_test / 2` samples per calibration fold,
many classes had 0–1 samples in a given fold. The `valid_keys` intersection (classes present
in both training and calibration) was typically 10–15 out of 40. The SVD rotation estimated
from so few anchor points is numerically unreliable, and the resulting mean rotated feature
`mu_rot` landed at an extreme point in tangent space. Inverting the scaler and tangent
projection at that extreme point produced a `G_target` with very large eigenvalues, whose
matrix square root `G_target_sqrt` then inflated the recolouring matrix `W` and, through it,
the entire spatial filter `A = M^{-1/2} @ W`.

At inference time, `transform_signals()` applies this filter as `x @ A^T`. With ‖A‖ ≈ 7221,
every signal was amplified by ~640×. The model's softmax received inputs with logits
hundreds of times larger than during training, immediately saturating to near-zero entropy
and pushing TeL to ~9.7 by epoch 10 (TeL/VL ratio ≈ 5.75), worse than before any ITSA
fix was applied.

**Diagnostic added to `transformer_eeg_signal_classification.py`** (run once after
`from_dataset_loso()` to catch this class of issue early):

```python
test_subj_id = int(opt.splits_path.split("subject")[-1].split(".")[0])
A_test = itsa._itsa.A_filters_.get(test_subj_id)
if A_test is not None:
    identity_norm = np.sqrt(A_test.shape[0])
    print(f"[ITSA] Subject {test_subj_id} filter norm: {np.linalg.norm(A_test):.2f}  "
          f"(identity scale = {identity_norm:.2f})")
```

### Fix

The root cause (unreliable back-projection) was addressed separately in Bug 7.
As an immediate patch for any already-saved `.pth` files, a standalone script
`patch_itsa_filter.py` was written that loads the saved object, normalises
`A_filters_[subject]` to identity scale, clears the filter cache, and saves a new file
without touching the original.

```python
# Core logic of the patch
A = itsa_core.A_filters_[subject]
n_ch = A.shape[0]
target_norm = np.sqrt(n_ch)
A_patched = A * (target_norm / np.linalg.norm(A))
itsa_core.A_filters_[subject] = A_patched
itsa_core._filters_cache.clear()
```

---

## Bug 7 — Back-projection of 40-class rotation produces wrong filter direction

### File: `ITSA.py`, method `adapt_loso_test_subject()`

### What was wrong

Even with the norm clipped to identity scale (Bug 6 fix), the *direction* of the filter
remained wrong. At epoch 10 after the norm patch, TeL was **9.66** and TeA was **0.0245**
(below chance level of 0.025), with a TeL/VL ratio of **5.75**. The filter was actively
steering subject 1's signals away from the training distribution.

The fundamental problem is that the back-projection strategy — estimating `G_target` by
inverting the tangent-space pipeline from a mean rotated feature — was designed for the
training-subject case, where each subject has many hundreds of samples and the rotation
is estimated from well-populated class centroids across all 40 classes. For the test subject
in a LOSO setup, the calibration fold has roughly `N_test / 2` samples split across 40
classes: on average fewer than 5 samples per class. The resulting anchor points are too
noisy to produce a meaningful rotation, and the mean of those rotated features is not a
reliable summary of where the subject's distribution should land in the training space.
Normalising the filter norm (Bug 6) corrects the scale but does not correct the direction
— the filter still points the signals toward a wrong region of feature space, just without
the catastrophic amplitude amplification.

The paper's ablation study is informative here: the "Adaptive M" baseline (recentering
only, no rescaling or rotation) achieves 57–59% F1 vs ITSA's 61% in their 2-class
problem. The rotation step contributes ~2% when class centroids are well estimated from
many samples. With 40 classes and very few test samples per class, the rotation estimate
degrades faster than the gain it provides, making recentering-only the better choice.

### Fix

Replaced the entire back-projection block in `adapt_loso_test_subject()` with a single
assignment that uses `M_inv_sqrt_[test_s]` directly as the spatial filter:

```python
# Recentering-only filter: A = M^{-1/2}
# The full back-projection (recolour via 40-class SVD rotation) is too
# noisy for high-class-count LOSO — the rotation estimate from N_test/2
# samples across 40 classes is unreliable and produces out-of-distribution
# directions. Recentering alone (Adaptive M) is stable and contributes
# most of ITSA's benefit. The rotation step can only help if class
# centroids are well-estimated, which requires far more test samples per
# class than are available here.
if self.A_filters_ is None:
    self.A_filters_ = {}
self.A_filters_[test_s] = self.M_inv_sqrt_[test_s].copy()
self._filters_cache.clear()
```

`M_inv_sqrt_[test_s]` is computed from the test subject's own covariance matrices
exclusively, so its norm is always at identity scale by construction — it cannot produce
an amplitude explosion regardless of class count or sample size.

For already-saved `.pth` files that still contain the wrong recolouring filter, a
standalone patch script `patch_itsa_recenter_only.py` was written that loads the saved
object, replaces `A_filters_[subject]` with `M_inv_sqrt_[subject]`, and saves a new file.
This requires `M_inv_sqrt_` to be present in the saved object (see Bug 8).

---

## Bug 8 — `export_light()` did not save `M_inv_sqrt_`

### File: `ITSA.py`, method `export_light()`

### What was wrong

`export_light()` strips the ITSA object down to the fields needed for inference, discarding
large intermediate arrays (`Rs_`, per-subject rotation matrices) to reduce file size.
`M_inv_sqrt_` — the per-subject recentering matrices — was cleared in the exported object:

```python
lite.M_inv_sqrt_ = {}  # no necesitamos las de sujetos base
```

This was a reasonable assumption when the spatial filters `A_filters_` were considered
sufficient for inference: `M_inv_sqrt_` is incorporated into `A_filters_` during
`_derive_subject_filters()`, so it is not needed again at inference time for training
subjects.

However, for the LOSO held-out subject, `adapt_loso_test_subject()` must be called
*after* loading the saved object in order to derive the test subject's filter. That method
reads `M_inv_sqrt_[test_s]` directly. With the field cleared in the exported file, the
method would compute `M_inv_sqrt_` from scratch — but only if it was passed the test
subject's raw covariances, which are not available at load time in the standard workflow.
This also meant `patch_itsa_recenter_only.py` could not function on a `_light.pth` file
produced before this fix, since `M_inv_sqrt_[1]` was absent.

### Fix

Added `M_inv_sqrt_` to the fields preserved by `export_light()`:

```python
# Before (discarded):
lite.M_inv_sqrt_ = {}

# After (preserved):
lite.M_inv_sqrt_ = {
    int(s): v.astype(np.float32)
    for s, v in self.M_inv_sqrt_.items()
}
```

The storage cost is negligible: for 6 subjects with 128-channel EEG,
`M_inv_sqrt_` is 6 × 128 × 128 × 4 bytes ≈ 0.4 MB, compared to `Rs_`
which can reach several GB.

---

## Summary table

| # | Bug | File | Method / Lines | Impact |
|---|-----|------|----------------|--------|
| 1 | Held-out subject gets identity filter (no alignment) | `ITSA.py` | `_derive_subject_filters`, new `adapt_loso_test_subject` | ITSA has zero effect on test set |
| 2 | Procrustes used instead of SVD; anchor row mismatch | `ITSA.py` | `transform_test_loso` lines 250–306 | Wrong rotation, often falls back to identity |
| 3 | `from_dataset_loso()` missing, causes `AttributeError` | `ITSA.py` | New classmethod on `ITSAIntegrator` | Script crashes on startup in LOSO mode |
| 4 | `pred` used instead of `pref` in test accuracy | `transformer_eeg_signal_classification.py` | Line 318 | Reported test accuracy is last train-batch accuracy |
| 5 | Test accumulation loop unnecessary and fragile | `transformer_eeg_signal_classification.py` | Lines 255–321 | Division-by-zero when `itsa_off=True`; stale `pred` variable |
| 6 | Filter norm explosion (~640× identity scale) | `ITSA.py` | `adapt_loso_test_subject` | Every signal amplified ~640×; TeL diverges immediately |
| 7 | Back-projection produces wrong filter direction for 40-class LOSO | `ITSA.py` | `adapt_loso_test_subject` | Filter steers test signals away from training distribution; TeA below chance |
| 8 | `export_light()` discarded `M_inv_sqrt_`, blocking post-load patching | `ITSA.py` | `export_light` | Cannot apply recentering-only fix to saved files without re-fitting |
