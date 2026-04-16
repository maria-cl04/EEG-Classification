# Similarity-Based Subject Selection for EEG Visual Classification

## 1. Project Overview

This project implements a **data-driven targeted transfer learning pipeline** for classifying EEG signals recorded during visual stimulation. The dataset is the Spampinato et al. (CVPR 2017) visual EEG dataset, where subjects viewed images from 40 ImageNet categories while their brain activity was recorded across 128 EEG channels.

### The Problem with Standard LOSO

The conventional approach to cross-subject EEG classification is **Leave-One-Subject-Out (LOSO)** cross-validation: train on all N-1 subjects, test on the one left out. While simple and reproducible, LOSO has a fundamental flaw — it treats all subjects as equally useful donors of training signal. In reality, EEG representations are highly idiosyncratic. A subject whose neural responses are structurally dissimilar to the target will contribute noise rather than signal, actively harming generalisation.

### The Proposed Solution

Instead of using all available subjects indiscriminately, this pipeline asks: **which subjects are most neurally similar to the target subject, and can we exploit that similarity to build a better-specialised model?**

The approach works as follows:

1. Train a baseline model on all subjects to learn general neural representations.
2. Use that frozen model to extract a compact embedding for every subject, then compute a pairwise **similarity (distance) matrix** over those embeddings.
3. For each target test subject, select only the most similar subjects as training donors, and train a new specialised model on that optimal subset.
4. Fine-tune the specialised model on a small held-out portion of the target subject's own data to adapt it to their idiosyncratic neural responses.

The key metric throughout is **Test Accuracy at maximum Validation Accuracy (TeA @ max VA)**, which captures the best generalisation point without any information leakage from the test set.

### Dataset Specifications

| Property | Value |
|---|---|
| Source | Spampinato et al., CVPR 2017 |
| Task | 40-class ImageNet object classification from EEG |
| Subjects | 6 healthy subjects |
| EEG file | `eeg_55_95_std.pth` (55–95 Hz Gamma band) |
| Channels | 128 |
| Time window | Samples 20–460 (440 time steps) |
| Input shape | `(Batch, Time, Channels)` → `(128, 440, 128)` |
| Trials | ~300 per class per subject |

The Gamma band (55–95 Hz) was chosen because it is strongly associated with high-level visual and object processing, making it the most task-relevant frequency range for this classification problem.

---

## 2. Transformer Architecture & Feature Extraction

### Model Architecture (`models/transformer2.py`)

The backbone is a lightweight, single-layer Transformer Encoder designed to operate directly on the temporal structure of EEG signals.

    Input: (B, 440, 128)
    │
    ├─ Linear Embedding      128 → 128 (d_model)
    ├─ Positional Encoding   sinusoidal, max_len=440
    ├─ Dropout               p=0.4
    │
    └─ Encoder Layer (×1)
    ├─ Multi-Head Self-Attention   4 heads, d_k=32
    ├─ Add & Norm
    ├─ Position-wise FFN           512-dim hidden
    └─ Add & Norm
    │
    ├─ Mean Pooling          (B, 440, 128) → (B, 128)    ← embedding lives here
    │
    └─ Linear Classifier     128 → 40

**Hyperparameters:**

| Parameter | Value |
|---|---|
| `d_model` | 128 (matches EEG channel count natively) |
| `num_heads` | 4 |
| `num_layers` | 1 |
| `d_ff` | 512 |
| `dropout` | 0.4 |
| `num_classes` | 40 |

### Decoupling Feature Extraction from Classification

A critical architectural change made during this project was the introduction of the `_encode()` private method, which decouples the shared encoder trunk from the classification head.

**The original `forward()` was a single monolithic pass:**
```python
# Original — inseparable trunk and head
def forward(self, x, mask=None):
    x = self.embedding(x)
    x = self.pos_encoder(x)
    x = self.dropout(x)
    for layer in self.encoder_layers:
        x = layer(x, mask)
    x = x.mean(dim=1)          # mean pooling
    return self.classifier(x)  # logits
```

**After refactoring, the trunk is isolated:**
```python
def _encode(self, x, mask=None):
    """Shared trunk — stops after mean pooling."""
    if x.dim() == 4:
        x = x.squeeze(1).permute(0, 2, 1)
    x = self.embedding(x)
    x = self.pos_encoder(x)
    x = self.dropout(x)
    for layer in self.encoder_layers:
        x = layer(x, mask)
    return x.mean(dim=1)       # (B, 128) — no classifier

def forward(self, x, mask=None):
    return self.classifier(self._encode(x, mask))

def get_embeddings(self, x, mask=None):
    return self._encode(x, mask)
```

### Why Mean Pooling Produces the Right Representation

After the Transformer Encoder, the signal has shape `(B, 440, 128)` — a sequence of 440 contextualised time-step vectors. Mean pooling collapses the temporal dimension by averaging across all 440 positions, yielding a single `(B, 128)` vector per sample. This is essential for two reasons:

- **Mathematical tractability:** A flat 2D tensor `(N_samples, 128)` allows direct computation of means, covariances, and pairwise distances. A 3D tensor `(N_samples, 440, 128)` does not admit a natural centroid without further design choices.
- **Semantic richness:** The mean pool integrates temporal context from the full 440-step window into a single summary vector that captures the subject's overall neural response pattern to visual stimulation, rather than any single time point.

Importantly, embeddings are always extracted with `model.eval()` and `torch.no_grad()`, which disables dropout and makes the representations **deterministic**. If dropout were active during extraction, each forward pass would yield a different centroid, making the distance matrix stochastic and unreliable.

---

## 3. The Three-Phase Pipeline (`similarity_based_selection.py`)

### Phase 1 — Baseline Multi-Subject Model

**Goal:** Train a single model on all 6 subjects to convergence, so that it learns general neural representations that transcend any individual's idiosyncrasies.

**Setup:**

| Split | Subjects | Purpose |
|---|---|---|
| Train | Subjects 1–4 | Optimise weights |
| Validation | Subject 5 | Early stopping signal |
| Test | Subject 6 | Held-out performance |

**Training configuration:**

| Parameter | Value |
|---|---|
| Epochs | 200 |
| Batch size | 128 |
| Optimiser | Adam, LR = 0.001 |
| LR scheduler | StepLR, γ=0.95 every 10 epochs |
| Loss | Cross-Entropy |

This phase can be skipped if a pre-trained baseline already exists, by passing `--baseline-model path/to/model.pth` at the command line.

### Phase 2 — Subject Centroids & Distance Matrix

**Goal:** Quantify how similar each subject's neural representations are to every other subject.

**Step 1 — Centroid extraction:**

The frozen baseline model is run in `eval` mode over all samples for each subject. `get_embeddings()` is called batch-by-batch to extract the 128-dim post-pooling embedding for every trial. The per-subject centroid is then computed as the mean over all trial embeddings:

    d_M(i, j) = sqrt( (μ_i - μ_j)^T  Σ_pooled^{-1}  (μ_i - μ_j) )

The resulting matrix is saved to `subject_distance_matrix.npy` for inspection and reproducibility.

**Example output (Euclidean, first iteration):**

            S1      S2      S3      S4      S5      S6
    S1 │  0.000   7.114   6.674   6.930   7.757   7.156
    S2 │  7.114   0.000   5.023   6.658   4.262   5.194
    S3 │  6.674   5.023   0.000   6.024   5.253   5.439
    S4 │  6.930   6.658   6.024   0.000   7.084   5.703
    S5 │  7.757   4.262   5.253   7.084   0.000   5.422
    S6 │  7.156   5.194   5.439   5.703   5.422   0.000

Notable findings from this matrix:
- **Subject 1 is a neural outlier.** Its minimum distance to any other subject (6.67 to S3) is higher than the minimum distance of every other subject to their closest neighbour.
- **Subjects 2 and 5 form the tightest pair** (4.26), suggesting strong alignment of their neural response patterns.
- Subject 1's centroid also has the highest L2 norm (6.009 vs. 4.0–4.7 for others), confirming it occupies a different region of embedding space entirely.

### Phase 3 — Targeted Training & Fine-Tuning

For each of the 6 subjects acting as the target test subject, the pipeline performs a two-stage training procedure.

**Subject selection:**

Using the distance matrix, the `n_train` subjects with the **lowest distance** to the target are selected as the training set, and the next `n_val` subjects form the validation set. The target subject itself is fully excluded from both sets.

Example for Subject 1 (Euclidean distances):

    Train: [S3, S4]   — distances 6.67, 6.93
    Val:   [S2]       — distance  7.11
    Test:  [S1]       — held out

**Stage B — Pre-training on similar subjects:**

A fresh model is initialised and trained from scratch on the selected training subjects (validated on the selected val subjects), using the same optimiser and scheduler configuration as Phase 1. Test accuracy is evaluated on the held-out target subject data throughout, so the pre-training TeA @ max VA is directly comparable to Phase 1 results.

**Stage C — Fine-tuning on the target subject:**

A small fraction (`--finetune-ratio`, default 20%) of the target subject's data is reserved for fine-tuning. The rest forms the held-out test set. The pre-trained model from Stage B is then adapted to the target subject using this small set, at a reduced learning rate (`--finetune-lr`, default 1e-4) with no LR scheduler.

Two fine-tuning modes are supported:
- **Full network** (`--freeze-encoder` not set): all parameters are updated at the low LR. Better when the target subject's representations are systematically shifted.
- **Head only** (`--freeze-encoder`): only the final `Linear(128 → 40)` classifier is updated. Safer when the fine-tuning set is very small (fewer than ~200 samples), as it prevents catastrophic forgetting of the encoder's learned representations.

The best epoch is tracked by fine-tuning loss (not val accuracy), since the fine-tuning set itself contains target subject data.

**Results are stored per subject:**

```python
results[target_subject] = {
    "train_subjects"      : [...],
    "val_subjects"        : [...],
    "pretrain_test_acc"   : ...,   # TeA@maxVA — Stage B only
    "finetune_test_acc"   : ...,   # TeA — after Stage C adaptation
}
```

---

## 4. Evolution of the Distance Metric: Euclidean → Mahalanobis

### First Iteration: Euclidean Distance

The first implementation used standard Euclidean distance between subject centroids:

```python
dist = (centroid_i - centroid_j).norm().item()
```

This was a natural starting point, but it carries a fundamental assumption that turned out to be problematic: **it treats all 128 embedding dimensions as independent and equally important.**

In practice, the 128 dimensions of the post-pooling embedding space are neither independent nor equally informative. Some dimensions may capture highly consistent, class-discriminative neural response patterns (low within-subject variance, high between-subject discriminability). Others may reflect high-frequency noise, subject movement artefacts, or other signals that are highly variable within a subject but carry no classification-relevant information.

The Euclidean metric is blind to this distinction. A subject pair can appear close simply because a handful of high-variance, noisy dimensions happen to align, while the truly informative, stable dimensions actually disagree. This makes the Euclidean distance matrix an unreliable proxy for genuine neural similarity.

**Observed consequences in the Subject 1 results:**

After training on the Euclidean-selected subjects [S3, S4] and evaluating on Subject 1, TeA @ max VA was **2.5% — exactly chance level** (1/40 = 2.5%). The model memorised S3 and S4 perfectly (TrA → 96%) but transferred nothing to the target, indicating that the two selected subjects were not genuinely useful donors despite being the Euclidean nearest neighbours.

### Intermediate Fix: Fine-Tuning Stage

Before switching the distance metric, a fine-tuning stage was added to Phase 3 as a partial remedy. Rather than relying entirely on cross-subject transfer, 20% of the target subject's own data is used to adapt the pre-trained model to their specific neural patterns (Stage C above).

This is the correct approach regardless of the distance metric — EEG representations are sufficiently idiosyncratic that some target-subject data will always be needed for strong personalised performance. However, fine-tuning can only work well if the pre-trained model has learned genuinely useful representations; if the pre-training subjects were poorly chosen by an unreliable distance metric, fine-tuning has a worse starting point.

### Final Implementation: Mahalanobis Distance

The complete fix was to replace the Euclidean metric with **Mahalanobis distance**, which accounts for the covariance structure of the embedding space.

**Mathematical formulation:**

    d_M(i, j) = sqrt( (μ_i - μ_j)^T  Σ_pooled^{-1}  (μ_i - μ_j) )

where `Σ_pooled` is the **pooled within-subject covariance matrix**, computed by accumulating the per-sample deviations from each subject's centroid across all subjects:

    Σ_pooled = [ Σ_i  Σ_j (x_{ij} - μ_i)(x_{ij} - μ_i)^T ] / (N_total - K)

where `N_total` is the total number of samples and `K` is the number of subjects (degrees of freedom correction).

**Why this is the right covariance to use:**

We deliberately use the *within-subject* covariance rather than the total or between-subject covariance. The total covariance would be inflated by the very between-subject differences we are trying to measure, producing a circular and distorted metric. The within-subject covariance captures only the noise and variability inherent to individual neural responses, which is exactly the right denominator: we want to know if two centroids are far apart *relative to how much subjects naturally vary internally*.

**What changes in the code:**

Three modifications were made to `similarity_based_selection.py`:

1. `extract_subject_centroids()` was updated to also return `all_embeddings` — a dictionary of every per-trial embedding tensor per subject, needed to compute the covariance matrix.

2. A new `compute_pooled_covariance()` function was added. It accumulates the within-subject scatter matrix, divides by the pooled degrees of freedom, adds a small regularisation term (`ε × I`, default `1e-5`) to guarantee invertibility, and returns the matrix inverse `Σ_pooled^{-1}`. Eigenvalue diagnostics (min, max, condition number) are printed to allow the user to detect numerical issues.

3. `compute_distance_matrix()` was updated to accept an optional `cov_inv` argument. When provided, it computes Mahalanobis distances; when `None`, it falls back to Euclidean, preserving backward compatibility.

**What to expect from the switch:**

- The absolute distance values will change scale — Mahalanobis distances are unitless and their magnitude depends on the eigenvalue spectrum of the covariance matrix, so direct comparison with the previous 4.2–7.7 Euclidean range is not meaningful.
- The **rank ordering of neighbours may change**. If a previous Euclidean pairing was driven by a high-variance, non-informative dimension, Mahalanobis will down-weight that dimension and may select different training subjects.
- If the ordering does change, the Mahalanobis selection is the more trustworthy one, and pre-training TeA @ max VA should improve because the donor subjects are more genuinely similar in the informative subspace.
- Subject 1's status as an outlier is expected to persist, since its centroid norm (6.009) was far above all others — that structural isolation is unlikely to be an artefact of a particular metric.

**Regularisation guidance:**

Monitor the condition number printed by `compute_pooled_covariance`. With 128 dimensions and ~11,500 total samples (6 subjects × 1,920 samples) the covariance matrix is well-determined in principle. However if the condition number exceeds ~10,000, some embedding dimensions are near-constant and the default `regularisation=1e-5` should be increased:

| Condition number | Recommended regularisation |
|---|---|
| < 1,000 | `1e-5` (default) |
| 1,000 – 10,000 | `1e-4` |
| > 10,000 | `1e-3` |

---

## 5. Usage & CLI Reference

### Full pipeline from scratch

```bash
python similarity_based_selection.py \
    -ed path/to/eeg_55_95_std.pth
```

### Skip Phase 1 using a pre-trained baseline (recommended)

```bash
python similarity_based_selection.py \
    -ed path/to/eeg_55_95_std.pth \
    --baseline-model baseline_model_all_subjects.pth
```

### Fine-tuning with frozen encoder (safer for small fine-tune sets)

```bash
python similarity_based_selection.py \
    -ed path/to/eeg_55_95_std.pth \
    --baseline-model baseline_model_all_subjects.pth \
    --freeze-encoder
```

### Full custom configuration

```bash
python similarity_based_selection.py \
    -ed path/to/eeg_55_95_std.pth \
    --baseline-model baseline_model_all_subjects.pth \
    --n-train 2 \
    --n-val 1 \
    --finetune-ratio 0.3 \
    --finetune-epochs 80 \
    --finetune-lr 0.00005 \
    --targeted-epochs 150 \
    --seed 42
```

### Complete CLI argument reference

| Argument | Default | Description |
|---|---|---|
| `-ed`, `--eeg-dataset` | *(required)* | Path to `eeg_55_95_std.pth` |
| `-tl`, `--time_low` | `20` | Start sample of EEG time window |
| `-th`, `--time_high` | `460` | End sample of EEG time window |
| `--num-subjects` | `6` | Total number of subjects in the dataset |
| `-e`, `--epochs` | `200` | Training epochs for Phase 1 baseline |
| `--targeted-epochs` | `200` | Training epochs for Phase 3 Stage B pre-training |
| `--baseline-model` | `''` | Path to pre-trained baseline — skips Phase 1 entirely |
| `--n-train` | `2` | Number of most-similar subjects used for training |
| `--n-val` | `1` | Number of subjects used for validation |
| `--finetune-ratio` | `0.2` | Fraction of target subject data used for fine-tuning |
| `--finetune-epochs` | `50` | Epochs for Stage C fine-tuning |
| `--finetune-lr` | `1e-4` | Learning rate for fine-tuning (lower than pre-training) |
| `--freeze-encoder` | `False` | If set, freeze encoder and train classifier head only |
| `-b`, `--batch-size` | `128` | Batch size for all stages |
| `-lr`, `--learning-rate` | `0.001` | Learning rate for Phases 1 and 3 pre-training |
| `-lrdb` | `0.95` | StepLR decay factor γ |
| `-lrde` | `10` | StepLR decay period (epochs) |
| `--seed` | `42` | Random seed for reproducibility |
| `--no-cuda` | `False` | Disable CUDA and run on CPU |

### Output files

| File | Description |
|---|---|
| `baseline_model_all_subjects.pth` | Saved baseline model from Phase 1 |
| `subject_distance_matrix.npy` | 6×6 NumPy array of pairwise distances |
| `targeted_model_subject{N}.pth` | Final fine-tuned model for subject N |
| `similarity_selection_results.pth` | Dict of all per-subject results (loadable with `torch.load`) |

---

## Project Structure

    .
    ├── models/
    │   └── transformer2.py                   # Transformer model with get_embeddings()
    ├── similarity_based_selection.py         # Main three-phase pipeline
    ├── transformer_eeg_signal_classification.py  # Original single-subject training script
    └── data/
    └── eeg_55_95_std.pth                 # Gamma-band EEG dataset (55–95 Hz)

---

## References

Spampinato, C., Palazzo, S., Kavasidis, I., Giordano, D., Souly, N., & Shah, M. (2017).
*Deep Learning Human Mind for Automated Visual Classification.*
IEEE Conference on Computer Vision and Pattern Recognition (CVPR 2017).