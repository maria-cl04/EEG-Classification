# Contrastive Learning Implementation Guide
## EEG Visual Classification — Transformer Thesis

---

## Overview of Changes

Three independent improvements are delivered, ordered by expected impact:

| # | Method | Files changed | Best for |
|---|--------|--------------|----------|
| 1 | **Supervised Contrastive Loss** | `models/transformer2.py`, `models/supcon_loss.py`, `transformer_eeg_signal_classification.py` | All 4 experiments |
| 2 | **MAE Self-supervised Pre-training** | `transformer_pretrain.py` (new) | LOSO, Fine-tuning |
| 3 | **Prototypical Network Inference** | `transformer_loso_proto.py` (new) | LOSO |

Each method is independently usable. They also compose: pre-train with MAE → fine-tune with CE + SupCon → evaluate with prototypical inference.

---

## Method 1 — Supervised Contrastive Loss

### What changed and exactly where

#### `models/supcon_loss.py` (new file)

A standalone `SupConLoss` module. It takes:
- `features`: `(N, proj_dim)` — L2-normalised projection vectors, one per sample
- `labels`: `(N,)` — integer class labels

The loss pulls together all same-class pairs and pushes apart all different-class pairs within a batch, using a temperature-scaled softmax over cosine similarities.

Key formula for anchor `i`:
```
L_i = -1/|P(i)| * Σ_{j∈P(i)} log [ exp(z_i·z_j/τ) / Σ_{k≠i} exp(z_i·z_k/τ) ]
```
where `P(i)` = all other samples in the batch with the same label as `i`, and `τ` is temperature.

#### `models/transformer2.py` — 3 additions

**Addition 1 — Projection head** (`__init__`, after `self.classifier`):
```python
# BEFORE (nothing here)

# AFTER
if proj_dim is not None:
    self.proj_head = nn.Sequential(
        nn.Linear(d_model, d_model),
        nn.ReLU(),
        nn.Linear(d_model, proj_dim),
    )
```
Why a separate projection head? The classifier needs to map representations to class logits. The contrastive projection should be free to form a differently-shaped embedding space optimised for clustering. Following SimCLR/SupCon, the projection is discarded at inference — only the encoder + classifier are used for predictions.

**Addition 2 — `_encode()` helper** (refactored internal method):
The original `forward()` code is extracted into `_encode()`, which returns the mean-pooled `(B, d_model)` vector. Both `forward()` and the new `get_embedding()` call this.

**Addition 3 — `get_embedding()` method** (public):
```python
def get_embedding(self, x, mask=None):
    return self._encode(x, mask)
```
Used by the prototypical evaluation script to extract embeddings without touching the classifier.

**Change to `forward()`** — new `return_proj` flag:
```python
# BEFORE
def forward(self, x, mask=None):
    ...
    return out

# AFTER
def forward(self, x, mask=None, return_proj=False):
    emb = self._encode(x, mask)
    logits = self.classifier(emb)
    if return_proj and self.proj_head is not None:
        proj = self.proj_head(emb)
        proj = F.normalize(proj, dim=1)   # unit hypersphere
        return logits, proj
    return logits
```
Default is `return_proj=False` so the existing script works without modification.

#### `transformer_eeg_signal_classification.py` — 4 additions

**Addition 1 — new CLI arguments** (after existing `parser.add_argument` calls):
```python
parser.add_argument('--supcon', default=True, action=argparse.BooleanOptionalAction)
parser.add_argument('--lambda-supcon', default=0.1, type=float)
parser.add_argument('--supcon-temp',   default=0.07, type=float)
parser.add_argument('--proj-dim',      default=128,  type=int)
```

**Addition 2 — import**:
```python
from models.supcon_loss import SupConLoss
```

**Addition 3 — model construction** — pass `proj_dim`:
```python
# BEFORE
model = module.Model(**model_options)
# AFTER
model = module.Model(**model_options, proj_dim=opt.proj_dim)
```

**Addition 4 — training loop** (inside `for split in ("train", "val", "test")`):
```python
# BEFORE
output = model(input)
loss = F.cross_entropy(output, target)

# AFTER (in the train split only)
if split == "train" and opt.supcon:
    output, proj = model(input, return_proj=True)
    ce_loss = F.cross_entropy(output, target)
    sc_loss = supcon_loss_fn(proj, target)
    loss = ce_loss + opt.lambda_supcon * sc_loss
else:
    output = model(input)
    loss = F.cross_entropy(output, target)
```

### Hyperparameters to tune

| Parameter | Default | Notes |
|-----------|---------|-------|
| `--lambda-supcon` | `0.1` | Start here. If val accuracy degrades vs baseline, try `0.05`. If it improves and saturates, try `0.3` |
| `--supcon-temp` | `0.07` | SupCon paper default. Lower → sharper clusters but harder to optimise |
| `--proj-dim` | `128` | Equals `d_model`. Reducing to `64` can regularise further |

### Why this helps each experiment type

- **Multi-subject**: 6 subjects' data in one batch. SupCon explicitly trains the encoder to ignore subject identity and cluster by visual category.
- **Single-subject**: Acts as an additional regulariser. Less critical but won't hurt.
- **LOSO**: Most impactful. Representations learned with SupCon generalise better to unseen subject because same-class EEG from 5 different subjects was explicitly pulled together during training.
- **Fine-tuning**: The pre-trained encoder already has good clustering; fine-tuning maintains this while adapting the classifier.

---

## Method 2 — MAE Self-supervised Pre-training

### New file: `transformer_pretrain.py`

**Run this before any supervised training for LOSO or fine-tuning experiments:**

```bash
python transformer_pretrain.py \
    --eeg-dataset path/to/eeg_55_95_std.pth \
    --epochs 100 \
    --mask-ratio 0.75 \
    --d-model 128 \
    --num-heads 4 \
    --num-layers 1 \
    --d-ff 512 \
    --save-path pretrained_encoder.pth
```

### Architecture

```
Input (B, T, C=128)
    │
    ▼
[MAE Encoder]   ← same architecture as transformer2's encoder
    │              only sees ~25% of time-steps (unmasked)
    ▼
encoder tokens (B, n_visible, d_model)
    │
    ▼
[MAE Decoder]   ← lightweight (2 layers, d_model=64)
    │              fills in mask tokens, runs self-attention over all T positions
    ▼
reconstruction (B, T, C=128)
    │
    ▼
MSE loss on masked positions only
```

The decoder is asymmetric (smaller, shallower) so the encoder is forced to carry all the representational load. The decoder is discarded after pre-training.

### Loading pre-trained weights in the main script

Add this block in `transformer_eeg_signal_classification.py`, immediately after `model = module.Model(...)`:

```python
if opt.pretrained_encoder != '':
    state = torch.load(opt.pretrained_encoder, map_location='cuda')
    # strict=False: classifier.* and proj_head.* keys won't be in the
    # checkpoint — they're missing by design and will train from scratch.
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"Pre-trained encoder loaded. Missing: {missing}")
```

And add the argument:
```python
parser.add_argument('--pretrained-encoder', default='', help="MAE pre-trained encoder .pth")
```

### Why 75% mask ratio?

At 75% the task is hard enough that the encoder cannot rely on interpolation — it must learn the temporal structure of EEG. For comparison, BERT uses 15% (too easy) and the original MAE paper uses 75% for images. EEG has strong temporal autocorrelation so 75% is a reasonable starting point; you can also try 50%.

---

## Method 3 — Prototypical Network Inference (LOSO only)

### New file: `transformer_loso_proto.py`

After training a LOSO model with the standard script, run:

```bash
python transformer_loso_proto.py \
    --model  transformer2__subject1_epoch_200.pth \
    --eeg-dataset path/to/eeg_55_95_std.pth \
    --splits-path path/to/block_splits_LOSO_subject1.pth \
    --test-subject 1 \
    --distance euclidean
```

The script outputs both softmax accuracy and prototypical accuracy so you can report the delta.

### How it works

```
Step 1: Extract embeddings for all TRAINING samples
        (5 subjects × 40 classes × ~300 trials = ~60 000 trials)
        using model.get_embedding()  →  shape (N_train, 128)

Step 2: For each of 40 classes, compute
        prototype_c = mean( embeddings where label == c )
        →  shape (40, 128)

Step 3: For each TEST sample, compute Euclidean distance to all 40 prototypes.
        Assign the label of the nearest prototype.

Step 4: Report accuracy.
```

### Why this is better than the linear head for LOSO

The linear classifier in the original model is trained on 5 subjects. Its weight vectors are a weighted combination of the training-subject embeddings. When applied to the 6th subject — whose neural patterns differ — these weights have a subject-specific bias baked in.

Prototypes are the arithmetic mean of all same-class training embeddings. They are less sensitive to any single subject's idiosyncrasies. If SupCon training has pulled same-class, different-subject embeddings together, the prototype will sit in a canonical location that generalises to the held-out subject.

### Distance metric choice

- **Euclidean** (default): Works well when embeddings have consistent norms. Recommended when SupCon is used (it normalises the projection space).
- **Cosine**: Better when embedding norms vary. Try this if Euclidean underperforms.

---

## Recommended Experiment Schedule

### For your thesis, run in this order:

**Experiment 1 — Multi-subject (all 6)**
```bash
python transformer_eeg_signal_classification.py \
    -sub 0 \
    --supcon --lambda-supcon 0.1
```
Compare to baseline (without `--supcon`) to isolate SupCon's contribution.

**Experiment 2 — Single-subject (repeat for subjects 1–6)**
```bash
python transformer_eeg_signal_classification.py \
    -sub 1 \                  # repeat for 2,3,4,5,6
    --supcon --lambda-supcon 0.1
```

**Experiment 3 — LOSO (run for each of 6 held-out subjects)**
```bash
# Step A: Pre-train encoder (once per dataset, not per subject)
python transformer_pretrain.py --eeg-dataset ... --save-path pretrained.pth

# Step B: Fine-tune with SupCon
python transformer_eeg_signal_classification.py \
    -sub 0 \
    -sp  block_splits_LOSO_subject1.pth \
    --pretrained-encoder pretrained.pth \
    --supcon --lambda-supcon 0.1 \
    -e 200

# Step C: Prototypical evaluation (optional but recommended)
python transformer_loso_proto.py \
    --model transformer2__subject0_epoch_200.pth \
    --splits-path block_splits_LOSO_subject1.pth \
    --distance euclidean
```

**Experiment 4 — Fine-tuning**
```bash
# Step A: Train on 5 subjects with SupCon
python transformer_eeg_signal_classification.py \
    -sub 0 \
    -sp  splits_fineTuning_subject1.pth \
    --supcon --lambda-supcon 0.1 \
    -e 200 \
    -sc 200

# Step B: Fine-tune on portion of subject 1's data
python transformer_eeg_signal_classification.py \
    -sub 1 \
    -sp  splits_fineTuning_subject1_70percent.pth \
    --pretrained_net transformer2__subject0_epoch_200.pth \
    --supcon --lambda-supcon 0.05 \    # lower weight during fine-tuning
    -lr 0.0005 \                       # lower LR for fine-tuning
    -e 100
```

---

## What NOT to change

- The `Splitter` and `EEGDataset` classes — unchanged.
- The optimizer, scheduler, and epoch loop structure — unchanged.
- Batch size 128 — required for SupCon to work (needs enough same-class pairs per batch; at 128 samples with 40 balanced classes you get ~3 samples/class/batch on average, sufficient).
- The `num_classes=40` and `input_dim=128` defaults in `Model` — unchanged.

---

## Troubleshooting

**SupCon loss returns NaN**
- Cause: all samples in a batch have unique labels (no positives). Cannot happen with batch_size=128 and 40 balanced classes, but could occur with small batches.
- Fix: ensure `drop_last=True` in DataLoader (already set).

**MAE pre-training loss doesn't decrease**
- Check that the EEG tensors are not all-zero after slicing `time_low:time_high`.
- Reduce learning rate to `1e-4`.
- Ensure `num_workers=4` doesn't cause data loading race conditions on Windows (set to 0 if so).

**Prototypical accuracy lower than softmax**
- This means the embedding space is not well-clustered. Run SupCon training first.
- Try cosine distance instead of Euclidean (`--distance cosine`).
- Verify `model.get_embedding()` exists (requires the updated `transformer2.py`).

**`proj_dim` argument error when loading pretrained model**
- Old checkpoints saved without `proj_head`. The `strict=False` load handles this.
- If loading with `torch.load(path)` (full model save), the loaded object is the old class — you need to re-instantiate and copy weights manually, or just retrain.
