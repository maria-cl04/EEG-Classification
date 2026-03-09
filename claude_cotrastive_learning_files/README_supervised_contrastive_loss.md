# README — Supervised Contrastive Loss (SupCon)

---

## Table of Contents

1. [What Problem Does This Solve?](#1-what-problem-does-this-solve)
2. [Background: How the Model Normally Learns](#2-background-how-the-model-normally-learns)
3. [The Core Idea: Contrastive Learning](#3-the-core-idea-contrastive-learning)
4. [What "Supervised" Means Here](#4-what-supervised-means-here)
5. [The Projection Head — Why It Exists](#5-the-projection-head--why-it-exists)
6. [The Mathematics (Plain English Version)](#6-the-mathematics-plain-english-version)
7. [Why This Helps Each Experiment Type](#7-why-this-helps-each-experiment-type)
8. [Files Changed](#8-files-changed)
9. [Exact Code Changes](#9-exact-code-changes)
10. [Hyperparameters and How to Tune Them](#10-hyperparameters-and-how-to-tune-them)
11. [How to Run](#11-how-to-run)
12. [What to Expect](#12-what-to-expect)

---

## 1. What Problem Does This Solve?

Your transformer learns to classify EEG signals by converting each trial into a **point in a high-dimensional space**. Imagine a 128-dimensional space (because your `d_model=128`) where every EEG trial gets placed somewhere. Trials that the model considers similar end up close together; trials it considers different end up far apart.

The standard training signal — cross-entropy loss — only gives the model one instruction: **"for this trial, make the score for the correct class high."** That's a useful signal, but it's indirect. The model learns *what class each trial belongs to*, but it doesn't get any explicit instruction about the *geometric structure* of the embedding space. In particular, it is never told:

- "All cat trials should be near each other"
- "Cat trials and dog trials should be far from each other"
- "A cat trial from Subject 1 and a cat trial from Subject 5 should be close, even though they came from different people"

This matters enormously for cross-subject experiments (LOSO and fine-tuning). Without explicit clustering pressure, the model's embedding space often ends up organised partly by category and partly by subject identity — because subject-specific noise can be just as predictive as category signal when training and testing on the same subjects. When you then test on a new subject, those subject-specific clusters are useless.

**Supervised Contrastive Loss is the solution.** It adds an explicit geometric instruction to every training step: *same class = close together, different class = far apart*, applied directly to all pairs of trials in each batch.

---

## 2. Background: How the Model Normally Learns

To understand why this is an improvement, you need to understand what normally happens.

### The Standard Pipeline

```
Raw EEG trial (440 time-steps × 128 channels)
        │
        ▼
[Transformer Encoder]
        │
        ▼
Mean-pooled embedding  (1 × 128 vector)
        │
        ▼
[Linear Classifier]  (128 → 40)
        │
        ▼
40 scores (one per class)
        │
        ▼
Cross-Entropy Loss: penalise if the correct class score is not the highest
```

The cross-entropy loss only operates at the very end — on the 40 class scores. The gradient of this loss flows backwards through the classifier and into the encoder, slowly shaping the embedding space. But the signal is *indirect*. The encoder is never told "your embeddings for same-class trials should be clustered." It only knows "the classifier made the wrong prediction, adjust accordingly."

Think of it like training an artist by only ever saying "that painting is wrong" without ever showing them examples of what a correct painting looks like. They'll improve, but slowly and with a lot of confusion.

### What Goes Wrong Across Subjects

Because the signal is indirect, the model finds whatever shortcut works best on the training data. With 6 subjects, there are systematic differences between individuals — electrode placement, skull thickness, medication, neural noise profiles. These differences create subject-specific patterns that can accidentally correlate with categories in the training set.

When you test on a held-out subject (LOSO), those subject-specific shortcuts fail, and accuracy collapses.

---

## 3. The Core Idea: Contrastive Learning

Contrastive learning turns the *relationship between samples* into a direct training signal.

### The Marble Analogy

Imagine every EEG trial is a marble, and the embedding space is a large table. After training, each marble ends up somewhere on the table based on what the model thinks of it.

- **Standard training** says: "move each marble toward its correct corner of the table." That's the only instruction. Marbles of the same colour (class) might end up near each other, or they might not — the model doesn't care, as long as each one is in roughly the right region.

- **Contrastive training** adds: "all red marbles must be clumped together, all blue marbles must be clumped together, and no two clumps should overlap." This is enforced directly on every pair of marbles, at every training step.

### The Positive/Negative Pair Concept

In contrastive learning, every sample in a batch plays the role of an **anchor**. For each anchor:
- **Positive samples**: other samples in the batch with the *same class label*. These should be pulled closer.
- **Negative samples**: all other samples in the batch with *different class labels*. These should be pushed away.

With a batch size of 128 and 40 balanced classes, each anchor has roughly 2–3 positives and ~125 negatives in every batch. The loss is computed over all these relationships simultaneously, every step.

### The Hypersphere

The contrastive loss operates on a **unit hypersphere** — all embeddings are scaled to have length 1 (L2 normalisation). This means position on the sphere is determined entirely by *direction*, not magnitude. Two embeddings are "close" if they point in nearly the same direction, "far" if they point in opposite directions.

Using a sphere (rather than unconstrained space) is important because it prevents a trivial solution: without it, the model could make embeddings "far apart" simply by making them very large numbers in all dimensions, which is meaningless.

---

## 4. What "Supervised" Means Here

There are two flavours of contrastive learning:

- **Self-supervised contrastive learning** (e.g., SimCLR): there are no labels. "Positives" are defined as augmented versions of the same sample (e.g., two different random crops of the same image). The model learns that two views of the same data point should be similar.

- **Supervised contrastive learning** (SupCon, what we're using): labels are available. "Positives" are defined as *any two samples that share the same label*, regardless of who the subject was or what the raw signal looks like. This is much more powerful because there can be many positives per anchor (all same-class samples in the batch), not just one augmented twin.

For your EEG dataset specifically, supervised contrastive is the right choice because:
1. You have labels (40 image categories).
2. You have multiple trials per class per subject — plenty of genuine positives in every batch.
3. The signal of interest (visual evoked response) should be *shared across subjects*. By treating all same-category trials as positives — regardless of subject — you directly train the model to find what's common across people.

---

## 5. The Projection Head — Why It Exists

You might wonder: why not apply the contrastive loss directly to the transformer's embeddings, the same vectors that feed the classifier?

The answer is that the classifier and the contrastive loss have **conflicting geometric requirements**.

- The classifier (a linear layer mapping 128 → 40) needs the embedding space to be linearly separable by class. Think of it as needing 40 flat half-planes cutting through the space.
- The contrastive loss needs the embedding space to be spherically clustered — tight balls of same-class points, separated by gaps.

These two structures are not the same, and forcing the encoder to satisfy both simultaneously under-serves both.

The solution, introduced in the SimCLR paper and validated in SupCon, is to add a **separate, small neural network** — the projection head — on top of the encoder. The contrastive loss is applied to the *projection head's output*, not the encoder's output. The classifier is applied to the *encoder's output*, not the projection head's output.

```
Encoder output (128-dim)
    │
    ├──────────────────────────────────────────────►  [Classifier]  →  40 class scores  →  Cross-Entropy Loss
    │
    └──────────────────────────────────────────────►  [Projection Head: Linear→ReLU→Linear]
                                                                │
                                                        L2 normalise
                                                                │
                                                        Projected vector  →  SupCon Loss
```

After training, the projection head is completely **discarded**. It is never used during evaluation. Its entire purpose was to give the contrastive loss a dedicated "scratchpad" to work in, without contaminating the representation that the classifier depends on.

This is why a new `proj_head` attribute exists in the model and a `return_proj` flag exists in `forward()`. When `return_proj=False` (the default, used at val/test time), the projection head is never called and adds zero computational cost.

---

## 6. The Mathematics (Plain English Version)

The SupCon loss for a single anchor sample `i` is:

```
L_i = - (1 / number_of_positives) × Σ over all positives j: log [ similarity(i,j) / sum of similarities(i, everyone_else) ]
```

Breaking this down:

**`similarity(i, j)`** = `exp( dot_product(z_i, z_j) / temperature )`. Because both `z_i` and `z_j` are L2-normalised (unit vectors), the dot product is the cosine similarity — a number between -1 and 1 measuring how similar their directions are. Dividing by temperature (τ = 0.07) sharpens the distribution: similarities that were 0.8 and 0.9 before become much more different after dividing by 0.07.

**The denominator** sums `similarity(i, k)` over all `k ≠ i` in the batch (positives *and* negatives). This is exactly like a softmax denominator.

**The log of the ratio** is therefore the log-probability of selecting a positive `j` when randomly sampling from all other samples proportional to their similarity to `i`. Maximising this log-probability = making positives more similar to `i` than negatives are.

**Averaging over all positives** means every same-class sample in the batch contributes equally to the gradient. With 128 samples and 40 classes, you have roughly 2–3 positives per anchor, so roughly 2–3 gradient signals per anchor per step, compared to just 1 with cross-entropy.

**The combined loss** used in training:

```
total_loss = cross_entropy_loss + λ × supcon_loss
```

where λ = 0.1 by default. The cross-entropy loss ensures the model still directly optimises classification accuracy. The SupCon loss adds the clustering regularisation. λ controls the trade-off.

---

## 7. Why This Helps Each Experiment Type

### Multi-subject (all 6 together)
Every batch contains trials from multiple subjects. SupCon explicitly treats same-class, different-subject trials as positives — they must be pulled together. The model is directly trained to find the *cross-subject category signal* and ignore subject identity. Expected gain: moderate but consistent.

### Single-subject
Acts as a regulariser — the embedding space is more structured, which can reduce overfitting on smaller per-subject datasets. Less critical here since training and testing are same-subject, but won't hurt.

### LOSO (most impactful for this loss)
Training on 5 subjects, testing on 1. SupCon directly trains the encoder to make same-category embeddings from different subjects look alike — exactly the property needed for cross-subject generalisation. Without it, the model might learn subject-specific representations that fail on subject 6.

### Fine-tuning
The pre-trained model (trained on 5 subjects with SupCon) starts with a well-clustered embedding space. Fine-tuning on subject 6 data then only needs to *move* the clusters slightly toward subject 6's patterns, rather than building clusters from scratch. Using a lower λ (0.05) during fine-tuning prevents the contrastive loss from undoing the subject-6-specific adjustments.

---

## 8. Files Changed

| File | Status | What changed |
|------|--------|-------------|
| `models/supcon_loss.py` | **New** | The SupConLoss module |
| `models/transformer2.py` | **Modified** | Added projection head, `_encode()` helper, `get_embedding()`, `return_proj` flag |
| `transformer_eeg_signal_classification.py` | **Modified** | Import, 4 new CLI args, `proj_dim` in model init, modified training loop |

---

## 9. Exact Code Changes

### 9.1 `models/supcon_loss.py` (entirely new)

This file contains one class: `SupConLoss`. It is a standard `nn.Module` with no learnable parameters — just a `forward()` function that takes `(features, labels)` and returns a scalar loss.

Key implementation details:
- Features are L2-normalised inside the loss function as a safety measure (even though the model already normalises them).
- The row-wise max subtraction before `exp()` is a numerical stability trick (equivalent to the log-sum-exp trick) that prevents overflow when similarities are large.
- Anchors with zero positives in the batch are excluded from the loss to avoid division by zero. This cannot happen with your setup (balanced 40-class, batch 128) but is good defensive coding.

### 9.2 `models/transformer2.py`

**In `__init__`** — added at the end, after `self.classifier`:

```python
# BEFORE: nothing here

# AFTER:
if proj_dim is not None:
    self.proj_head = nn.Sequential(
        nn.Linear(d_model, d_model),
        nn.ReLU(),
        nn.Linear(d_model, proj_dim),
    )
else:
    self.proj_head = None
```

The `proj_dim` argument defaults to 128, so existing code that constructs `Model()` without specifying it gets a projection head automatically. Setting `proj_dim=None` disables it entirely (useful for ablation experiments).

**New internal method `_encode()`**:

```python
# BEFORE: this code lived directly inside forward()

# AFTER: extracted into _encode() so both forward() and get_embedding() can call it
def _encode(self, x, mask=None):
    if x.dim() == 4:
        x = x.squeeze(1).permute(0, 2, 1)
    x = self.embedding(x)
    x = self.pos_encoder(x)
    x = self.dropout(x)
    for layer in self.encoder_layers:
        x = layer(x, mask)
    x = x.mean(dim=1)   # mean pooling over time → (B, d_model)
    return x
```

**New public method `get_embedding()`**:

```python
# NEW — used by the prototypical evaluation script (README_prototypical.md)
def get_embedding(self, x, mask=None):
    return self._encode(x, mask)
```

**Modified `forward()`**:

```python
# BEFORE:
def forward(self, x, mask=None):
    # ... encoder code ...
    x = x.mean(dim=1)
    out = self.classifier(x)
    return out

# AFTER:
def forward(self, x, mask=None, return_proj=False):
    emb = self._encode(x, mask)       # (B, d_model)
    logits = self.classifier(emb)     # (B, num_classes)
    if return_proj and self.proj_head is not None:
        proj = self.proj_head(emb)    # (B, proj_dim)
        proj = F.normalize(proj, dim=1)
        return logits, proj
    return logits                     # unchanged default behaviour
```

The `return_proj=False` default means every existing call to `model(input)` continues to work exactly as before. No existing code breaks.

### 9.3 `transformer_eeg_signal_classification.py`

**New imports** (after existing imports):

```python
from models.supcon_loss import SupConLoss
```

**New CLI arguments** (after existing `parser.add_argument` calls):

```python
parser.add_argument('--supcon', default=True, action=argparse.BooleanOptionalAction,
                    help="enable supervised contrastive loss (default: True)")
parser.add_argument('--lambda-supcon', default=0.1, type=float)
parser.add_argument('--supcon-temp', default=0.07, type=float)
parser.add_argument('--proj-dim', default=128, type=int)
```

**Model construction** — pass `proj_dim`:

```python
# BEFORE:
model = module.Model(**model_options)

# AFTER:
model = module.Model(**model_options, proj_dim=opt.proj_dim)
```

**Loss function instantiation** (once, before the epoch loop):

```python
supcon_loss_fn = SupConLoss(temperature=opt.supcon_temp)
```

**Training loop** — the only place where behaviour changes at runtime:

```python
# BEFORE:
output = model(input)
loss = F.cross_entropy(output, target)

# AFTER (inside the train split block):
if split == "train" and opt.supcon:
    output, proj = model(input, return_proj=True)
    ce_loss = F.cross_entropy(output, target)
    sc_loss = supcon_loss_fn(proj, target)
    loss = ce_loss + opt.lambda_supcon * sc_loss
else:
    output = model(input)
    loss = F.cross_entropy(output, target)
```

Note that `val` and `test` splits always take the `else` branch — no projection, no overhead, no change in behaviour.

---

## 10. Hyperparameters and How to Tune Them

### `--lambda-supcon` (default: 0.1)
Controls how strongly the contrastive loss influences training relative to cross-entropy.

- `total_loss = CE + lambda × SupCon`
- At `lambda=0.0`, this is identical to the original training (SupCon has no effect).
- At `lambda=1.0`, both losses contribute equally.
- **Recommended tuning strategy**: start at 0.1. If validation accuracy is higher than the baseline, try 0.3. If it's lower, try 0.05. The right value depends on how much subject noise exists relative to category signal.
- During fine-tuning, use a lower value (e.g., 0.05) so that subject-6-specific fine-tuning is not over-regularised.

### `--supcon-temp` (default: 0.07)
The temperature τ in the softmax denominator.

- Lower temperature (e.g., 0.05): the loss concentrates more on hard negatives (embeddings that are dangerously close to the anchor despite having a different label). Leads to tighter clusters but harder optimisation — can cause training instability.
- Higher temperature (e.g., 0.2): smoother gradients, more stable but potentially less discriminative clusters.
- 0.07 is the value used in the original SupCon paper (Khosla et al., NeurIPS 2020). Start here.

### `--proj-dim` (default: 128)
The output dimensionality of the projection head.

- Should be ≤ `d_model` (128). Equal to `d_model` is standard.
- Reducing to 64 adds more of a bottleneck, which can act as additional regularisation.
- Increasing above 128 is unlikely to help.

---

## 11. How to Run

### Enable SupCon (default, recommended):
```bash
python transformer_eeg_signal_classification.py \
    --supcon \
    --lambda-supcon 0.1 \
    --supcon-temp 0.07
```

### Disable SupCon (for baseline comparison):
```bash
python transformer_eeg_signal_classification.py --no-supcon
```

### Ablation (vary lambda):
```bash
# Weaker contrastive signal
python transformer_eeg_signal_classification.py --supcon --lambda-supcon 0.05

# Stronger contrastive signal
python transformer_eeg_signal_classification.py --supcon --lambda-supcon 0.3
```

---

## 12. What to Expect

| Experiment | Expected change vs. baseline |
|---|---|
| Multi-subject | +1–3% accuracy |
| Single-subject | Neutral to +1% (acts as regulariser) |
| LOSO | Most variable. +0–5% depending on subject. Will not make things worse. |
| Fine-tuning | +1–3% on test subject after fine-tuning phase |

The biggest gains are in LOSO and multi-subject, which is exactly where cross-subject generalisation matters most. If you see no improvement in LOSO, check that the batch contains samples from multiple subjects — if your DataLoader somehow groups by subject, there are no cross-subject positives and the loss degenerates.
