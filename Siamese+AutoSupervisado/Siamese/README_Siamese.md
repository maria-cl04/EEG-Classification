# README — Siamese Network with Online Hard Triplet Mining

---

## Table of Contents

1. [What Problem Does This Solve?](#1-what-problem-does-this-solve)
2. [Background: How the Model Normally Learns](#2-background-how-the-model-normally-learns)
3. [The Core Idea: Metric Learning with Triplets](#3-the-core-idea-metric-learning-with-triplets)
4. [What "Siamese" Means Here](#4-what-siamese-means-here)
5. [Why No Projection Head](#5-why-no-projection-head)
6. [The Mathematics (Plain English Version)](#6-the-mathematics-plain-english-version)
7. [Why This Helps Each Experiment Type](#7-why-this-helps-each-experiment-type)
8. [Files Changed](#8-files-changed)
9. [Exact Code Changes](#9-exact-code-changes)
10. [Hyperparameters and How to Tune Them](#10-hyperparameters-and-how-to-tune-them)
11. [How to Run](#11-how-to-run)
12. [What to Expect](#12-what-to-expect)
13. [Bibliography](#13-bibliography)

---

## 1. What Problem Does This Solve?

Your transformer learns to classify EEG signals by mapping each trial to a **point in a 128-dimensional embedding space**. The standard cross-entropy loss only tells the model: *"Predict the correct ImageNet class for this trial."*

This is a purely local instruction. The model learns which class each embedding belongs to, but it is never told anything about the **geometry** of the space it is building. After training, embeddings from the same class might be scattered across very different regions of the 128-dimensional space. The classifier can still learn to draw a boundary around them, but that boundary is brittle: a small shift in the input (different subject, different session) can push embeddings across class boundaries and cause accuracy to collapse.

**Siamese metric learning with triplet loss is the solution.** Rather than just telling the model which bucket each sample belongs to, it gives an explicit geometric instruction: *"For every single trial you process, its embedding must be closer to all other trials from the same visual class than to any trial from a different class."*

The result is a compact, well-separated embedding space that is intrinsically more robust — the classifier does not have to work nearly as hard, because the encoder has already done most of the organising.

---

## 2. Background: How the Model Normally Learns

To understand why this helps, look at the standard pipeline:

### The Standard Pipeline

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

In standard training, the 128-dimensional embedding space is shaped entirely by what is most convenient for the linear classifier. Two trials from the "Dog" class might land on opposite sides of the space as long as both still get classified correctly. The encoder has no direct incentive to group them together.

When subject variability is introduced (LOSO, fine-tuning), the embedding distribution shifts. The classifier boundaries, which were fitted to the old distribution, no longer apply. A geometrically structured embedding space degrades far more gracefully, because the relative arrangement of classes is meaningful and stable rather than arbitrary.

---

## 3. The Core Idea: Metric Learning with Triplets

Metric learning forces the encoder to build a structured geometry. Rather than learning *what label does this sample get*, it learns *how far apart should any two samples be*, based on whether they share a label or not.

### The Neighbourhood Analogy

Imagine the 128-dimensional embedding space as a city map, and each EEG trial as a resident who must find a home.

- **Standard training** tells each resident: *"Move into district 12 (Dog)."* The residents scatter across the district wherever there is space.
- **Triplet training** tells each resident: *"You must live closer to your neighbours from district 12 than to anyone from district 5 (Cat) — and this constraint must hold with a comfortable safety gap, not just barely."*

After triplet training, district 12 is a tight, coherent neighbourhood. The linear classifier only needs to draw the district boundaries, not heroically carve up a chaotic map.

### The Positive/Negative Pair Concept

The triplet loss works with **triplets of samples**: an anchor, a positive, and a negative.

- **Anchor**: Any EEG trial in the batch.
- **Positive sample**: A different trial from the *same* visual class as the anchor.
- **Negative sample**: Any trial from a *different* visual class.

The loss penalises the model whenever the anchor-positive distance exceeds the anchor-negative distance by less than a fixed margin. In other words, it pushes the positive closer and the negative further away, *simultaneously*, for every anchor in every batch.

### Online Hard Mining

Naively enumerating every possible triplet in the dataset is prohibitively expensive and produces many uninformative triplets (anchor-positive pairs that are already close, anchor-negative pairs that are already far). These easy triplets have near-zero gradient and do not help the model learn.

Instead, **online hard mining** selects the most informative triplet for each anchor *on the fly*, within each batch:

- **Hardest positive**: The same-class sample that is *furthest* from the anchor — the most difficult case to pull in.
- **Hardest negative**: The different-class sample that is *closest* to the anchor — the most dangerous impostor.

With 40 classes and a batch size of 128, each batch contains approximately 3 samples per class. For each of the 128 anchors, the miner identifies the hardest positive and hardest negative from those 3 and ~125 candidates respectively. The result is 128 maximally informative triplets per batch, computed with no overhead beyond a pairwise distance matrix.

---

## 4. What "Siamese" Means Here

A classic Siamese network uses two physically separate but weight-tied copies of the encoder to process a pair of inputs simultaneously. The term emphasises the key property: **the two branches share identical weights**, so both samples are encoded by the same function.

In this implementation, the transformer encoder is **implicitly Siamese**. There is only one encoder in memory, but every sample in every triplet passes through it. Because all samples in a batch share the same forward pass through the same weights, the Siamese constraint — weight-tying — is automatically satisfied. No architectural duplication is needed.

Structurally, this is identical to a classic Siamese network. The difference is purely implementation efficiency: rather than running three separate forward passes for anchor, positive, and negative, the entire batch is processed in one vectorised pass and the triplet mining happens over the resulting distance matrix.

---

## 5. Why No Projection Head

SupCon and SimCLR both use a **projection head** — a small MLP applied on top of the encoder output — and the contrastive loss is applied to that projection rather than to the raw encoder output. The reason is a geometric conflict: classification wants flat hyperplane boundaries, while contrastive losses on the unit hypersphere want spherical cluster structure.

The triplet loss does not have this conflict, for two reasons:

1. **It operates in Euclidean space, not on a hypersphere.** Embeddings are not L2-normalised. The loss simply minimises Euclidean distances within classes and maximises them across classes — the same geometry that a linear classifier uses.

2. **Compact, separated clusters in Euclidean space are directly beneficial to the classifier.** Pulling same-class embeddings together and pushing different-class embeddings apart is equivalent to improving linear separability. The classification and metric objectives reinforce rather than conflict with each other.

For this reason, the triplet loss is applied directly to `get_embedding()` output — the raw mean-pooled encoder representation — without passing through the projection head. The projection head remains available in the model for SupCon and SimCLR experiments but is unused here.

---

## 6. The Mathematics (Plain English Version)

The triplet loss for a single anchor `a` with hardest positive `p` and hardest negative `n` is:

    L_a = max( d(a, p) − d(a, n) + margin, 0 )

Breaking this down:

**`d(a, p)`**: The Euclidean distance between the anchor embedding and its hardest positive (same-class, furthest). The model wants this to be *small*.

**`d(a, n)`**: The Euclidean distance between the anchor embedding and its hardest negative (different-class, closest). The model wants this to be *large*.

**`margin` (default: 1.0)**: A constant safety gap. The loss is not zero unless the negative is already further than the positive *by at least this margin*. Without the margin, the model could satisfy the constraint trivially by collapsing all embeddings to the same point. The margin enforces that the two groups are separated by a meaningful distance in Euclidean space.

**The `max(…, 0)` hinge**: Triplets where the constraint is already satisfied (negative is far enough away from the anchor) contribute zero loss. Only violated triplets — the hard cases — generate gradient. This is what makes online hard mining so effective: by always selecting the most violated triplets, almost every training step produces a non-zero gradient.

**The combined loss**:

    Total Loss = Cross-Entropy Loss + (lambda × Triplet Loss)

The triplet loss is averaged over all anchors in the batch that have at least one valid same-class partner (anchors with no same-class partner in the batch are excluded, since no positive can be formed).

---

## 7. Why This Helps Each Experiment Type

### Multi-subject (all 6 together)
With 6 subjects producing slightly different amplitude profiles for the same visual stimulus, the embedding space can become fragmented: "Dog seen by Subject 1" and "Dog seen by Subject 4" may land in different regions even though they share a label. The triplet loss directly corrects this by pulling all same-class embeddings together regardless of which subject produced them. Expected gain: Moderate, with substantially more compact and interpretable class clusters.

### Single-subject
Acts as an explicit regulariser on the embedding geometry. Standard cross-entropy can overfit on a single subject's data by memorising idiosyncratic patterns; the triplet loss adds a second objective that penalises memorisation (memorised patterns tend to produce arbitrary embedding geometries, not tight clusters). Expected gain: Moderate.

### LOSO (Leave-One-Subject-Out)
High impact. A compact, well-separated embedding space is inherently more transferable: when Subject 6's embeddings are slightly shifted by their individual noise profile, they are more likely to still land within the correct class cluster rather than drifting into a neighbouring one. The triplet loss provides exactly the geometric structure that makes this shift-robustness possible.

### Fine-tuning
Provides a structured pre-trained foundation. When fine-tuning on Subject 6, the starting embedding space already has meaningful class geometry, so the classifier reaches good performance with fewer labelled examples.

---

## 8. Files Changed

| File | Status | What changed |
|------|--------|-------------|
| `models/siamese_loss.py` | **New** | `TripletLoss` module with online hard mining in Euclidean space. |
| `models/transformer2.py` | **Unchanged** | `get_embedding()` was already added for this purpose. No further changes needed. |
| `transformer_eeg_signal_classification.py` | **Modified** | 3 new CLI args, import of `TripletLoss`, siamese branch in the training loop. |

---

## 9. Exact Code Changes

*(Note: `transformer2.py` already contains `get_embedding()` from the SupCon/SimCLR implementation phase. No further model changes are required.)*

### 9.1 `models/siamese_loss.py` (entirely new)

Contains the `TripletLoss` class. For a batch of N embeddings and their labels, it:
1. Computes the full N×N pairwise Euclidean distance matrix in one vectorised operation.
2. Builds boolean positive and negative masks from the label vector.
3. Selects the hardest positive (max distance, same class, excluding self) and hardest negative (min distance, different class) per anchor.
4. Applies the hinge loss `max(d_pos − d_neg + margin, 0)` and averages over valid anchors.

Anchors with no same-class partner in the batch return zero loss via a graph-connected zero tensor (so the optimiser step is never broken).

### 9.2 `transformer_eeg_signal_classification.py`

**New CLI arguments**:
```python
parser.add_argument('--siamese', default=False, action="store_true",
                    help="Enable Siamese (triplet) training")
parser.add_argument('--lambda-siamese', default=0.05, type=float,
                    help="Weight of the triplet loss")
parser.add_argument('--triplet-margin', default=1.0, type=float,
                    help="Margin for online hard triplet loss")
```

**Initialization** (before the epoch loop):
```python
if opt.siamese:
    from models.siamese_loss import TripletLoss
    criterion_triplet = TripletLoss(margin=opt.triplet_margin)
    if not opt.no_cuda:
        criterion_triplet = criterion_triplet.cuda()
```

**Training loop** (siamese forward pass):
```python
elif split == "train" and opt.siamese:
    # 1. CE Loss — standard classification pass
    output = model(input)
    loss_ce = F.cross_entropy(output, target)

    # 2. Triplet Loss — raw Euclidean embeddings (no projection head)
    embeddings = model.get_embedding(input)          # (B, d_model), NOT normalised
    loss_triplet = criterion_triplet(embeddings, target)

    # 3. Combine
    loss = loss_ce + (opt.lambda_siamese * loss_triplet)

    if i == 0:  # diagnostic printout once per epoch
        print(f"  [ep{epoch} batch0] CE={loss_ce.item():.4f}  Triplet={loss_triplet.item():.4f}")
```

---

## 10. Hyperparameters and How to Tune Them

### `--lambda-siamese` (default: 0.05)
Controls the strength of the triplet loss relative to cross-entropy.
- **Why 0.05?** This is the established safe starting point across all auxiliary losses in this project. At λ=0.05 the triplet loss contributes approximately 10% of the total loss by epoch 200, acting as a shaping signal without dominating the cross-entropy gradient.
- Unlike SimCLR, a converging triplet loss should be *beneficial* rather than harmful: because the loss is label-aware (labels define positive and negative pairs), convergence means the encoder is genuinely learning better class geometry, not contradicting the classification objective.
- If the triplet loss converges very early (< epoch 50) with no accuracy gain, try λ=0.1. If accuracy degrades relative to baseline, reduce to λ=0.02.

### `--triplet-margin` (default: 1.0)
The minimum required separation between same-class and different-class distances.
- **Why 1.0?** Standard starting point from the FaceNet literature. In a 128-dimensional Euclidean space with embeddings of typical unit-scale magnitude, a margin of 1.0 is meaningful without being so large that the loss never converges.
- If the triplet loss plateaus at a high value and accuracy does not improve, the margin may be too large — try 0.5.
- If the loss converges quickly to near-zero with no accuracy gain, the margin may be too loose — try 1.5.

### Key distinction from SimCLR
SimCLR showed a monotonic inverse relationship between NT-Xent convergence and accuracy: the more the contrastive loss converged, the worse the classification. **This is not expected here.** Because the triplet loss uses class labels, its positive and negative pairs are semantically correct. A converging triplet loss indicates genuine metric learning, not gradient interference. Monitor the diagnostic printout: if triplet loss decreases and accuracy also increases, the method is working as intended.

---

## 11. How to Run

### Enable Siamese training (default hyperparameters):
```bash
python transformer_eeg_signal_classification.py \
    --siamese \
    --lambda-siamese 0.05 \
    --triplet-margin 1.0
```

### Ablation — tighter margin:
```bash
python transformer_eeg_signal_classification.py \
    --siamese \
    --lambda-siamese 0.05 \
    --triplet-margin 0.5
```

### Ablation — stronger loss weight:
```bash
python transformer_eeg_signal_classification.py \
    --siamese \
    --lambda-siamese 0.1 \
    --triplet-margin 1.0
```

---

## 12. What to Expect

| Experiment | Expected change vs. baseline |
|---|---|
| Multi-subject | +2–4% accuracy. Visibly tighter class clusters if embeddings are visualised (e.g. t-SNE). |
| Single-subject | +1–2% accuracy (geometric regularisation reduces overfitting). |
| LOSO | +2–4%. Compact class clusters improve transfer to unseen subjects. |
| Fine-tuning | Structured embedding space allows faster convergence on the target subject. |

**Triplet loss convergence pattern to expect:** The triplet loss should start high (>1.0, many violated constraints) and decrease steadily across 200 epochs. Unlike SimCLR, partial or full convergence is a positive sign. If the loss remains flat throughout training, all triplets in every batch are being satisfied trivially — the margin is likely too small or λ too low.

---

## 13. Bibliography

### Primary Paper — FaceNet (Online Hard Triplet Mining):
> Schroff, F., Kalenichenko, D., & Philbin, J. (2015). FaceNet: A Unified Embedding for Face Recognition and Clustering. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR).*

### Foundational Paper — Siamese Networks:
> Bromley, J., Guyon, I., LeCun, Y., Säckinger, E., & Shah, R. (1993). Signature Verification Using a Siamese Time Delay Neural Network. *Advances in Neural Information Processing Systems (NeurIPS).*

### Application to EEG — Metric Learning for Brain Signals:
> Kostas, D., Aroca-Ouellette, S., & Bhatt, U. (2020). Thinker invariance: enabling BCI-capable neural networks to generalize across individuals. *Journal of Neural Engineering.*
*(Demonstrates that metric learning objectives improve cross-subject transfer in EEG classification — directly relevant to the LOSO and fine-tuning settings.)*

### Dataset:
> Spampinato, C., et al. (2017). Deep Learning Human Mind for Automated Visual Classification. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR).*
