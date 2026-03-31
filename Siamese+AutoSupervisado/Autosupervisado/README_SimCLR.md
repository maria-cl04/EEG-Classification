# README — Self-Supervised Contrastive Learning (SimCLR)

---

## Table of Contents

1. [What Problem Does This Solve?](#1-what-problem-does-this-solve)
2. [Background: How the Model Normally Learns](#2-background-how-the-model-normally-learns)
3. [The Core Idea: Contrastive Learning](#3-the-core-idea-contrastive-learning)
4. [What "Self-Supervised" Means Here](#4-what-self-supervised-means-here)
5. [The Projection Head — Why It Exists](#5-the-projection-head--why-it-exists)
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

Your transformer learns to classify EEG signals by converting each trial into a **point in a high-dimensional space** (128 dimensions). 

EEG data is notoriously noisy. A signal contains the visual brainwave you care about, but it also contains amplifier noise, electrode impedance drops, and physiological artifacts. The standard cross-entropy loss only tells the model: **"Predict the correct ImageNet class for this trial."**

Because the model is lazy, it will use *any* pattern it finds to get the right answer. If a specific electrode glitch happens to correlate with the "Dog" class in your training data, the model will learn to look for the glitch instead of the brainwave. It is never explicitly taught what parts of the signal are "real" and what parts are "noise."

**Self-Supervised Contrastive Learning (SimCLR) is the solution.** It explicitly teaches the model the concept of invariance: *no matter how much physical noise or sensor failure occurs, the core digital representation of this trial should remain exactly the same.*

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

With standard training, if you feed the model an EEG trial with a slightly higher amplitude, the embedding might jump to a completely different part of the 128-dimensional space. The encoder has no internal concept of "robustness." 

When you test on a held-out subject (LOSO) or even just a different recording session, the baseline noise profile changes. The embeddings shift unpredictably, the classifier boundaries fail, and accuracy collapses.

---

## 3. The Core Idea: Contrastive Learning

Contrastive learning forces the model to understand relationships, rather than just memorizing paths to a final answer.

### The Marble Analogy

Imagine every raw EEG trial is a marble.
- **Standard training** just tries to throw the marble into the correct bucket (class).
- **Self-Supervised Contrastive training** takes a single marble. It looks at the marble through a scratched lens (adding Gaussian noise), and then looks at it through a tinted lens (masking out channels). 

The model is then given a massive pile of marbles and told: *"Find the two images that came from the exact same marble, and push all the other marbles away."*

### The Positive/Negative Pair Concept

In SimCLR, every sample in a batch goes through **two different random augmentations** (noise, masking, scaling) to create two "views" (z_i and z_j).
- **Positive sample**: For view z_i, its *only* positive is z_j (the other augmented view of the exact same trial).
- **Negative samples**: Every other augmented view in the entire batch.

If your batch size is 128, you generate 256 views. For any given view, there is exactly **1 positive** and **254 negatives**. The model must pull the positive pair together and push the 254 negatives away.

### The Hypersphere

Just like SupCon, this operates on a **unit hypersphere** using L2 normalisation. Two embeddings are "close" if they point in the same direction. This prevents the model from cheating by just making the numbers infinitely large.

---

## 4. What "Self-Supervised" Means Here

There are two flavours of contrastive learning:

- **Supervised contrastive learning (SupCon):** Uses labels. "Positives" are defined as any two samples that share the same class label. It groups all "Cats" together.
- **Self-supervised contrastive learning (SimCLR, what we are doing here):** Labels are completely ignored for the contrastive loss. "Positives" are purely defined by the data itself (two augmented versions of the same trial). It doesn't group "Cats" together; it groups "Trial 45 with noise" and "Trial 45 with dropped channels" together.

Why do this if we have labels? Because by learning to match augmented views without relying on the 40 classes, the encoder is forced to learn the deep, fundamental structure of human brainwaves. It learns a generic "language" of EEG that makes the classifier's job much easier later.

---

## 5. The Projection Head — Why It Exists

Just like in SupCon, the classifier and the contrastive loss have **conflicting geometric requirements**.

- The classifier needs the space divided by flat hyperplanes into 40 regions.
- The contrastive loss needs the space organized into spherical pairs that are invariant to noise.

We resolve this by adding a **projection head**.

    Encoder output (128-dim)
        │
        ├──────────────────────────────────────────────►  [Classifier]  →  40 scores  →  CE Loss
        │
        └──────────────────────────────────────────────►  [Projection Head: Linear→ReLU→Linear]
                                                                    │
                                                            L2 normalise
                                                                    │
                                                            Projected vector  →  NT-Xent Loss

The contrastive NT-Xent loss is applied to the *projection head's output*. The classification cross-entropy loss is applied to the *encoder's raw output*. After training, the projection head is discarded.

---

## 6. The Mathematics (Plain English Version)

The SimCLR loss (NT-Xent) for a single augmented view `i` is:

    L_i = -log( exp(similarity(i, j) / temperature) / sum_of_all_other_similarities )

Breaking this down:

**`similarity(i, j)`**: The dot product between the two L2-normalised views of the *same* trial. Because they are unit vectors, this is the cosine similarity (from -1 to 1). 

**The Temperature (`τ = 0.5`)**: Dividing by `τ` scales the similarities. A temperature of 0.5 is smoother than SupCon's 0.07. It prevents the model from being too aggressive when pushing negative samples away, which is important because in a batch without labels, some "negatives" might actually be from the same visual class by pure coincidence!

**The Denominator**: Sums the similarity between view `i` and *every other view* in the batch (the 254 negatives). 

**The Log Ratio**: This is essentially a softmax probability. Maximizing this equation means forcing the model to pick its augmented twin out of a lineup of 255 total options.

**The combined loss**:
    
    Total Loss = Cross-Entropy Loss + (lambda * SimCLR Loss)

---

## 7. Why This Helps Each Experiment Type

### Multi-subject (all 6 together)
Because SimCLR teaches the model to ignore amplitude shifts and channel noise, the encoder stops overfitting to the specific impedance levels of individual subjects. Expected gain: Moderate improvement, highly stable embeddings.

### Single-subject
Acts as a powerful regulariser. By artificially simulating noise (Gaussian, channel dropping) via augmentations, you effectively multiply the size of your training data, preventing the model from memorizing the limited single-subject dataset.

### LOSO (Leave-One-Subject-Out)
High impact. When testing on Subject 6, the model will encounter baseline amplitudes and noise patterns it has never seen. Because SimCLR explicitly trained the encoder to be invariant to these exact types of shifts, the model won't panic when Subject 6's data looks a bit different.

### Fine-tuning
Provides an incredibly robust pre-trained foundation. The model already understands what "noise" looks like before you even show it Subject 6's labels.

---

## 8. Files Changed

| File | Status | What changed |
|------|--------|-------------|
| `models/eeg_augmentations.py` | **New** | Handles Gaussian noise, channel masking, amplitude scaling. |
| `models/simclr_loss.py` | **New** | The NTXentLoss module. |
| `models/transformer2.py` | **Modified** | Same modifications as SupCon (Projection head, `_encode`, `return_proj`). |
| `transformer_eeg_signal_classification.py` | **Modified** | Import, 3 new CLI args, modified training loop for dual-view logic. |

---

## 9. Exact Code Changes

*(Note: The changes to `transformer2.py` are identical to those required by SupCon. If SupCon is already implemented, no further model architecture changes are needed).*

### 9.1 `models/eeg_augmentations.py` (entirely new)
Contains the `EEGAugmentation` class. It applies three label-preserving transformations stochastically:
- `gaussian_noise`: Adds zero-mean noise (`std=0.1`).
- `channel_mask`: Drops random channels (`p=0.1`).
- `amplitude_scale`: Uniformly scales the trial (`0.8` to `1.2`).
The `random_augment()` method applies each with a 50% probability.

### 9.2 `models/simclr_loss.py` (entirely new)
Contains `NTXentLoss`. It takes a batch of N views and N twin views, concatenates them into a 2N matrix, calculates the (2N × 2N) similarity matrix, masks the diagonal (so a sample isn't compared to itself), and applies cross-entropy to find the positive pairs.

### 9.3 `transformer_eeg_signal_classification.py`

**New CLI arguments**:
```python
parser.add_argument('--simclr', default=False, action="store_true", help="Enable SimCLR")
parser.add_argument('--lambda-simclr', default=0.05, type=float)
parser.add_argument('--simclr-temp', default=0.5, type=float)
```

**Initialization** (before the epoch loop):
```python
if opt.simclr:
    from models.eeg_augmentations import EEGAugmentation
    from models.simclr_loss import NTXentLoss
    augmenter = EEGAugmentation()
    device_target = torch.device("cuda" if not opt.no_cuda else "cpu")
    criterion_simclr = NTXentLoss(temperature=opt.simclr_temp, device=device_target)
    if not opt.no_cuda:
        criterion_simclr = criterion_simclr.cuda()
```

**Training loop** (hybrid forward pass):
```python
if split == "train" and opt.simclr:
    # 1. CE Loss on CLEAN data
    output = model(input)
    loss_ce = F.cross_entropy(output, target)

    # 2. NT-Xent Loss on AUGMENTED views
    view1 = augmenter.random_augment(input)
    view2 = augmenter.random_augment(input)
    _, z_i = model(view1, return_proj=True)
    _, z_j = model(view2, return_proj=True)
    loss_simclr = criterion_simclr(z_i, z_j)

    # 3. Combine
    loss = loss_ce + (opt.lambda_simclr * loss_simclr)
```

---

## 10. Hyperparameters and How to Tune Them

### `--lambda-simclr` (default: 0.05)
Controls the strength of the contrastive loss.
- **Why 0.05 instead of 0.1?** Because self-supervised tasks are inherently "noisier" than supervised tasks (since it doesn't use labels, it occasionally pushes samples from the same class apart). We weight it slightly lower than SupCon so it acts as a gentle regulariser rather than dominating the gradient.

### `--simclr-temp` (default: 0.5)
The temperature `τ`.
- Standard SimCLR literature uses 0.5. Unlike SupCon (which uses 0.07 to sharply cluster exact classes), SimCLR needs a softer temperature to avoid excessively penalising "false negatives" (samples of the same class that end up in the denominator). Start at 0.5 and do not change unless the loss refuses to converge.

---

## 11. How to Run

### Enable SimCLR:
```bash
python transformer_eeg_signal_classification.py \
    --simclr \
    --lambda-simclr 0.05 \
    --simclr-temp 0.5
```

---

## 12. What to Expect

| Experiment | Expected change vs. baseline |
|---|---|
| Multi-subject | +1–3% accuracy. Smoother validation curves (less spiking). |
| Single-subject | +1–2% accuracy (acts as a strong data-augmentation regulariser). |
| LOSO | +2–4%. Greatly improves baseline robustness against novel subjects. |
| Fine-tuning | Provides a highly stable starting point for the Subject 6 classifier. |

---

## 13. Bibliography

### Primary Paper - The SimCLR Framework:
> Chen, T., Kornblith, S., Norouzi, M., & Hinton, G. (2020). A Simple Framework for Contrastive Learning of Visual Representations. Proceedings of the International Conference on Machine Learning (ICML).

### Secondary Paper - Application of SSL to EEG:
> Mohsenvand, M. N., Izadi, M. R., & Maes, P. (2020). Contrastive Representation Learning for Electroencephalogram Classification. Machine Learning for Health (ML4H).
*(This paper validates that SimCLR-style augmentations like Gaussian noise and channel masking are mathematically sound for preserving EEG label integrity).*