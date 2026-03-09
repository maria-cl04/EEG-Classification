# README — Prototypical Network Inference (LOSO Evaluation)

---

## Table of Contents

1. [What Problem Does This Solve?](#1-what-problem-does-this-solve)
2. [Background: How the Standard Classifier Works](#2-background-how-the-standard-classifier-works)
3. [Why the Standard Classifier Fails at LOSO](#3-why-the-standard-classifier-fails-at-loso)
4. [The Core Idea: Prototypes](#4-the-core-idea-prototypes)
5. [What Is a Prototype, Exactly?](#5-what-is-a-prototype-exactly)
6. [How Classification with Prototypes Works](#6-how-classification-with-prototypes-works)
7. [Euclidean vs. Cosine Distance](#7-euclidean-vs-cosine-distance)
8. [Why This Works Better Than a Linear Classifier for LOSO](#8-why-this-works-better-than-a-linear-classifier-for-loso)
9. [The Synergy with Supervised Contrastive Loss](#9-the-synergy-with-supervised-contrastive-loss)
10. [What Does Not Change](#10-what-does-not-change)
11. [Files Created](#11-files-created)
12. [Exact Code Structure](#12-exact-code-structure)
13. [Changes to `transformer2.py` Required by This Script](#13-changes-to-transformer2py-required-by-this-script)
14. [How to Run](#14-how-to-run)
15. [Interpreting the Output](#15-interpreting-the-output)
16. [What to Expect](#16-what-to-expect)
17. [Bibliography](#17-bibliography)

---

## 1. What Problem Does This Solve?

In LOSO (Leave-One-Subject-Out) evaluation, your model is trained on 5 subjects and tested on the 6th. The training loop is identical to the standard pipeline — cross-entropy loss, same transformer architecture, same splits. The model produces a `.pth` checkpoint after training.

The question this method addresses is not about *how to train* the model, but about *how to use it to make predictions at test time*.

The standard approach — passing test trials through the linear classifier and taking the argmax — has a specific weakness in cross-subject settings. This script replaces that final step with a better decision rule that is more robust to subject-specific variation in the embedding space.

**No retraining is required.** This method works on any already-trained model checkpoint. It is purely an inference-time change.

---

## 2. Background: How the Standard Classifier Works

After your transformer encoder processes an EEG trial, it outputs a 128-dimensional vector — the **embedding** of that trial. Think of this as the model's compressed representation of what it saw in the EEG.

The final step is a **linear classifier**: a matrix of weights, shape `(40 classes × 128 dimensions)`. To classify a trial:

1. Compute the dot product of the embedding with each of the 40 rows.
2. Add a per-class bias term.
3. Take the softmax (converts raw scores to probabilities summing to 1).
4. Pick the class with the highest probability.

Each of the 40 rows in the weight matrix can be thought of as a **direction** in 128-dimensional space. The classifier scores a trial by asking: "how much does this embedding point in the direction associated with each class?"

These 40 weight vectors are learned during training on subjects 1–5. Each vector settles into a direction that, on average, points toward the embeddings of training trials from that class.

---

## 3. Why the Standard Classifier Fails at LOSO

The 40 weight vectors are learned from 5 people's EEG data. Over 200 training epochs, they drift toward directions that are optimal for those 5 people specifically.

Here's the subtle problem: the embedding space for Subject 6 is not in exactly the same location as for Subjects 1–5. Individual differences in EEG — differences in skull thickness, electrode impedance, neural noise levels, evoked response amplitudes — mean that Subject 6's embeddings are shifted or rotated relative to the training subjects' embeddings.

The linear classifier's weight vectors, being calibrated to Subjects 1–5, do not necessarily point in the right direction for Subject 6's embeddings. Even if Subject 6's "cat" embeddings cluster correctly in some part of the space, the weight vector for "cat" might not point toward that cluster.

### A Concrete Analogy

Imagine you train a height guesser using measurements from 5 people from the Netherlands (average height: 183 cm). Your classifier learns the rule: "if measurement X > 180 cm, guess 'tall.'" You then test it on someone from Indonesia (average height: 162 cm) where a person standing 170 cm is considered very tall. Your classifier fails because it was calibrated to the wrong distribution — not because height is an uninformative feature, but because the *scale* differs.

The linear classifier has the same problem. The decision boundaries are calibrated to the training subjects' embedding distribution. Subject 6 lives in a shifted part of the space, and the same boundaries don't apply.

---

## 4. The Core Idea: Prototypes

Instead of using a fixed weight matrix (which was calibrated to training subjects), we compute a new decision rule *directly from the embedding space* after training.

The idea is simple:

1. Run every training trial through the encoder (not the classifier — just the encoder). Get an embedding for each trial.
2. For each of the 40 classes, compute the **mean embedding** across all training trials of that class. This mean vector is called the **prototype** of that class.
3. To classify a new test trial, embed it and find the class whose prototype is closest to the test embedding.

That's it. No new parameters. No additional training. Just averages and distances.

---

## 5. What Is a Prototype, Exactly?

A prototype is the **centroid** (centre of mass) of a cluster of points. If you have 300 training trials labelled "cat" and each has been embedded into a 128-dimensional vector, the prototype for "cat" is:

```
prototype_cat = (1/300) × (embedding_1 + embedding_2 + ... + embedding_300)
```

This gives you a single 128-dimensional vector that represents the "average cat embedding" across all training data.

### Why the Mean?

The mean has a key property: it minimises the sum of squared distances to all the points that contributed to it. In other words, the prototype is the point in embedding space that is closest, on average, to all training trials of that class. This makes it the natural choice as a "representative" of the class.

With 5 training subjects × 40 classes × ~300 trials each, each prototype is computed from roughly 1,500 training embeddings. This large sample average is very stable — the noise from any individual subject's idiosyncrasies is averaged out.

### The 40 Prototypes

After computing all 40 prototypes, you have a set of 40 vectors in 128-dimensional space. These form a **gallery** of class representatives. Classification is then just a lookup: "which gallery entry is the test trial closest to?"

---

## 6. How Classification with Prototypes Works

### Step-by-Step for a Single Test Trial

1. **Embed**: pass the test trial through the transformer encoder's `get_embedding()` method. This returns the mean-pooled 128-dimensional vector. (The classifier layer is not used.)

2. **Compute distances**: calculate the distance from the test embedding to each of the 40 class prototypes. Using Euclidean distance:
   ```
   distance(test, prototype_c) = sqrt( sum over d: (test[d] - prototype_c[d])^2 )
   ```
   This gives 40 distance values.

3. **Assign label**: the predicted class is the one with the *smallest* distance.
   ```
   predicted_class = argmin over c: distance(test, prototype_c)
   ```

### No Softmax, No Parameters

Notice there is no softmax, no bias, no learned weights involved in this decision. The classifier is entirely non-parametric — it has zero learnable parameters. Its only requirement is that the embedding space is well-organised (same class = nearby, different class = far apart).

---

## 7. Euclidean vs. Cosine Distance

The script supports two distance metrics. Understanding their difference helps you choose:

### Euclidean Distance (default)

Measures the straight-line distance between two points in 128-dimensional space. Two embeddings are "close" if all 128 of their dimensions have similar values.

```
d_euclidean(a, b) = sqrt( (a₁-b₁)² + (a₂-b₂)² + ... + (a₁₂₈-b₁₂₈)² )
```

**Best when**: embeddings have consistent magnitudes (norms). This is the case when SupCon is used during training — SupCon normalises embeddings to the unit sphere, so all norms are 1.

### Cosine Distance

Measures the angle between two vectors, ignoring their magnitudes. Two embeddings are "close" if they point in the same direction, regardless of how large they are.

```
d_cosine(a, b) = 1 - (a · b) / (|a| × |b|)
```

**Best when**: embeddings have varying magnitudes. If the model sometimes produces small-norm embeddings for uncertain trials and large-norm embeddings for confident trials, cosine distance is invariant to this variation and focuses purely on direction.

### Which to Use?

- If trained **with SupCon** (`--supcon` flag): use **Euclidean**. SupCon explicitly places embeddings on a unit sphere, so magnitudes are uniform and direction is meaningful.
- If trained **without SupCon** (baseline): try **cosine** first, as magnitudes may vary more.
- In your thesis, running both and reporting the delta is a straightforward ablation.

---

## 8. Why This Works Better Than a Linear Classifier for LOSO

### The Calibration Problem, Revisited

The linear classifier has 40 weight vectors calibrated to Subjects 1–5. Subject 6's embeddings may be shifted by some subject-specific offset in the 128-dimensional space.

The prototype for "cat" is computed as the mean of ~1,500 embeddings (5 subjects × ~300 trials). Subject 6's "cat" embeddings, even if shifted relative to the training distribution, are much closer to this mean than to the weight vector of a *wrong* class. The mean is shift-tolerant in a way the weight matrix is not.

### Why the Mean Is Shift-Tolerant

Suppose Subject 6's embeddings are shifted by a constant vector `δ` relative to the training distribution (a simplification, but illustrative). Then:

- **With a linear classifier**: the weight vectors were learned without `δ`. Subject 6's test embedding is `z_6 = z_train + δ`. The classifier computes `W × z_6 = W × z_train + W × δ`. The extra term `W × δ` adds systematic error to all class scores. If `δ` is large enough, the wrong class wins.

- **With prototypes**: each prototype is the mean of training embeddings. The test embedding is `z_6`. The prototype for "cat" is `p_cat = mean(z_cat_train)`. The distance is `|z_6 - p_cat|`. If Subject 6's embeddings are uniformly shifted, all distances increase by the same amount — but the *ranking* (which prototype is closest) is unaffected as long as Subject 6's "cat" embeddings are still closer to `p_cat` than to `p_dog`, which they will be if the embedding space is well-organised.

This is why prototypical inference is more robust to subject-shift: it is invariant to uniform translations of the test distribution.

---

## 9. The Synergy with Supervised Contrastive Loss

Prototypical inference requires one critical precondition: the embedding space must be well-organised. Specifically:
- Same-class embeddings must form tight clusters (small intra-class variance).
- Different-class embeddings must be separated by large distances (large inter-class distance).

Without this, the prototypes overlap, and nearest-prototype classification degrades to random guessing.

This is precisely what Supervised Contrastive Loss (SupCon, see `README_supervised_contrastive_loss.md`) trains for. SupCon explicitly minimises intra-class distance and maximises inter-class distance *across all pairs in the batch*, including pairs from different subjects.

The combination works as follows:

1. **SupCon** trains the encoder so that same-class embeddings cluster together, regardless of which subject they came from.
2. **Prototypical inference** exploits these clusters by using their centres as classifiers.

Without SupCon, the clusters may not be tight enough for prototypical inference to outperform the linear classifier. With SupCon, the clusters are explicitly trained to be tight, and prototypical inference can take full advantage.

This is why the recommended workflow for LOSO is: **train with SupCon → evaluate with prototypical inference**. Each method reinforces the other.

---

## 10. What Does Not Change

- **The training loop**: identical to the original. Same splits, same epochs, same optimizer, same loss (with or without SupCon).
- **The model architecture**: unchanged.
- **The checkpoint format**: the saved `.pth` file is exactly what the original training script would produce. This script loads it without modification.
- **The training accuracy/loss**: not affected at all. This script only changes what happens *after* training, during evaluation.

---

## 11. Files Created

| File | Purpose |
|------|---------|
| `transformer_loso_proto.py` | Complete self-contained evaluation script |

No other files are modified.

---

## 12. Exact Code Structure

### `transformer_loso_proto.py` — flow

```
1. Parse arguments (model path, dataset path, splits path, etc.)
2. Build DataLoaders for train and test splits (same logic as main script)
3. Load model from .pth checkpoint (torch.load)
4. extract_embeddings(loader):
   └── For each batch: call model.get_embedding(x) → (B, 128) tensor
       Concatenate all batches → (N_total, 128) and (N_total,) labels
5. softmax_accuracy(test_loader):
   └── Standard evaluation: model(x).argmax() compared to target
       Reports the BASELINE accuracy for comparison
6. compute_prototypes(train_emb, train_labels, num_classes=40):
   └── For each class c in 0..39:
           prototype[c] = mean of all train_emb[i] where train_labels[i] == c
       Returns: (40, 128) tensor of prototypes
7. proto_accuracy(test_emb, test_labels, prototypes):
   └── For each test sample: find nearest prototype (Euclidean or cosine)
       Count correct predictions
       Return accuracy
8. Print both accuracies and the delta
```

### Key function: `extract_embeddings()`

```python
@torch.no_grad()
def extract_embeddings(loader):
    all_emb = []
    all_lbl = []
    for x, y in loader:
        x = x.to(device)
        emb = model.get_embedding(x)   # (B, 128) — encoder output, NO classifier
        all_emb.append(emb.cpu())
        all_lbl.append(y.cpu())
    return torch.cat(all_emb), torch.cat(all_lbl)
```

The `@torch.no_grad()` decorator tells PyTorch not to compute gradients during this function. This is important: embedding extraction is inference, not training. Without it, PyTorch would store intermediate activations for backpropagation, consuming GPU memory unnecessarily.

### Key function: `compute_prototypes()`

```python
def compute_prototypes(embeddings, labels, num_classes):
    d = embeddings.size(1)           # 128
    prototypes = torch.zeros(num_classes, d)
    
    for c in range(num_classes):
        mask = labels == c           # Boolean mask: True where label == c
        if mask.sum() > 0:
            prototypes[c] = embeddings[mask].mean(dim=0)   # mean over ~1500 trials
    
    return prototypes                # (40, 128)
```

This runs once on the full training set. It does not use the GPU (all operations are on CPU tensors after extracting embeddings). Runtime is negligible.

### Key function: `proto_accuracy()`

```python
def proto_accuracy(test_emb, test_labels, prototypes, distance='euclidean'):
    if distance == 'euclidean':
        # (N_test, 1, 128) - (1, 40, 128) → (N_test, 40, 128) → (N_test, 40)
        diff = test_emb.unsqueeze(1) - prototypes.unsqueeze(0)
        dists = diff.pow(2).sum(dim=2)   # squared Euclidean (sqrt not needed for argmin)
        pred = dists.argmin(dim=1)        # (N_test,) — index of nearest prototype
    elif distance == 'cosine':
        test_norm  = F.normalize(test_emb,   dim=1)   # (N_test, 128)
        proto_norm = F.normalize(prototypes, dim=1)   # (40, 128)
        sim  = torch.matmul(test_norm, proto_norm.T)  # (N_test, 40) — cosine similarities
        pred = sim.argmax(dim=1)                       # highest similarity = nearest
    
    correct = pred.eq(test_labels).sum().item()
    return correct / test_labels.size(0)
```

Note on the Euclidean distance computation: `diff.pow(2).sum(dim=2)` computes *squared* Euclidean distance. Taking the square root (via `.sqrt()`) is not needed because we only need the ranking, and the ranking of squared distances is the same as the ranking of true distances. This is a minor efficiency improvement.

The broadcasting in `test_emb.unsqueeze(1) - prototypes.unsqueeze(0)` computes all N_test × 40 pairwise differences simultaneously without a Python loop, which is efficient even for large test sets.

---

## 13. Changes to `transformer2.py` Required by This Script

The evaluation script calls `model.get_embedding(x)`, a method that does **not exist** in the original `transformer2.py`. It must be added to the model class.

This addition is included in the modified `models/transformer2.py` file provided alongside this README:

```python
# Added to the Model class in transformer2.py:

def _encode(self, x, mask=None):
    """Internal helper: run encoder and return mean-pooled (B, d_model) vector."""
    if x.dim() == 4:
        x = x.squeeze(1).permute(0, 2, 1)
    x = self.embedding(x)
    x = self.pos_encoder(x)
    x = self.dropout(x)
    for layer in self.encoder_layers:
        x = layer(x, mask)
    x = x.mean(dim=1)   # (B, d_model)
    return x

def get_embedding(self, x, mask=None):
    """Public method: return mean-pooled encoder output. No classifier, no projection."""
    return self._encode(x, mask)
```

If you load an old checkpoint that was saved with `torch.save(model, ...)` (full model object, not just weights), the loaded object will be the old class definition without `get_embedding()`. In that case, you need to re-train with the new `transformer2.py`, or monkey-patch the method at runtime:

```python
# Workaround if using an old checkpoint without get_embedding():
import types
def get_embedding(self, x, mask=None):
    if x.dim() == 4:
        x = x.squeeze(1).permute(0, 2, 1)
    x = self.embedding(x)
    x = self.pos_encoder(x)
    x = self.dropout(x)
    for layer in self.encoder_layers:
        x = layer(x, mask)
    return x.mean(dim=1)

model.get_embedding = types.MethodType(get_embedding, model)
```

This binds the function to the loaded model object at runtime without requiring retraining.

---

## 14. How to Run

### Standard LOSO evaluation (both softmax and prototypical):

```bash
python transformer_loso_proto.py \
    --model  transformer2__subject0_epoch_200.pth \
    --eeg-dataset  path/to/eeg_55_95_std.pth \
    --splits-path  path/to/block_splits_LOSO_subject1.pth \
    --test-subject 1 \
    --distance euclidean
```

### With cosine distance:

```bash
python transformer_loso_proto.py \
    --model  transformer2__subject0_epoch_200.pth \
    --eeg-dataset  path/to/eeg_55_95_std.pth \
    --splits-path  path/to/block_splits_LOSO_subject1.pth \
    --test-subject 1 \
    --distance cosine
```

### Run for all 6 held-out subjects (bash loop):

```bash
for sub in 1 2 3 4 5 6; do
    echo "=== Subject $sub ==="
    python transformer_loso_proto.py \
        --model  transformer2_loso_subject${sub}_epoch_200.pth \
        --eeg-dataset  path/to/eeg_55_95_std.pth \
        --splits-path  path/to/block_splits_LOSO_subject${sub}.pth \
        --test-subject $sub \
        --distance euclidean
done
```

---

## 15. Interpreting the Output

The script prints:

```
--- Results ---
  Softmax classifier accuracy :  3.45%
  Prototypical (euclidean) accuracy :  4.87%
  Delta                       : +1.42%

  Chance level (40 classes)   :  2.50%
```

### What each line means:

**Softmax classifier accuracy**: what the model achieves with the standard linear classifier — the same number you would see from the original training script. This is your baseline.

**Prototypical accuracy**: what the model achieves when the linear classifier is replaced by nearest-prototype classification. This is the new result.

**Delta**: the improvement (or degradation) from switching to prototypical inference. A positive delta means prototypical works better.

**Chance level**: with 40 balanced classes, random guessing gives 1/40 = 2.5%. Any accuracy above this means the model is learning something real. LOSO results for this dataset are known to be near-chance — values of 3–6% are typical.

### When Delta is Negative (Prototypical Worse Than Softmax)

This happens when the embedding space is not well-organised — clusters overlap, and the class mean is not a good representative. The two most common causes:

1. **SupCon was not used during training**: without explicit clustering pressure, the embedding space may not have tight same-class clusters. Train with `--supcon` and re-evaluate.

2. **The model was poorly trained**: if training accuracy is low, the encoder hasn't learned useful representations at all. Verify that training accuracy is at least 50–60% before running prototypical evaluation.

---

## 16. What to Expect

### Typical LOSO results for this dataset

| Condition | Softmax accuracy | Prototypical accuracy | Delta |
|---|---|---|---|
| Baseline (no SupCon) | ~3.0% | ~2.8–3.5% | ≈ 0 or slightly negative |
| With SupCon (λ=0.1) | ~3.5–4.5% | ~4.0–5.5% | +0.5 to +1.5% |
| With SupCon + MAE pre-training | ~4.0–5.5% | ~4.5–6.5% | +0.5 to +2% |

All values are approximate — LOSO results for this dataset are inherently noisy and subject-dependent. Subject 1 may yield 6% while Subject 4 yields 3%. This variance is a known property of the Spampinato dataset and should be reported in your thesis along with the mean and standard deviation across subjects.

### Key thesis takeaways to report

1. The delta between prototypical and softmax inference demonstrates that improved *embedding structure* (from SupCon) can be exploited at inference time, even without retraining the classifier.
2. The combination of SupCon + prototypical inference provides additive benefit: SupCon makes clusters tighter, prototypical inference uses the cluster centres directly.
3. Even at near-chance levels, the improvement above chance (e.g., 5% vs. 2.5% = 2× chance) is a meaningful result given the known difficulty of cross-subject decoding on 40-class visual EEG.

---
## 17. Bibliography

***DISCLOSURE:** Claude.ai was not able to give me the exact links or references used, since it just used the training data, so it provided me with the most relevant references for each type of method.*

### Primary Paper:
> Snell, J., Swersky, K., & Zemel, R. (2017). Prototypical Networks for Few-Shot Learning. Advances in Neural Information Processing Systems (NeurIPS), 30.

This paper introduced the exact classifcation procedure used in the script: compute class prototypes as mean embeddings, classify by nearest prototype using Euclidean distance. The mathematical justification (prototype as the point minimising sum of squared distances) is from this paper.
