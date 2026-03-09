# README — Masked Autoencoder (MAE) Self-Supervised Pre-training

---

## Table of Contents

1. [What Problem Does This Solve?](#1-what-problem-does-this-solve)
2. [Background: What Is Pre-training?](#2-background-what-is-pre-training)
3. [The Core Idea: Masked Autoencoding](#3-the-core-idea-masked-autoencoding)
4. [Why Masking Works Better Than Other Approaches](#4-why-masking-works-better-than-other-approaches)
5. [The Encoder-Decoder Architecture](#5-the-encoder-decoder-architecture)
6. [Why 75% Masking?](#6-why-75-masking)
7. [The Training Signal: MSE on Masked Positions](#7-the-training-signal-mse-on-masked-positions)
8. [What Gets Saved and What Gets Thrown Away](#8-what-gets-saved-and-what-gets-thrown-away)
9. [Why This Helps Each Experiment Type](#9-why-this-helps-each-experiment-type)
10. [Files Created](#10-files-created)
11. [Architecture in Detail](#11-architecture-in-detail)
12. [Exact Code Structure](#12-exact-code-structure)
13. [Integration with the Main Training Script](#13-integration-with-the-main-training-script)
14. [Hyperparameters and How to Tune Them](#14-hyperparameters-and-how-to-tune-them)
15. [How to Run](#15-how-to-run)
16. [What to Expect](#16-what-to-expect)

---

## 1. What Problem Does This Solve?

Your four experiments vary in how many labelled samples the model can learn from:

| Experiment | Training data available |
|---|---|
| Multi-subject | ~6 subjects × 40 classes × 300 trials = ~72,000 labelled trials |
| Single-subject | ~40 classes × 300 trials = ~12,000 labelled trials |
| LOSO | 5 subjects × ~12,000 = ~60,000 labelled trials, but **tested on a different subject** |
| Fine-tuning | 5 subjects for pre-train; only a fraction of 1 subject's data for fine-tune |

The scarcest situations — LOSO and fine-tuning — share a common problem: **the model must generalise to a subject it either hasn't seen or has seen very little of.** The bottleneck is not the total amount of data; it's the amount of *labelled* data from the test subject.

But notice something: you have *unlabelled* EEG data from all 6 subjects at no extra cost — the same recordings, just without using the category labels. MAE pre-training exploits this by running a completely different training task (predict masked time-steps) that requires no labels at all. After this task, the encoder has learned the structure of EEG signals across all subjects — before a single label has been shown.

---

## 2. Background: What Is Pre-training?

Pre-training means training a model on one task, then transferring the learned weights to a different (usually harder or label-scarce) task.

The most famous example is GPT and BERT for language. These models are trained on billions of words of raw text (predict the next word / predict masked words) before being fine-tuned on specific tasks like sentiment analysis or question answering. The pre-training phase teaches the model the structure of language; the fine-tuning phase teaches it to apply that structure to a specific problem.

The same principle applies here. EEG signals have structure — oscillatory rhythms, evoked response components (like the P300 wave), temporal correlations between adjacent time-steps. This structure does not depend on what image the subject was looking at. It exists in all EEG recordings, labelled or not. Pre-training forces the encoder to learn this structure by making prediction of masked regions impossible without understanding it.

### The Key Insight

A model that already understands EEG structure needs far less labelled data to learn the category-discriminative part. Think of the problem as two subtasks:

1. **Understanding EEG** (what does an EEG signal look like? what are its temporal patterns?): hard, but requires no labels, and can use all available data.
2. **Linking EEG patterns to visual categories** (which patterns correspond to "looking at a cat"?): easier once task 1 is solved, and requires labels.

Standard training tries to solve both simultaneously from labelled data only. Pre-training solves task 1 first, from all available data, then uses labelled data only for task 2.

---

## 3. The Core Idea: Masked Autoencoding

An **autoencoder** is a model trained to compress and then reconstruct its own input. The "auto" in autoencoder means the model is its own teacher — it just has to produce the same thing it was given as input.

A **masked** autoencoder adds a twist: before the model sees the input, some parts of it are randomly hidden (masked). The model must reconstruct the complete original from the incomplete version.

### Concrete Example for Your EEG Data

Take one EEG trial: a tensor of shape `(440 time-steps, 128 channels)`.

1. Randomly select 75% of time-steps (= 330 out of 440). These are the **masked positions**.
2. Show the model only the remaining 25% (= 110 time-steps). These are the **visible positions**.
3. Ask the model to reconstruct all 440 time-steps — including the 330 it never saw.
4. Measure the error only on the masked positions (mean squared error between prediction and ground truth).
5. Backpropagate. Repeat for millions of trials.

There are no labels. No category information. The only signal is: "how well did you reconstruct the parts of the EEG you couldn't see?"

### Why This Forces Useful Learning

To do well at this task, the model must learn:
- **Temporal structure**: EEG at time-step 200 is correlated with EEG at time-step 150 and 250. The model must learn this to make good predictions.
- **Oscillatory patterns**: Alpha waves (8–12 Hz), gamma bursts (55–95 Hz in your dataset), and event-related synchronisation all follow predictable rhythmic patterns. The model must infer them from partial observations.
- **Cross-channel relationships**: EEG channels are spatially arranged on the scalp and are correlated. Occipital channels (visual cortex) are highly correlated during visual stimulation. The model must learn these cross-channel dependencies.
- **Evoked response timing**: After a visual stimulus, the EEG follows a stereotyped sequence of components (N100, P200, N200, P300). These occur at specific time-offsets relative to the stimulus onset. The model must learn these to predict masked time-steps in the evoked response window.

Crucially, all of this structure is **shared across subjects**. The physics of evoked potentials doesn't change between individuals. The gamma rhythm exists in everyone. By learning to predict masked regions, the model is forced to internalise the subject-invariant EEG structure.

---

## 4. Why Masking Works Better Than Other Approaches

There are other self-supervised learning approaches — why MAE specifically?

### Alternative: Predict the Next Time-step (Autoregressive)
Like GPT for text: train the model to predict time-step `t+1` given time-steps `0…t`. This works but has a problem for transformers: it requires **causal masking** (the model cannot see the future), which means the encoder can't use full bidirectional context. Your transformer is a bidirectional encoder — it attends to all time-steps simultaneously. Forcing causal masking would underuse this capacity.

### Alternative: Add Gaussian Noise (Denoising Autoencoder)
Corrupt the entire signal with random noise and ask the model to denoise it. This can work, but the task is much easier — the model can always "see" every time-step (just corrupted). The gradient signal is spread thin across all positions, not concentrated on genuinely hard predictions.

### MAE's Advantage: Hard Task, Concentrated Gradient
By masking 75% completely (making them invisible, not just noisy), the task is genuinely hard. The model must make long-range inferences from a sparse set of observations. The loss only counts on the masked positions, so the gradient signal is concentrated where the learning happens. This is why MAE produces far richer representations than simpler autoencoder variants.

### Why Transformers Are Especially Good at This Task
Transformers use self-attention, which means every visible time-step can directly attend to every other visible time-step in a single layer. When predicting masked positions, the model can flexibly combine information from any combination of the 110 visible time-steps. A recurrent network (LSTM) would have to sequentially pass information forward, making long-range inference harder. The match between MAE and transformer architecture is part of why this combination works so well.

---

## 5. The Encoder-Decoder Architecture

MAE uses an asymmetric encoder-decoder design. This is one of its most important features.

```
Input: Full EEG trial (440 × 128)
           │
           ▼
    ┌──────────────┐
    │   MASKING    │  ← randomly hide 75% of time-steps
    └──────────────┘
           │
     110 visible time-steps (440 × 25%)
           │
           ▼
    ┌──────────────────────────────┐
    │   ENCODER  (kept after      │  ← your full transformer
    │   pre-training)             │     d_model=128, num_layers=1
    │                             │     processes only 110 time-steps
    └──────────────────────────────┘
           │
     110 encoded tokens (110 × 128)
           │
           ▼
    ┌──────────────────────────────┐
    │   DECODER  (thrown away     │  ← small, cheap transformer
    │   after pre-training)       │     d_model=64, num_layers=2
    │                             │     processes all 440 positions
    └──────────────────────────────┘
           │
     Reconstruction (440 × 128)
           │
           ▼
    MSE loss on the 330 masked positions only
```

### Why Is the Decoder Small?

The encoder does the expensive work of extracting representations from the visible tokens. If the decoder were also large and powerful, it could reconstruct the masked positions without the encoder needing to do anything meaningful — the decoder would essentially re-do the encoder's job.

By making the decoder small (64-dimensional hidden states vs. 128 for the encoder, only 2 layers vs. 1 for the encoder), the decoder cannot compensate for a weak encoder. The encoder is forced to produce rich, informative representations from the 110 visible tokens — representations that contain enough information about EEG structure that even a simple decoder can reconstruct the missing 330.

### The Mask Token

The decoder processes all 440 positions, not just the 110 that the encoder processed. For the 330 masked positions, the decoder uses a **learned mask token** — a single vector (initialised randomly, trained with the rest of the network) that serves as a placeholder. Every masked position gets the same mask token as its starting representation; the decoder's self-attention then fills in the content from the encoded visible tokens.

This is similar to how BERT uses a `[MASK]` token in language, but here the mask token is a continuous vector rather than a discrete vocabulary entry.

---

## 6. Why 75% Masking?

This number comes from the original MAE paper (He et al., 2022) for images, and it transfers well to EEG for a specific reason.

### The Difficulty Calibration Argument

- At **15% masking** (like BERT for text): the task is too easy. EEG at time-step 200 can be almost perfectly predicted from its neighbours at 195, 197, 202, 205 by interpolation. The model learns to interpolate, not to understand structure. This was explicitly demonstrated in the MAE paper — 15% masking produces worse representations than 75%.

- At **50% masking**: better, but still somewhat easy due to EEG's high temporal autocorrelation. The model can usually find nearby visible time-steps that are highly correlated with the masked ones.

- At **75% masking**: the model typically has no visible time-steps within the same local neighbourhood as a masked one. It must make long-range inferences. To predict what happens during the evoked response peak (time-steps ~180–250 post-stimulus), it must infer from the baseline period (time-steps 20–150) and the later periods. This requires understanding the *causal structure* of visual evoked potentials — exactly what you want the encoder to learn.

- At **90% masking**: too difficult. With only 44 visible time-steps out of 440, even a perfect model would struggle because many predictions become genuinely ambiguous (there are multiple valid EEG continuations from so few observations).

75% is the sweet spot for EEG: hard enough to require genuine structural understanding, easy enough that the training signal is not dominated by irreducible noise.

---

## 7. The Training Signal: MSE on Masked Positions

The loss function is simple: mean squared error between the decoder's predictions and the true (original, unmasked) EEG values, computed only at the masked positions.

```python
bool_mask = ...  # True at masked positions, False at visible
loss = MSE( prediction[bool_mask], original[bool_mask] )
```

Why only masked positions? Because the visible positions are given to the encoder unchanged — measuring reconstruction error there would just verify that the model can copy its input, which is trivially easy and uninformative.

### What the MSE Actually Measures

Your EEG data is standardised (`eeg_55_95_std.pth` — the `std` means it has been z-scored per channel). So the MSE is measuring error in units of standard deviations. A loss of 1.0 means predictions are off by one standard deviation on average. After good pre-training, this should converge to around 0.3–0.5 (you cannot predict EEG perfectly because there is genuine trial-to-trial variability).

---

## 8. What Gets Saved and What Gets Thrown Away

After pre-training:

**Saved**: the encoder weights — specifically, the weights of:
- `embedding` (Linear: 128 → 128)
- `pos_encoder` (PositionalEncoding buffers)
- `encoder_layers` (the transformer blocks)
- `dropout`

**Discarded**: the decoder (all of it). It served its purpose during pre-training and is never used again. The saved file `pretrained_encoder.pth` contains only the encoder's `state_dict`.

**When loading into the main training script**: `model.load_state_dict(state, strict=False)`. The `strict=False` is essential — the classifier and projection head exist in the full `Model` but not in the encoder checkpoint. `strict=False` tells PyTorch to load what it can and ignore missing/unexpected keys. The missing keys (classifier, projection head) will be randomly initialised and trained from scratch on the labelled data.

This is called **transfer learning**: the encoder's knowledge about EEG structure transfers; the task-specific classifier is learned anew.

---

## 9. Why This Helps Each Experiment Type

### Multi-subject
Moderate benefit. The model already sees plenty of data, so the pre-training advantage is smaller. However, the encoder starts with better-organised internal representations, so convergence is faster and the final accuracy may be marginally higher.

### Single-subject
Small benefit. With ~12,000 labelled trials per subject, the model can learn EEG structure from the labelled data alone. Pre-training helps mainly by speeding up convergence.

### LOSO (most impactful)
Large benefit. The encoder is pre-trained on all subjects' EEG without labels — it learns EEG patterns that generalise across everyone. When then fine-tuned on 5 subjects with labels, the classifier only needs to learn the category → EEG pattern mapping, building on an already cross-subject encoder. Testing on the 6th subject, the encoder's representations are more familiar-looking to the held-out subject than if it had been trained purely on labelled data.

### Fine-tuning (most impactful alongside LOSO)
Large benefit. In the fine-tuning setup, you have only a fraction of one subject's labelled data to adapt to them. Without pre-training, this small amount of data is used to simultaneously learn EEG structure and the category mapping — an inefficient use of limited labels. With pre-training, EEG structure is already learned; the small fine-tuning set only needs to teach the category mapping and any subject-specific adjustments.

---

## 10. Files Created

| File | Purpose |
|------|---------|
| `transformer_pretrain.py` | Complete self-contained pre-training script |

No other files are modified. The pre-training is entirely separate from the main training script. The connection is through the saved `.pth` encoder weights.

---

## 11. Architecture in Detail

### Encoder (`MAEEncoder`)

Identical in structure to the encoder portion of `transformer2.Model`:

| Component | Config |
|---|---|
| Input projection | Linear(128, 128) |
| Positional encoding | Sinusoidal, max_seq=440 |
| Encoder layers | 1 × EncoderLayer |
| Each EncoderLayer | MultiHeadAttention (4 heads) + FFN (512-dim) + LayerNorm |
| Dropout | 0.4 |
| Output | (B, n_visible, 128) — NOT mean-pooled (decoder needs all tokens) |

Note: in the pre-training encoder, there is no mean pooling and no classifier. The output is the full sequence of encoded visible tokens.

### Decoder (`MAEDecoder`)

Small and cheap by design:

| Component | Config |
|---|---|
| Projection | Linear(128, 64) — encoder to decoder dimension |
| Mask token | Learned parameter, shape (1, 1, 64) |
| Positional encoding | Sinusoidal over full 440 positions |
| Decoder layers | 2 × EncoderLayer (self-attention over all 440 positions) |
| Prediction head | Linear(64, 128) — reconstruct EEG channels |
| Output | (B, 440, 128) — reconstruction of full trial |

The decoder uses the same `EncoderLayer` building block as the encoder (self-attention + FFN + LayerNorm), but with `d_model=64` and `d_ff=128` instead of `d_model=128` and `d_ff=512`. It has no dropout (`dropout=0.0`) because it is a reconstruction head, not a classifier — overfitting during pre-training is not a concern.

---

## 12. Exact Code Structure

### `transformer_pretrain.py` — flow

```
1. Parse arguments
2. Load EEGPretrainDataset
   └── loads ALL trials from eeg_*.pth, no subject filtering, no labels
       filters by EEG length (450–600 samples, same as main script)
3. Build MAEEncoder (same architecture as transformer2 encoder)
4. Build MAEDecoder (small asymmetric decoder)
5. Build MaskedAutoencoder wrapper (encoder + decoder + masking logic)
6. Pre-training loop (AdamW + cosine LR schedule):
   a. For each batch of EEG trials (B, 440, 128):
   b.   random_masking(): randomly select 75% of time-steps to mask
        returns: x_visible (B, 110, 128), mask_idx (B, 330)
   c.   encoder(x_visible) → enc_tokens (B, 110, 128)
   d.   decoder(enc_tokens, mask_idx) → pred (B, 440, 128)
   e.   loss = MSE(pred[masked], original[masked])
   f.   backward + optimizer step
7. Save encoder.state_dict() to pretrained_encoder.pth
   (decoder is not saved)
```

### Key function: `random_masking()`

```python
def random_masking(self, x):
    B, T, C = x.shape
    n_masked = int(T * self.mask_ratio)   # e.g., 330 for T=440, ratio=0.75
    
    noise = torch.rand(B, T)              # different random mask per sample in batch
    shuffle_idx = noise.argsort(dim=1)   # random permutation of time indices
    
    mask_idx = shuffle_idx[:, :n_masked]  # first 330 indices (to be masked)
    keep_idx = shuffle_idx[:, n_masked:]  # last 110 indices (visible)
    
    # Gather only the visible time-steps
    x_visible = gather(x, keep_idx)       # (B, 110, 128)
    return x_visible, mask_idx
```

Each sample in the batch gets an *independent* random mask — there's no correlation between which time-steps are masked across samples. This is important: if all samples in a batch had the same mask, the decoder could potentially memorise a reconstruction for "the 330 positions that are always masked," which would not generalise.

### Optimiser: AdamW + Cosine Annealing

Standard MAE training uses AdamW (Adam with weight decay) rather than plain Adam, because weight decay acts as L2 regularisation on the pre-trained weights. This prevents the encoder from overfitting to reconstruction artefacts. A cosine annealing schedule reduces the learning rate smoothly to near-zero over 100 epochs, which is known to improve final representation quality.

---

## 13. Integration with the Main Training Script

After pre-training, add this block to `transformer_eeg_signal_classification.py` immediately after `model = module.Model(...)`:

```python
# NEW — load pre-trained encoder weights before supervised training
if opt.pretrained_encoder != '':
    state = torch.load(opt.pretrained_encoder, map_location='cuda' if not opt.no_cuda else 'cpu')
    # strict=False: classifier.weight, classifier.bias, proj_head.* 
    # are absent from the encoder checkpoint. That is expected.
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"Pre-trained encoder loaded.")
    print(f"  Keys not in checkpoint (will train from scratch): {missing}")
    print(f"  Keys in checkpoint not in model (ignored): {unexpected}")
```

And add this argument:

```python
parser.add_argument('--pretrained-encoder', default='',
                    help="Path to MAE pre-trained encoder .pth (from transformer_pretrain.py)")
```

### Why `strict=False`?

The pre-trained file contains only encoder weights. The full `Model` has additional parameters: `classifier.weight`, `classifier.bias`, `proj_head.0.weight`, etc. With `strict=True` (the default), PyTorch would raise an error because these keys are missing from the checkpoint. With `strict=False`, PyTorch loads what it finds and silently skips the rest. The skipped parameters remain at their random initialisation and are trained normally during supervised fine-tuning.

---

## 14. Hyperparameters and How to Tune Them

### `--mask-ratio` (default: 0.75)
The fraction of time-steps masked per trial.

- **0.75** is recommended (from original MAE paper). Strong enough to force structural learning.
- **0.5**: try if 0.75 gives very high pre-training loss that doesn't converge (unlikely but possible for very noisy datasets).
- **0.9**: only try if you want to push the task to the extreme — probably too hard for 440 time-steps.

### `--epochs` (default: 100)
Pre-training epochs. More is generally better up to a point.

- **50 epochs**: fast experiment to verify the pipeline works.
- **100 epochs**: good default.
- **200 epochs**: if compute allows, may give marginally better representations.

### `--lr` (default: 1e-3)
Learning rate for AdamW.

- 1e-3 is standard for transformer pre-training.
- If loss is unstable (spiky), reduce to 5e-4.

### `--decoder-d-model` (default: 64)
Hidden dimension of the decoder. Keep this smaller than the encoder's `d_model` (128). The smaller the decoder, the more the encoder must carry the representational load.

### `--decoder-layers` (default: 2)
Number of transformer layers in the decoder. 2 is standard (from original MAE paper). More layers would make the decoder more powerful and reduce the pressure on the encoder.

---

## 15. How to Run

### Step 1: Pre-train the encoder
```bash
python transformer_pretrain.py \
    --eeg-dataset path/to/eeg_55_95_std.pth \
    --epochs 100 \
    --mask-ratio 0.75 \
    --d-model 128 \
    --num-heads 4 \
    --num-layers 1 \
    --d-ff 512 \
    --dropout 0.4 \
    --save-path pretrained_encoder.pth
```

This uses ALL data in the dataset file (all subjects, all trials, no labels). Runtime: approximately 10–30 minutes on GPU for 100 epochs, depending on dataset size.

### Step 2: Use the pre-trained encoder in main training

After adding the integration code from Section 13:

```bash
# LOSO example (held-out subject 1)
python transformer_eeg_signal_classification.py \
    -sub 0 \
    -sp  block_splits_LOSO_subject1.pth \
    --pretrained-encoder pretrained_encoder.pth \
    --supcon \
    --lambda-supcon 0.1 \
    -e 200
```

---

## 16. What to Expect

### Pre-training loss curve

| Epoch | Expected MAE loss |
|---|---|
| 1 | 0.8–1.2 |
| 10 | 0.5–0.7 |
| 50 | 0.3–0.5 |
| 100 | 0.25–0.4 |

Convergence below 0.3 is excellent. If loss plateaus above 0.6, try reducing the learning rate.

### Downstream accuracy improvement

| Experiment | Expected improvement over no pre-training |
|---|---|
| Multi-subject | +0–1% |
| Single-subject | +0–1% |
| LOSO | +1–5% (most variable, subject-dependent) |
| Fine-tuning | +2–5% on test subject |

The LOSO improvement is the most uncertain because it depends heavily on how different Subject 6's EEG patterns are from the other 5. In the best case, pre-training aligns the encoder to universal EEG structure that Subject 6 shares; in the worst case, if Subject 6 is an outlier, the improvement is smaller but pre-training should never hurt.
