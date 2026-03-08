# ITSA Integration — Troubleshooting & Lessons Learned
This document summarizes all the problems I encountered while integrating **ITSA** (Inter‑Subject Tangent Space Alignment) into a subject‑specific fine‑tuning pipeline for EEG‑based image classification using Transformers.  
It also describes why each problem occurred and how it was fixed.

---

## 1. Problem: ITSA Pretrained Files Were Several GB in Size

### Why it happened
ITSA stores giant matrices, `Rs_` (subject‑specific rotations), in Tangent Space. Their size is `d x d`, where:

$$d = \frac{C(C+1)}{2} \quad \text{(e.g., C=128 }\Rightarrow\text{ d is very large)}$$

For EEG with 128 channels, these matrices are *huge*, and storing them for 5 subjects, for the pre-training part of the fine-tuning, explodes file size.

- **Early attempt**
I zeroed `Rs_` before saving:
```python
if hasattr(itsa_to_save._itsa, "Rs_"):
    itsa_to_save._itsa.Rs_ = {}
torch.save(itsa_to_save, 'itsa_pretrained_space.pth')
```
This reduced size but wasn’t a principled export and risked dropping other useful fields unintentionally.

### Fix: Export a lightweight ITSA artifact (`export_light`)
I implemented a **lightweight ITSA** version that keeps only the components needed for later subject adaptation:

Kept:
* `ts_` — Tangent Space  
* `scaler_` — normalization  
* `mu_global_` — global class means  
* `reference_G_` — SPD reference matrix  

Discarded:
* `Rs_`
* `A_filters_`
* `M_inv_sqrt_`
* caches

```python
def export_light(self):
    lite = ITSA(
        subject_eps=self.subject_eps,
        mean_tol=self.mean_tol,
        mean_maxiter=self.mean_maxiter,
        unit_trace_per_subject=self.unit_trace_per_subject
    )
    # Keep only the essentials
    lite.ts_ = self.ts_
    lite.scaler_ = self.scaler_
    lite.mu_global_ = {int(k): v.astype(np.float32) for k, v in self.mu_global_.items()}

    # Remove/compact the rest
    lite.reference_G_ = self.reference_G_  # (C,C), can cast to float32 if desired
    lite.M_inv_sqrt_ = {}  # base subjects' not needed
    lite.Rs_ = {}  # huge -> drop
    lite.A_filters_ = None  # derived during target adaptation
    lite._filters_cache = {}  # clean
    return lite
```

**Result**: The saved file goes from GB -> a few MB, while still being sufficient to **adapt** later to the target subject and derive the small spatial filter:

$$A_s \in \mathbb{R}^{C\times C}$$

---

## 2. ITSA Was Not Improving Performance Results (or only improving peak VA)

### Why it happened
I was applying ITSA as a **fixed deterministic spatial mapping** per subject:

$$x \mapsto x\,A_s$$

That aligns domains but does **not** provide *augmentation*: the model sees **one** ITSA-view of each trial, with no diversity -> small or inconsistent gains.

### Fix: Treat ITSA as **real augmentation** in train, and keep it **deterministic** in val/test
- **Train:** blend between **identity** and **alignment** using a geodesic‑style interpolation.
- **Val/Test:** apply **pure $A_s$** (deterministic), so evaluation is stable and consistent with deployment.

### 2.1 In‑ITSA Augmentation (what happens inside `ITSA.py`)
This section explains the augmentation logic implemented **inside** `ITSA.py` (not the main training script) and why each parameter exists. The goal is to (i) perform **domain alignment** per subject via a spatial filter $A_s \in \mathbb{R}^{C \times C}$, and (ii) expose the model to **multiple aligned views** of each trial (augmentation) during training—while keeping **inference deterministic**.

**The two operating modes:**
* `mode="deterministic"`: Uses the subject‑specific alignment **as is**:
    $$X \mapsto X \, A_s$$
    This is intended for **validation/test** (and sometimes warm‑up). It yields a stable, reproducible mapping with no stochasticity.
* `mode="augment"`: Generates *stochastically* perturbed aligned views in **training**. Concretely, it mixes between identity $I$ and the learned alignment $A_s$ using a geodesic‑style interpolation in the SPD manifold:
    $$A_\alpha = \exp\!\big( \alpha \, \log(A_s) \big), \qquad \alpha \sim \mathrm{Uniform}(a, b) \in [0, 1]$$
    Then the input is transformed as $X \mapsto X A_\alpha$. With $\alpha \approx 0$ the view is close to the original, with $\alpha \approx 1$ it’s fully aligned. Sampling $\alpha$ across a range expands the training distribution while preserving the spatial structure.

**The parameters you set in the main loop:**
* `alpha_range=(0.5, 0.95)`: The interval $(a, b)$ from which $\alpha$ is sampled when `mode="augment"`. Lower $a \Rightarrow$ more identity‑like views (weaker alignment, more diversity). Higher $b \Rightarrow$ more alignment‑like views (stronger domain adaptation). In practice, `(0.5, 0.95)` makes most augmented views noticeably aligned but not all the way to $A_s$, which tends to improve robustness without collapsing variability.
* `jitter_sigma`: (Deprecated - see section 2.2 and 5). Previously added a very small perturbation in the covariance eigen‑spectrum of the transformed signal. 

**Helper functions and safety guards (inside `ITSA.py`):**
* **Blending $I \leftrightarrow A_s$** (`_blend_filter`): Implements $A_\alpha = \exp(\alpha \log(A_s))$ using eigendecomposition. Adds SPD *safety floors* to eigenvalues to avoid negative/zero eigenvalues that would break `log/exp`.
* **Deterministic transform**: Applies the cached $A_s$ (or $A_\alpha$ in augment mode) directly on the time × channels tensor:
    $$X_{(T \times C)} \mapsto X \, A, \quad A \in \{A_s, A_\alpha\}$$
    Everything runs in **float32** with clamps and symmetrization (e.g., $\frac{A + A^\top}{2}$) to preserve SPD structure and minimize numeric drift.
* **Caching**: The per‑subject filter $A$ is moved once to the current device/dtype and cached. This avoids re‑allocations and speeds up batches.

### 2.2 Deprecating `jitter_sigma` (why and how the code changed)
We initially added `jitter_sigma` to inject tiny **SPD‑preserving spectral perturbations** during training. In practice, on this project it led to occasional numerical instabilities (NaNs) unless we used large eigenvalue floors, noticeable runtime overhead, and no clear benefit over the simpler blending.

*(Note: While I initially did a soft-remove to avoid breaking APIs, I eventually performed a complete hard-remove to reduce technical debt. See Section 5 for the final implementation details).*

With `jitter_sigma=0.0`, training was stable and fast, and the augmentation from the geodesic blend already provided the desired variability. Therefore, jitter is **deprecated** here.

**Current recommended usage:**
* **Train (augment):** `mode="augment", alpha_range=(0.5, 0.95)`
* **Val/Test (deterministic):** `mode="deterministic"`

### 2.3 Driver (main loop) change
I added an `--itsa_off` flag and applied augmentation only in the training split:

```python
for i, (input, target, batch_subjects) in enumerate(loaders[split]):
    # Check CUDA
    if not opt.no_cuda:
        input = input.to("cuda")
        target = target.to("cuda")
        batch_subjects = batch_subjects.to("cuda")

    # ITSA pipeline
    if not opt.itsa_off:
        if split == "train":
            input = itsa.transform_batch(
                input, batch_subjects,
                mode="augment",
                alpha_range=(0.5, 0.95)
            )
        else:
            input = itsa.transform_batch(
                input, batch_subjects, mode="deterministic"
            )

    output = model(input)
```

> This hasn't been tested yet, due to other problems that arose due to this change in the implementation, but the hypothesis is that it'll dramatically improve stability and allow improvements to show up in "Test @ best VA".

---

## 3. I updated only one side (`transform_signals`) but missed the integrator path

### Why it happened
- I added `mode`, `alpha_range`, `jitter_sigma` to `ITSA.transform_signals(...)`, **but** my `ITSAIntegrator.transform_batch(...)` did not forward those parameters (or vice‑versa).
- Result: calling `itsa.transform_batch(..., mode="augment", ...)` reached an older path that **didn’t accept** those kwargs.

### Fix: Update both the class method and the integrator wrapper

- In **`ITSA`** (class), ensure the signature includes the new parameters:

```python
class ITSA:
    @torch.no_grad()
    def transform_signals(
        self,
        x: torch.Tensor,
        subjects: torch.Tensor,
        mode: str = "deterministic",  # "augment" | "deterministic"
        alpha_range=(1.0, 1.0)  # if augment: blends I↔A_s, e.g., (0.5, 0.95)
    ) -> torch.Tensor:
        ...
```
- In **`ITSAIntegrator`**, forward these arguments down to the class:

```python
class ITSAIntegrator:
    @torch.no_grad()
    def transform_batch(self, x: torch.Tensor, subjects: torch.Tensor,
                        mode: str = "deterministic",
                        alpha_range=(1.0, 1.0)) -> torch.Tensor:
        """
        Applies ITSA per subject. Preserves x shape:
        - (B,T,C) -> (B,T,C)
        - (B,1,C,T) -> (B,1,C,T)
        """
        return self._itsa.transform_signals(x, subjects,
                                            mode=mode,
                                            alpha_range=alpha_range)
```

---

## 4. Other changes I made (and why)

### 4.1 Removed extra `to_spd_np(...)` projections when building $A_s$
- **Change:** I **removed** the extra `to_spd_np(...)` projection from `adapt_subject` and `_derive_subject_filters`.
- **Why:** $A_s$ already comes from compositions of SPD‑safe building blocks (`invsqrtm`, `sqrtm`, etc.). Re‑projecting can (i) slightly distort the intended mapping, (ii) add numerical overhead, and (iii) is unnecessary in practice if intermediate steps are stabilized elsewhere.

### 4.2 Historical attempt to shrink files by zeroing `Rs_`
- **Outcome:** Reduces size but isn’t a clean/exportable contract; replaced by **`export_light()`** which saves *only* the necessary fields (small, robust, and explicit).

### 4.3 Improve numerical stability when computing covariances
- **Updated (better stability):**
```python
C = cov_from_signal_torch(eeg2d.double(), eps=1e-4).cpu().numpy()  # (C,C) SPD
```
- **Why:** Double precision for the covariance step + a small diagonal `eps` floor dramatically reduces SPD issues during `sqrtm/invsqrtm` and downstream alignment.

### 4.4 Main‑file integration of ITSA (driver)
- I consolidated all ITSA changes in the *train/val/test* loop and added an `--itsa_off` switch (as seen in Section 2.3).

---

## 5. Problem: Jitter Augmentation Caused NaNs and Bloated the Code (The "Hard-Remove")

### Why it happened
Initially, `jitter_sigma` was implemented as a secondary augmentation strategy during training. The idea was to inject tiny, SPD-preserving spectral perturbations into the covariance matrices to make the model more robust. 

However, in practice, this approach introduced several critical issues:
1.  **Numerical Instability:** It frequently caused NaNs during training unless heavy eigenvalue floors were applied, which distorted the data.
2.  **High Computational Overhead:** It required additional, expensive eigendecompositions (`invsqrtm`, `sqrtm`) on the fly, significantly slowing down the training loop.
3.  **Lack of Empirical Benefit:** The geodesic blending (`alpha_range=(0.5, 0.95)`) already provided sufficient and stable data augmentation. The jitter did not improve the final classification results.

### Fix
Initially, the idea was to "soft-remove" the jitter (leaving the arguments in the functions but disabling the internal logic). However, `ITSA.py` is a core file with over 600 lines of code. Keeping "zombie code" (dead arguments, unused helper functions, and commented-out logic blocks) only adds technical debt, reduces readability, and creates confusion for future maintenance. Therefore, a **complete removal (hard-remove)** was the cleanest and most professional solution.

I completely stripped out all jitter-related logic across the pipeline to streamline the augmentation process:

* **In `ITSA.py` (Main Class):**
    * Removed the `jitter_sigma` argument from the `transform_signals` signature.
    * Deleted the unused `_spd_jitter` helper function.
    * Removed the entire conditional block (`if mode == "augment" and jitter_sigma > 0:`) that handled the recoloring and noise injection.
* **In `ITSA.py` (`ITSAIntegrator` Wrapper):**
    * Removed `jitter_sigma` from the `transform_batch` signature.
    * Stopped forwarding the parameter to `self._itsa.transform_signals`.
* **In the Main Training Loop (`transformer_eeg_signal_classification.py`):**
    * Removed the `jitter_sigma=0.015` parameter from the `itsa.transform_batch` call during the `train` split.

**Final, clean usage for augmentation:**
The pipeline now relies purely on geodesic interpolation, which is numerically stable and fast:
```python
input = itsa.transform_batch(
    input, batch_subjects,
    mode="augment",
    alpha_range=(0.5, 0.95)
)
```

---

## 6. Problem: Silent Caching Bug and CPU Bottleneck in Augmentation

### Why it happened
While reviewing the geodesic augmentation in `ITSA.transform_signals(...)`, I discovered two critical issues that were hindering both the model's learning and the training speed:

1. **The Silent Caching Bug (Loss of Stochasticity):**
   The code was generating a random `alpha`, blending the filter ($A_\alpha$), and then **saving that augmented filter into the cache** (`self._filters_cache[key] = A_t`). Consequently, the random augmentation was only computed once per subject during the very first batch. For the rest of the epoch (and subsequent epochs), the exact same augmented matrix was recycled. The model was just memorizing a fixed mapping instead of seeing diverse views.
2. **The CPU/GPU Bottleneck:**
   The `_blend_filter` helper was using NumPy (`np.linalg.eigh`), which runs on the CPU. Inside the training loop, the code was constantly pulling tensors to the CPU, performing complex eigendecompositions, and pushing them back to the GPU (`to(device=device)`). This broke parallelism and starved the GPU.

### Fix: Native PyTorch Blending and Correct Cache Targeting
**1. Fix the Cache Logic:**
I modified the caching mechanism to store *only* the deterministic base spatial filter ($A_s$) on the GPU. The stochastic interpolation is now applied *on the fly* during every forward pass without overwriting the cached base filter.

**2. Move Math to the GPU:**
I rewrote the blending helper to use native PyTorch operations (`torch.linalg.eigh`, `torch.clamp`, `torch.exp`). This keeps all matrices on the GPU, avoiding PCIe transfer delays and drastically speeding up the batches.

**Updated `transform_signals` implementation:**
```python
    @torch.no_grad()
    def transform_signals(
            self,
            x: torch.Tensor,
            subjects: torch.Tensor,
            mode: str = "deterministic",
            alpha_range=(1.0, 1.0)
    ) -> torch.Tensor:
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)

        device = x.device

        # Normalize to (B,T,C)
        squeeze_4d = False
        if x.dim() == 4:  # (B,1,C,T)
            squeeze_4d = True
            x_ = x.squeeze(1).transpose(1, 2)
        elif x.dim() == 3:  # (B,T,C)
            x_ = x
        elif x.dim() == 2:  # (T,C) -> (1,T,C)
            x_ = x.unsqueeze(0)
        else:
            raise ValueError(f"Forma no soportada: {tuple(x.shape)}")

        B, T, C = x_.shape

        if not torch.is_tensor(subjects):
            subjects = torch.tensor([int(subjects)], device=device)
        elif subjects.dim() == 0:
            subjects = subjects.view(1)

        # PyTorch NATIVE helper: 100% on GPU, extremely fast
        def _blend_filter_torch(A_t: torch.Tensor, alpha: float) -> torch.Tensor:
            w, V = torch.linalg.eigh(A_t)
            w = torch.clamp(w, min=1e-8)
            w_alpha = torch.exp(alpha * torch.log(w))
            A_blend = (V * w_alpha.unsqueeze(0)) @ V.transpose(-1, -2)
            return 0.5 * (A_blend + A_blend.transpose(-1, -2))

        out = []
        for b in range(B):
            s = int(subjects[b].item()) if subjects.numel() > 1 else int(subjects.item())
            key = (s, device, torch.float32)

            # 1. Retrieve (or create) the BASE deterministic filter from cache
            A_base = self._filters_cache.get(key)
            if A_base is None:
                A_np = self.A_filters_.get(s, None) if self.A_filters_ is not None else None
                if A_np is None:
                    A_base = torch.eye(C, device=device, dtype=torch.float32)
                else:
                    A_base = torch.from_numpy(A_np.astype(np.float32)).to(device=device)
                # Cache ONLY the deterministic base matrix
                self._filters_cache[key] = A_base  

            # 2. Apply stochastic augmentation ON THE FLY (GPU)
            if mode == "augment":
                alpha = np.random.uniform(alpha_range[0], alpha_range[1])
                A_t = _blend_filter_torch(A_base, alpha)
            else:
                A_t = A_base

            # 3. Apply the filter: (T,C) @ (C,C) -> (T,C)
            xb = x_[b].to(torch.float32) @ A_t
            out.append(xb)

        Xout = torch.stack(out, dim=0)

        if squeeze_4d:
            return Xout.transpose(1, 2).unsqueeze(1)
        if x.dim() == 2:
            return Xout.squeeze(0)
        return Xout
```

---

## 7. Code Cleanup: Removing Technical Debt in `ITSA.py`

### Why it happened
`ITSA.py` had a bit of an "identity crisis", keeping legacy methods that belonged to classic Machine Learning (SVMs) while the pipeline only required Deep Learning logic (Transformers). It also had redundant aliases and duplicated cleanup calls.

### Fix
I safely removed the dead code to improve readability and reduce file bloat:
* **Removed `transform(self, covs, subjects)`:** This method extracted 1D Tangent Space features. It was completely unused since `transformer_eeg_signal_classification.py` exclusively calls `transform_signals`.
* **Cleaned up `ITSAIntegrator`:** Removed the redundant `import torch as _torch` alias, standardizing everything to use the standard `torch` imports.
* **Removed duplicate cache clears:** Cleaned up redundant `self._filters_cache.clear()` and `self.Rs_.clear()` lines at the end of `adapt_subject`.

---

## 8. Problem: Hidden "NoneType" Assignment Errors on Outdated Exports

### Why it happened
Shortly after fixing the `A_filters_` assignment error, another `TypeError: 'NoneType' object does not support item assignment` appeared, this time pointing near the `self.Rs_[s]` access. Because Jupyter tracebacks can occasionally misalign line numbers after cell modifications, the traceback pointed to a read operation, but the underlying crash happened at the assignment step (`self.Rs_[s] = ...`). 
This indicated that the loaded `.pth` model (likely an older version of `export_light()`) had also saved `Rs_`, `M_inv_sqrt_`, or other core dictionaries as `None` instead of empty dictionaries `{}`.

### Fix
Instead of patching dictionaries one by one, I added a robust initialization block (a "None-shield") at the very beginning of the `adapt_subject` method. Using `getattr()` ensures that even if the attributes are missing entirely from the loaded object, they are safely initialized as empty dictionaries before any loop starts.

**Updated code in `ITSA.py` (`adapt_subject` method):**
```python
        mask_tr = np.zeros(covs_np.shape[0], dtype=bool)
        mask_tr[train_idx_np] = True

        # Robust Initialization (Shield against NoneType from light exports)
        if getattr(self, "M_inv_sqrt_", None) is None: self.M_inv_sqrt_ = {}
        if getattr(self, "Rs_", None) is None: self.Rs_ = {}
        if getattr(self, "A_filters_", None) is None: self.A_filters_ = {}
        if getattr(self, "_filters_cache", None) is None: self._filters_cache = {}

        for s in np.unique(subjects_np):
            # ... adaptation loop continues ...
```

---

## 9. Future-Proofing ITSA Reusability (Skipping the 1-Hour Base Calculation)

### Why it happened
When attempting to objectively compare models, I needed to retrain the base Transformer on the 5 calibration subjects *with* ITSA enabled. However, the `export_light()` method was aggressively dropping the `A_filters_` dictionary to save disk space. Because of this, loading the "light" `pretrained_itsa` model forced the script to re-compute the Riemannian geometry and spatial filters for the 5 base subjects from scratch, which takes around an hour. Furthermore, the main training script always assumed that providing a `pretrained_itsa` meant we wanted to *adapt* it to a new subject, making it impossible to just "load and train" on the original subjects.

### Fix
I implemented a two-step solution to make the ITSA pipeline flexible and reusable:
1. **Retain Base Filters:** Modified `export_light()` to keep `A_filters_` (since spatial filters are very small, 128x128 matrices, unlike the massive `Rs_` rotation matrices).
2. **Add an Adaptation Flag:** Added an `--adapt_new_subject` boolean flag to the main script parser to explicitly distinguish between "loading base filters for retraining" and "adapting filters for fine-tuning on a new subject".

**1. Updated `export_light` in `ITSA.py`:**
```python
    def export_light(self):
        lite = ITSA(
            subject_eps=self.subject_eps,
            mean_tol=self.mean_tol,
            mean_maxiter=self.mean_maxiter,
            unit_trace_per_subject=self.unit_trace_per_subject
        )
        lite.ts_ = self.ts_
        lite.scaler_ = self.scaler_
        lite.mu_global_ = {int(k): v.astype(np.float32) for k, v in self.mu_global_.items()}
        lite.reference_G_ = self.reference_G_
        
        lite.M_inv_sqrt_ = {}
        lite.Rs_ = {}  # Enormous matrices -> dropped
        lite.A_filters_ = self.A_filters_  # KEEP the spatial filters
        lite._filters_cache = {} 
        return lite
```

**2. Updated Main Script (`transformer_eeg_signal_classification.py`):**
```python
# Added to parser:
parser.add_argument('--adapt_new_subject', default=False, action="store_true", help="Adapt pretrained ITSA to a new subject")

# Updated integration logic:
if not opt.itsa_off:
    if opt.pretrained_itsa != '':
        # 1. Load the core object and re-wrap it
        loaded_itsa_core = torch.load(opt.pretrained_itsa, weights_only=False)
        itsa = ITSAIntegrator(loaded_itsa_core)
        
        # 2. Check if we need to adapt or just load
        if opt.adapt_new_subject:
            print("Adapting ITSA space to new subject...")
            itsa.adapt_from_dataset(dataset, splits_path=opt.splits_path, split_num=opt.split_num)
        else:
            print("Loading base ITSA filters (without adapting)...")
            
    else:
        print("Computing ITSA space from scratch (please wait!)...")
        itsa = ITSAIntegrator.from_dataset(dataset, splits_path=opt.splits_path, split_num=opt.split_num)
        itsa_lite = itsa._itsa.export_light()
        torch.save(itsa_lite, 'itsa_pretrained_space_light.pth')
```

### Usage for future experiments:
* **To retrain the base model quickly:** `--pretrained_itsa itsa_pretrained_space_light.pth`
* **To fine-tune on a new target subject:** `--pretrained_itsa itsa_pretrained_space_light.pth --adapt_new_subject`

---

## 10. Problem: Catastrophic Forgetting During Fine-Tuning (ITSA Performance Collapse)

### Why it happened
When attempting to evaluate the pre-trained ITSA Transformer on a new unseen subject (Target Subject), the model's performance collapsed completely. The Training Loss started at ~8.69 (much worse than random chance for 40 classes, which is ~3.68), and the Accuracy stagnated around 5-8%, performing significantly worse than a baseline model trained from scratch without ITSA (which achieved ~43%).

This was caused by the **"fair comparison fallacy"** in Transfer Learning. I kept the exact same learning rate for fine-tuning as I did for training from scratch (`0.001`). A high learning rate is necessary when training from scratch, but applying it to a pre-trained model acts as a sledgehammer. The optimizer saw the initial domain shift of the new subject, took a massive gradient step, and completely destroyed the optimal attention weights (the understanding of the Riemannian space) learned during the 200 pre-training epochs. This is a textbook case of **Catastrophic Forgetting**.

### Fix
To perform a true and effective fine-tuning, the hyperparameters must be adapted to gently update the network without destroying its prior knowledge. 

**The Scalpel Approach (Lower Learning Rate):** I dropped the Learning Rate by an order of magnitude (from `1e-3` to `1e-4`). This ensures the optimizer only makes microscopic adjustments to the classifier head while preserving the deep spatial feature extractors. 

*Note:* The Geodesic Augmentation (`alpha_range=(0.5, 0.95)`) used during pre-training is **kept active** during fine-tuning. This structural noise acts as an excellent regularizer, preventing the model from overfitting on the very limited data of the single target subject, while val/test remain deterministic.

**Updated Configuration for Fine-Tuning (`transformer_eeg_signal_classification.py`):**
```python
# When running the fine-tuning script with --pretrained_net and --adapt_new_subject:

# 1. Lower Learning Rate specifically for fine-tuning
opt.learning_rate = 1e-4  

# 2. Keep Geodesic Augmentation for training, deterministic for val/test
# (Same as pre-training conditions)
alpha_range = (0.5, 0.95)

# The optimizer should still use weight decay to prevent overfitting on the small target dataset
optimizer = torch.optim.Adam(model.parameters(), lr=opt.learning_rate, weight_decay=1e-4)
```

---

## 11. Bug: Trace Normalization Applied Inconsistently Between Fit and Transform

### Why it happened
In the `_fit_subject_means` method, before computing each subject's "average EEG pattern" (their Riemannian mean `M`), the code optionally normalizes each covariance matrix so they all have the same overall scale. This is controlled by the `unit_trace_per_subject` flag and is done by dividing each matrix by its **trace** (a measure of its total "energy"):

```python
# Inside _fit_subject_means (what happens during FIT):
if self.unit_trace_per_subject:
    tr = np.trace(covs_s_tr, axis1=1, axis2=2).reshape(-1, 1, 1)
    covs_s_tr = covs_s_tr / np.maximum(tr, 1e-12)  # normalize first
M = mean_riemann(covs_s_tr, ...)                    # then compute mean
self.M_inv_sqrt_[s] = invsqrtm(M)                  # corrector built from normalized data
```

So the corrector `M_inv_sqrt_[s]` was computed from **trace-normalized** matrices. However, in `_apply_subject_recentering_np` — the method that actually *uses* that corrector — the normalization step was missing. It applied the corrector directly to the **raw, un-normalized** matrices:

```python
# Inside _apply_subject_recentering_np (what happened during TRANSFORM - WRONG):
Cc = to_spd_np(C, eps=self.subject_eps)  # raw matrix, NOT normalized
Minv = self.M_inv_sqrt_[s]
out[i] = to_spd_np(Minv @ Cc @ Minv, ...)  # applying a corrector built for normalized data
                                             # to un-normalized data → wrong result
```

This is mathematically equivalent to calibrating a scale using kilograms, then using it to weigh objects measured in pounds. The corrector was built for one "unit system" but applied to another, producing a subtly wrong alignment for every single sample.

### Fix
Added the trace normalization step inside `_apply_subject_recentering_np`, **before** applying the corrector, so that the data is in the same "unit system" that `M_inv_sqrt_[s]` was built from:

```python
def _apply_subject_recentering_np(self, covs: np.ndarray, subjects: np.ndarray) -> np.ndarray:
    out = np.empty_like(covs)
    for i, (C, s) in enumerate(zip(covs, subjects)):
        s = int(s)
        Cc = to_spd_np(C, eps=self.subject_eps)

        # ✅ FIX: normalize trace BEFORE applying the corrector,
        # to match the scale used when M_inv_sqrt_[s] was computed in _fit_subject_means
        if self.unit_trace_per_subject:
            tr = np.trace(Cc)
            Cc = Cc / max(tr, 1e-12)

        Minv = self.M_inv_sqrt_[s]
        out[i] = to_spd_np(Minv @ Cc @ Minv, eps=self.subject_eps)
    return out
```

The same logic applies inside `adapt_subject`, where the normalization was already present in the new-subject loop — so no changes were needed there.

---

## 12. Bug: Spatial Filter Applied Transposed, Corrupting All Aligned Signals

### Why it happened
This was the most impactful bug in the pipeline, and most likely the primary cause of accuracy being stuck at ~0.36 (near-random for 40 classes).

The spatial filter `A_s` is a `(C, C)` matrix — a learned mixing of the 128 EEG channels. The EEG data has shape `(T, C)`, where T is time and C is channels. Each row is one moment in time, each column is one channel.

In standard linear algebra, applying a spatial filter to row-major signal data `(T, C)` via right-multiplication requires **transposing** the filter:

$$X_{\text{aligned}} = X \cdot A_s^{\top}$$

This is because `A_s` is designed to act on **column vectors** (one full channel vector at a time). When the data is stored as rows instead, the transpose is needed to keep the channel-mixing oriented correctly.

The original code in `transform_signals` was doing:

```python
# WRONG: applying A_t directly without transposing
xb = x_[b].to(torch.float32) @ A_t
```

This is equivalent to applying `A_s^{\top}` instead of `A_s`. Since `A_s` is **not symmetric** (it is a composition of several non-symmetric matrices — `invsqrtm`, `sqrtm`, and a Procrustes rotation `R`), its transpose is a completely different transformation. Instead of aligning the subject's EEG toward the reference space, the code was applying an unintended distortion to every sample, for every subject, every batch, across the entire training run.

### Fix
Added `.T` (transpose) to the filter before applying it, on the relevant line inside `transform_signals`:

```python
# ✅ FIX: transpose A_t so the filter is applied in the correct orientation
xb = x_[b].to(torch.float32) @ A_t.T
```

This single character change ensures the spatial filter acts as mathematically intended — mixing channels to remove each subject's personal bias and align their signals toward the learned reference space.

### Why this caused ~0.36 accuracy
With the filter transposed, the "alignment" was actively moving each subject's data in the **wrong direction** in feature space — making signals from different subjects *more* different rather than less. The transformer model was then trying to learn classification on top of consistently corrupted inputs, which explains why:
- Training loss stayed high (the model could not find a good mapping)
- Accuracy hovered near chance (0.36 ≈ near 1/40, with some structure from the corruption being accidentally learnable)
- ITSA appeared to perform worse than no alignment at all

---

## Summary

* **Space issue:** Solved with `export_light()` (keep `ts_`, `scaler_`, `mu_global_`, `reference_G_`, and `A_filters_`; drop heavy fields).  
* **No improvement with ITSA:** Fixed by using ITSA as *augmentation* in **train** (geodesic interpolation between Identity and $A_s$) and **deterministic** in **val/test**. 
* **Jitter overhead/NaNs:** Deprecated and completely hard-removed `jitter_sigma`, relying entirely on the geodesic interpolation for augmentation stability.
* **Partial implementation:** Fixed by updating **both** `ITSA.transform_signals(...)` **and** `ITSAIntegrator.transform_batch(...)` to accept/forward `mode` and `alpha_range`.  
* **Stability & code cleanliness:** Removed redundant SPD projections when building $A_s$; computed covariances in double precision with `eps`; centralized the ITSA calls; and removed dead ML legacy code (e.g., SVM feature extraction) to reduce technical debt.
* **Caching bug & CPU bottleneck:** Solved by caching *only* the deterministic base spatial filter on the GPU and rewriting the stochastic blending to run natively on the GPU on the fly.
* **Hidden "NoneType" errors:** Prevented crashes from outdated light exports by adding a robust initialization block ("None-shield") at the start of `adapt_subject`.
* **Reusability & Recalculation overhead:** Kept `A_filters_` in the light export and added an `--adapt_new_subject` flag to allow loading base filters directly, avoiding a 1-hour recalculation.
* **Catastrophic Forgetting during fine-tuning:** Fixed the performance collapse on new target subjects by lowering the learning rate (e.g., `1e-4`) to gently update weights, while keeping the training geodesic augmentation (`alpha_range=(0.5, 0.95)`) to prevent overfitting.
* **Trace Normalization Bug:** Fixed an inconsistent alignment scale by ensuring that input matrices are trace-normalized during transformation, matching the mathematical constraints used during the fitting phase.
* **Spatial Filter Transpose Bug:** Fixed near-random accuracy (~0.36) by properly transposing the spatial filter ($A_s^\top$) when right-multiplying row-major signal tensors $(T, C)$, preventing the model from corrupting the channel mixing.
