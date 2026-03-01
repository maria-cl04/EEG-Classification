# ITSA Integration — Troubleshooting & Lessons Learned
This document summarizes all the problems I encountered while integrating **ITSA** (Inter‑Subject Tangent Space Alignment) into a subject‑specific fine‑tuning pipeline for EEG‑based image classification using Transformers.  
It also describes why each problem occurred and how it was fixed.

---

## 1. Problem: ITSA Pretrained Files Were Several GB in Size

### Why it happened
ITSA stores giant matrices, `Rs_` (subject‑specific rotations), in Tangent Space. Their size is `d x d`, where:

$$d = \frac{C(C+1)}{2} \quad \text{(e.g., C=128 \Rightarrow d is very large)}$$

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
    # Conservamos lo imprescindible
    lite.ts_ = self.ts_
    lite.scaler_ = self.scaler_
    lite.mu_global_ = {int(k): v.astype(np.float32) for k, v in self.mu_global_.items()}

    # Eliminamos/compactamos el resto
    lite.reference_G_ = self.reference_G_  # (C,C), puedes castear a float32 si quieres
    lite.M_inv_sqrt_ = {}  # no necesitamos las de sujetos base
    lite.Rs_ = {}  # enorme → fuera
    lite.A_filters_ = None  # se derivan en la adaptación al destino
    lite._filters_cache = {}  # limpio
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
* `jitter_sigma`: (Deprecated - see section 2.2). Previously added a very small perturbation in the covariance eigen‑spectrum of the transformed signal. 

**Helper functions and safety guards (inside `ITSA.py`):**
* **Blending $I \leftrightarrow A_s$** (`_blend_filter`): Implements $A_\alpha = \exp(\alpha \log(A_s))$ using eigendecomposition. Adds SPD *safety floors* to eigenvalues to avoid negative/zero eigenvalues that would break `log/exp`.
* **Deterministic transform**: Applies the cached $A_s$ (or $A_\alpha$ in augment mode) directly on the time × channels tensor:
    $$X_{(T \times C)} \mapsto X \, A, \quad A \in \{A_s, A_\alpha\}$$
    Everything runs in **float32** with clamps and symmetrization (e.g., $\frac{A + A^\top}{2}$) to preserve SPD structure and minimize numeric drift.
* **Caching**: The per‑subject filter $A$ is moved once to the current device/dtype and cached. This avoids re‑allocations and speeds up batches.

### 2.2 Deprecating `jitter_sigma` (why and how the code changed)
We initially added `jitter_sigma` to inject tiny **SPD‑preserving spectral perturbations** during training:
$$C \mapsto C_j \quad \text{with} \quad \lambda_j = \lambda \cdot (1 + \varepsilon), \;\varepsilon \sim \mathcal{N}(0,\sigma^2)$$
and then $X \mapsto X \, C^{-1/2} C_j^{1/2}$. In practice, on this project it led to occasional numerical instabilities (NaNs) unless we used large eigenvalue floors, noticeable runtime overhead, and no clear benefit over the simpler blending.

With `jitter_sigma=0.0`, training was stable and fast, and the augmentation from the geodesic blend already provided the desired variability. Therefore, jitter is **deprecated** here.

**What changed in `ITSA.py` (Two-step deprecation):**
1.  **Soft‑remove (non‑breaking):** Keep the argument in function signatures so training scripts don’t need edits. Remove the internal jitter block and add a comment: `# jitter disabled (no-op)`.
2.  **Hard‑remove (optional, later):** Drop `jitter_sigma` from the signatures and delete all jitter‑related helpers yielding a cleaner API.

**Current recommended usage:**
* **Train (augment):** `mode="augment", alpha_range=(0.5, 0.95)`
* **Val/Test (deterministic):** `mode="deterministic"`
* `jitter_sigma` is **unused** in this repository.

### 2.3 Driver (main loop) change
I added an `--itsa_off` flag and applied augmentation only in the training split (passing the deprecated jitter for API compatibility only):

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
                alpha_range=(0.5, 0.95),
                jitter_sigma=0.015  # NOTE: Now a no-op internally
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
        alpha_range=(1.0, 1.0),  # si augment: mezcla I↔A_s, p.ej. (0.5, 0.95)
        jitter_sigma: float = 0.0  # (Deprecated) jitter SPD opcional
    ) -> torch.Tensor:
        ...
```
- In **`ITSAIntegrator`**, forward these arguments down to the class:

```python
class ITSAIntegrator:
    @_torch.no_grad()
    def transform_batch(self, x: _torch.Tensor, subjects: _torch.Tensor,
                        mode: str = "deterministic",
                        alpha_range=(1.0, 1.0),
                        jitter_sigma: float = 0.0) -> _torch.Tensor:
        """
        Aplica ITSA por sujeto. Conserva la forma de x:
        - (B,T,C) -> (B,T,C)
        - (B,1,C,T) -> (B,1,C,T)
        """
        return self._itsa.transform_signals(x, subjects,
                                            mode=mode,
                                            alpha_range=alpha_range,
                                            jitter_sigma=jitter_sigma)
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

## Summary

* **Space issue:** Solved with `export_light()` (keep `ts_`, `scaler_`, `mu_global_`, `reference_G_`; drop heavy fields).  
* **No improvement with ITSA:** Fixed by using ITSA as *augmentation* in **train** (geodesic interpolation between Identity and $A_s$) and **deterministic** in **val/test**. 
* **Jitter overhead/NaNs:** Deprecated `jitter_sigma`, relying entirely on the geodesic interpolation for augmentation stability.
* **Partial implementation:** Fixed by updating **both** `ITSA.transform_signals(...)` **and** `ITSAIntegrator.transform_batch(...)` to accept/forward `mode`, `alpha_range`, `jitter_sigma`.  
* **Stability & cleanliness:** Removed redundant SPD projections when building $A_s$; computed covariances in double precision with `eps`; centralized the ITSA calls in the main loop.
