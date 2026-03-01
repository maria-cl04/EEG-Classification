# ITSA Integration — Troubleshooting & Lessons Learned
This document summarizes all the problems I encountered while integrating **ITSA**  (Inter‑Subject Tangent Space Alignment) into a subject‑specific fine‑tuning pipeline for EEG‑based image classification using Transformers.  
It also describes why each problem occurred and how it was fixed.

---

## 1. Problem: ITSA Pretrained Files Were Several GB in Size

### **Why it happened**
ITSA stores giant matrices, `Rs_` (subject‑specific rotations), in Tangent Space. Their size is `d x d`, where

$$
 d = \frac{C(C+1)}{2} \quad \text{(e.g., C=128 ⇒ d is very large)}
$$

For EEG with 128 channels, these matrices are *huge*, and storing them for 5 subjects, for the pre-training part of the fine-tuning, explodes file size.

- **Early attempt**
I zeroed `Rs_` before saving:
  ```python
  if hasattr(itsa_to_save._itsa, "Rs_"):
      itsa_to_save._itsa.Rs_ = {}
  torch.save(itsa_to_save, 'itsa_pretrained_space.pth')
  ```
This reduced size but wasn’t a principled export and risked dropping other useful fields unintentionally.

### Fix: **Export a lightweight ITSA artifact (`export_light`)**
I implemented a **lightweight ITSA** version that keeps only the components needed for later subject adaptation:

Kept:
- `ts_` — Tangent Space  
- `scaler_` — normalization  
- `mu_global_` — global class means  
- `reference_G_` — SPD reference matrix  

Discarded:
- `Rs_`
- `A_filters_`
- `M_inv_sqrt_`
- caches

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

**Result**: The saved file goes from GB -> a few MB, while still being sufficient to **adapt** later to the target subject and derive the small spatial filter.

$$
A_s \in \mathbb{R}^{C\times C}
$$


---

## 2. ITSA Was Not Improving Performance Results(or only improving peak VA)

### Why it happened
I was applying ITSA as a **fixed deterministic spatila mapping** per subject:

$$
x \mapsto x\,A_s
$$

That aligns domains but does **not** provide *augmentation*: the model sees **one** ITSA-view of each trial, with no diversity -> small or inconsisten gains.

### Fix: Treat ITSA as **real augmentation** in train, and keep it **deterministic** in val/test

- **Train:** blend between **identity** and **alignment** using a geodesic‑style interpolation,
- 
$$
A_\alpha = \exp\big(\alpha \,\log(A_s)\big),\quad \alpha \sim \text{Uniform}(a,b)
$$

  plus an optional **SPD jitter** (very small) for robustness.
- **Val/Test:** apply **pure \(A_s\)** (deterministic), so evaluation is stable and consistent with deployment.

### Driver (main loop) change
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
                alpha_range=(0.5, 0.95),
                jitter_sigma=0.015
            )
        else:
            input = itsa.transform_batch(
                input, batch_subjects, mode="deterministic"
            )

    output = model(input)
```

> This hasn't been tested yet, due to other problems that arose due to this change in the implementation, but the hypothesis is that it'll dramatically imporve stability and allow improvements to show up in "Test @ best VA".

## 3. I updated only one side (`transform_signals`) but missed the integrator path

### Why it happened
- I added `mode`, `alpha_range`, `jitter_sigma` to `ITSA.transform_signals(...)`,
  **but** my `ITSAIntegrator.transform_batch(...)` did not forward those parameters (or vice‑versa).
- Result: calling `itsa.transform_batch(..., mode="augment", ...)` reached an older path that **didn’t accept** those kwargs.

### Fix: **Update both** the class method and the integrator wrapper
- In **`ITSA`** (class), ensure the signature includes the new parameters (and accept `**kwargs` for forward compatibility):

```python
class ITSA:
    @torch.no_grad()
    def transform_signals(
        self,
        x: torch.Tensor,
        subjects: torch.Tensor,
        mode: str = "deterministic",  # "augment" | "deterministic"
        alpha_range=(1.0, 1.0),  # si augment: mezcla I↔A_s, p.ej. (0.5, 0.95)
        jitter_sigma: float = 0.0  # jitter SPD opcional, p.ej. 0.01–0.02
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
## 4) Other changes I made (and why)

### 4.1) Removed extra `to_spd_np(...)` projections when building \(A_s\)
- **Original lines** (inside `adapt_subject` and `_derive_subject_filters`):
  ```python
  self.A_filters_[s] = to_spd_np(self.M_inv_sqrt_[s] @ W_s)
  # and
  self.A_filters_[s] = to_spd_np(A_s)
  ```
- **Change:** I **removed** the extra `to_spd_np(...)` projection.
- **Why:** \(A_s\) already comes from compositions of SPD‑safe building blocks (`invsqrtm`, `sqrtm`, etc.). Re‑projecting can (i) slightly distort the intended mapping, (ii) add numerical overhead, and (iii) is unnecessary in practice if intermediate steps are stabilized elsewhere (see 4.3).

### 4.2) Historical attempt to shrink files by zeroing `Rs_`
- As noted in §1, I initially did:
  ```python
  if hasattr(itsa_to_save._itsa, "Rs_"):
      itsa_to_save._itsa.Rs_ = {}
  torch.save(itsa_to_save, 'itsa_pretrained_space.pth')
  ```
- **Outcome:** reduces size but isn’t a clean/exportable contract; replaced by **`export_light()`** which saves *only* the necessary fields (small, robust, and explicit).

### 4.3) Improve numerical stability when computing covariances
- **Original:**
  ```python
  C = cov_from_signal_torch(eeg2d).cpu().numpy()  # (C,C) SPD
  ```
- **Updated (better stability):**
  ```python
  C = cov_from_signal_torch(eeg2d.double(), eps=1e-4).cpu().numpy()  # (C,C) SPD
  ```
- **Why:** double precision for the covariance step + a small diagonal `eps` floor dramatically reduces SPD issues during `sqrtm/invsqrtm` and downstream alignment.

### 4.4) Main‑file integration of ITSA (driver)
- I consolidated all ITSA changes in the *train/val/test* loop and added an `--itsa_off` switch:

```python
for i, (input, target, batch_subjects) in enumerate(loaders[split]):
    if not opt.no_cuda:
        input = input.to("cuda")
        target = target.to("cuda")
        batch_subjects = batch_subjects.to("cuda")

    if not opt.itsa_off:
        if split == "train":
            input = itsa.transform_batch(
                input, batch_subjects,
                mode="augment",
                alpha_range=(0.5, 0.95),
                jitter_sigma=0.015
            )
        else:
            input = itsa.transform_batch(
                input, batch_subjects,
                mode="deterministic"
            )

    output = model(input)
```

---

## Summary

- **Space issue:** solved with `export_light()` (keep `ts_`, `scaler_`, `mu_global_`, `reference_G_`; drop heavy fields).  
- **No improvement with ITSA:** fixed by using ITSA as *augmentation* in **train** (I↔Aₛ + jitter) and **deterministic** in **val/test**.  
- **Partial implementation:** fixed by updating **both** `ITSA.transform_signals(...)` **and** `ITSAIntegrator.transform_batch(...)` to accept/forward `mode`, `alpha_range`, `jitter_sigma`.  
- **Stability & cleanliness:** removed redundant SPD projections when building \(A_s\); computed covariances in double precision with `eps`; centralized the ITSA calls in the main loop with an `--itsa_off` switch.


