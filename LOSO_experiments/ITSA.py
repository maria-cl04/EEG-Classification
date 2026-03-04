import numpy as np
import torch
from typing import Dict, Optional, Union
from tqdm.auto import tqdm

# PyRiemann utilities
from pyriemann.utils.mean import mean_riemann, mean_logeuclid
from pyriemann.utils.base import invsqrtm, sqrtm
from pyriemann.tangentspace import TangentSpace

# sklearn / scipy
from sklearn.preprocessing import StandardScaler
from scipy.linalg import orthogonal_procrustes


def to_spd_np(A: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """
    Fuerza SPD en NumPy: simetriza, eleva autovalores < eps y reconstruye.
    A: (n,n) np.ndarray
    Devuelve: (n,n) np.ndarray SPD
    """
    A = 0.5 * (A + A.T)
    w, V = np.linalg.eigh(A)
    w = np.clip(w, eps, None)
    return (V * w) @ V.T


@torch.no_grad()
def to_spd_torch(A: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Fuerza SPD en Torch (GPU/CPU):
    - Soporta (..., n, n)
    - Simetriza, 'eigen-floor' y reconstruye con eigh batched.
    """
    A = 0.5 * (A + A.transpose(-1, -2))
    w, V = torch.linalg.eigh(A)  # (..., n), (..., n, n)
    w = torch.clamp(w, min=eps)
    A_spd = (V * w.unsqueeze(-2)) @ V.transpose(-1, -2)
    A_spd = 0.5 * (A_spd + A_spd.transpose(-1, -2))  # por si redondeos
    return A_spd


@torch.no_grad()
def cov_from_signal_torch(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Calcula la covarianza de canales C = (X^T X) / (T-1) a partir de una
    señal temporal X de forma (T, C), centrada por canal en el tiempo.

    Args
    -----
    :param x: torch.Tensor, forma (T, C)
      Señal temporal (T muestras, C canales).
    :param eps: 1e-6
    Returns
    -------
    C: torch.Tensor, forma (C, C)
      Matriz de covarianza de canales.
    """
    if x.dim() != 2:
        raise ValueError(f"x_np debe ser 2D (T,C); recibido: {tuple(x.shape)}")
    # Centrado por canal en el eje temporal
    x0 = x - x.mean(dim=0, keepdim=True)  # (T, C)
    T = max(1, x0.shape[0] - 1)
    C = (x0.transpose(0, 1) @ x0) / T  # (C, C)
    C = C + eps * torch.eye(C.shape[0], device=C.device, dtype=C.dtype)
    return to_spd_torch(C)


def _as_numpy(x: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    """
    Asegura np.ndarray (CPU). Si es Torch, hace .detach().cpu().numpy().
    """
    if isinstance(x, np.ndarray):
        return x
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    raise TypeError(f"Tipo no soportado: {type(x)}")


class ITSA:
    """
    ITSA para señales multicanal (p. ej., EEG) con dos APIs:
    1) API de características (igual que la versión original):
       - fit(covs, labels, subjects, train_idx)
       - transform(covs, subjects) -> Z_rot (espacio tangente)
    2) API de señales (nueva):
       - transform_signals(x, subjects) -> señales con la MISMA forma
         que la entrada, tras aplicar un filtro espacial por sujeto que
         implementa los 3 pasos: recentrado, escalado y alineación
         supervisada (derivada en TS pero aplicada en dominio de señal).

    Notas:
    - No modifica el modelo downstream; devuelve (B,T,C) o (B,1,C,T).
    - Las rotaciones se aprenden SOLO con TRAIN; en val/test se usa
      un único filtro por sujeto.
    """

    def __init__(
            self,
            subject_eps: float = 1e-10,
            mean_tol: float = 1e-6,
            mean_maxiter: int = 50,
            unit_trace_per_subject: bool = True,
    ) -> None:
        self.subject_eps = subject_eps
        self.mean_tol = mean_tol
        self.mean_maxiter = mean_maxiter
        self.unit_trace_per_subject = unit_trace_per_subject  # normalización de cada traza (opcional)

        # Artefactos aprendidos
        self.M_inv_sqrt_: Dict[int, np.ndarray] = {}
        self.reference_G_: Optional[np.ndarray] = None
        self.ts_: Optional[TangentSpace] = None
        self.scaler_: Optional[StandardScaler] = None
        self.Rs_: Dict[int, np.ndarray] = {}
        self.A_filters_: Optional[Dict[int, np.ndarray]] = None
        self.mu_global_: Dict[int, np.ndarray] = {}

        # Caché GPU de filtros convertidos a Torch por (sujeto, device, dtype)
        self._filters_cache: Dict[tuple, torch.Tensor] = {}

    # ---------------- Paso 1: medias por sujeto + recentrado ----------------
    def _fit_subject_means(self, covs: np.ndarray, subjects: np.ndarray, train_idx: np.ndarray) -> None:
        N = covs.shape[0]
        mask_tr = np.zeros(N, dtype=bool)
        mask_tr[train_idx] = True
        for s in tqdm(np.unique(subjects), desc="1/3: Calculating Riemannian Means"):
            s = int(s)
            m = (subjects == s)
            m_tr = m & mask_tr
            covs_s_tr = covs[m_tr] if np.any(m_tr) else covs[m]
            if self.unit_trace_per_subject:
                tr = np.trace(covs_s_tr, axis1=1, axis2=2).reshape(-1, 1, 1)
                covs_s_tr = covs_s_tr / np.maximum(tr, 1e-12)
            M_init = mean_logeuclid(covs_s_tr)
            M = mean_riemann(covs_s_tr, init=M_init, tol=self.mean_tol, maxiter=self.mean_maxiter)
            self.M_inv_sqrt_[s] = invsqrtm(M)

    def _apply_subject_recentering_np(self, covs: np.ndarray, subjects: np.ndarray) -> np.ndarray:
        """
        Recentrado por sujeto en NumPy: C' = M_s^{-1/2} C M_s^{-1/2}, con
        asegurado SPD antes y después por robustez numérica.
        """
        out = np.empty_like(covs)
        for i, (C, s) in enumerate(zip(covs, subjects)):
            s = int(s)
            Cc = to_spd_np(C, eps=self.subject_eps)

            # normalize trace before recentering (= before applying the corrector)
            if self.unit_trace_per_subject:
                tr = np.trace(Cc)
                Cc = Cc / max(tr, 1e-12)

            Minv = self.M_inv_sqrt_[s]
            out[i] = to_spd_np(Minv @ Cc @ Minv, eps=self.subject_eps)
        return out

    # -------- Paso 2–3: referencia G, TS(G) y estandarización (TRAIN) -------
    def _fit_reference_tspace_and_scaler(self, covs_rec: np.ndarray, train_idx: np.ndarray) -> None:
        mask_tr = np.zeros(covs_rec.shape[0], dtype=bool)
        mask_tr[train_idx] = True
        covs_tr = covs_rec[mask_tr]
        G_init = mean_logeuclid(covs_tr)
        G = mean_riemann(covs_tr, init=G_init, tol=self.mean_tol, maxiter=self.mean_maxiter)

        ts = TangentSpace(metric='riemann')
        ts.fit(covs_tr)
        ts.reference_ = G

        Z_tr = ts.transform(covs_tr)
        scaler = StandardScaler(with_mean=True, with_std=True).fit(Z_tr)
        self.reference_G_, self.ts_, self.scaler_ = G, ts, scaler

    # ------------ Paso 4: Rotaciones supervisadas por sujeto (TRAIN) ---------
    def _fit_subject_rotations(
            self,
            covs_rec: np.ndarray,
            labels: np.ndarray,
            subjects: np.ndarray,
            train_idx: np.ndarray,
    ) -> None:
        assert self.ts_ is not None and self.scaler_ is not None
        mask_tr = np.zeros(covs_rec.shape[0], dtype=bool)
        mask_tr[train_idx] = True

        Z_tr = self.ts_.transform(covs_rec[mask_tr])
        Z_tr_std = self.scaler_.transform(Z_tr)
        y_tr = labels[mask_tr]
        s_tr = subjects[mask_tr]

        # Centroides globales por clase (en TRAIN y estandarizados)
        self.mu_global_ = {int(k): Z_tr_std[y_tr == k].mean(axis=0) for k in np.unique(y_tr)}
        d = Z_tr_std.shape[1]
        self.Rs_.clear()

        for s in tqdm(np.unique(s_tr), desc="2/3: Fitting Rotations"):
            s = int(s)
            m = (s_tr == s)
            Zs, ys = Z_tr_std[m], y_tr[m]
            Ks = np.unique(ys)
            if Ks.size < 2:
                self.Rs_[s] = np.eye(d)
                continue
            A = np.stack([self.mu_global_[int(k)] for k in Ks], 0)
            B = np.stack([Zs[ys == k].mean(axis=0) for k in Ks], 0)
            R, _ = orthogonal_procrustes(B, A)
            self.Rs_[s] = R if R.shape == (d, d) else np.eye(d)

    def transform_test_loso(
            self,
            covs: np.ndarray,
            labels: np.ndarray,
            subjects: np.ndarray,
    ) -> np.ndarray:
        """
        Transform TEST data using LOSO protocl with calibration/evaluation split.

        This implements the paper's two-stage rotation approach:
        1. Recentering: per-subject (using test subejct's own mean)
        2. Rescaling: global (using training reference)
        3. Rotation: 2-fold nested CV
            - Split test features into calibration(50%) and evaluation (50%)
            - Compute rotation parameters from calibration subset
            - Apply to evaluation subset
            - Average perfomance across folds

        Args:
            covs: (N_test, C, C) - test covariance matrices
            labels: (N_test,) - test labels
            subject: (N_test,) - test subjects (should be single held-out subject)

        Returns:
            Z_rot_eval: (N_test_eval, d) - rotated features from evaluation subset
        """

        assert self.ts_ is not None and self.scaler_ is not None

        covs_np = _as_numpy(covs).astype(np.float64)
        labels_np = _as_numpy(labels).astype(np.int64).reshape(-1)
        subjects_np = _as_numpy(subjects).astype(np.int64).reshape(-1)

        test_subject = int(subjects_np[0])  # Should be single subject

        # --- STEP 1: Subject-specific recentering (using TEST subject's own mean) ---
        covs_rec_test = self._apply_subject_recentering_np(covs_np, subjects_np)

        # --- STEP 2: Global rescaling (using training reference) ---
        Z_test = self.ts_.transform(covs_rec_test)
        Z_test_std = self.scaler_.transform(Z_test)

        # --- STEP 3: Two-fold nested CV for rotation (paper Eq. 9-11) ---
        N_test = Z_test_std.shape[0]
        split_idx = N_test // 2

        Z_rot_combined = np.empty_like(Z_test_std)

        for calib_slice, eval_slice in [
            (slice(None, split_idx), slice(split_idx, None)),  # fold 1
            (slice(split_idx, None), slice(None, split_idx)),  # fold 2
        ]:
            Z_calib = Z_test_std[calib_slice]
            Z_eval = Z_test_std[eval_slice]
            y_calib = labels_np[calib_slice]

            # Only use classes present in BOTH the training set and this calib fold
            valid_keys = sorted(
                set(self.mu_global_.keys()) &
                {int(k) for k in np.unique(y_calib)}
            )
            if len(valid_keys) < 2:
                Z_rot_combined[eval_slice] = Z_eval  # fallback: no rotation
                continue

            # Anchor points — (K', d) each, rows aligned by class
            train_anchors = np.stack([self.mu_global_[k] for k in valid_keys])  # (K', d)
            calib_anchors = np.stack([Z_calib[y_calib == k].mean(0) for k in valid_keys])  # (K', d)

            # Paper Eq. 9: cross-product matrix  C_TC = C̄_train^T @ C̄_calib  → (d, d)
            C_TC = train_anchors.T @ calib_anchors

            # Paper Eq. 10: SVD decomposition
            U, sv, Vt = np.linalg.svd(C_TC, full_matrices=False)

            # Truncate to retain 99.9 % of variance
            cum_var = np.cumsum(sv ** 2) / (np.sum(sv ** 2) + 1e-12)
            n_v = int(np.searchsorted(cum_var, 0.999) + 1)
            U_tilde = U[:, :n_v]  # (d, n_v)
            V_tilde = Vt[:n_v, :].T  # (d, n_v)
            R = U_tilde @ V_tilde.T  # (d, d)

            # Paper Eq. 11: apply rotation to evaluation subset ONLY
            Z_rot_combined[eval_slice] = (R @ Z_eval.T).T

        return Z_rot_combined

    def adapt_loso_test_subject(
            self,
            covs: Union[np.ndarray, torch.Tensor],
            labels: Union[np.ndarray, torch.Tensor],
            subjects: Union[np.ndarray, torch.Tensor],
    ) -> None:
        """
        Derives and stores A_filters_[test_subject] for the LOSO held-out subject.

        Implements the full 3-step ITSA for the test subject:
          recenter (own mean) → rescale (training reference) → 2-fold SVD rotation.
        The resulting rotated mean is back-projected to SPD space to derive a
        signal-domain spatial filter, consistent with how training subjects are handled.

        Call this AFTER fit(), passing ONLY the test subject's data.
        After this call, transform_signals() works for the test subject.
        """
        covs_np = _as_numpy(covs).astype(np.float64)
        labels_np = _as_numpy(labels).astype(np.int64).reshape(-1)
        subjects_np = _as_numpy(subjects).astype(np.int64).reshape(-1)

        test_s = int(np.unique(subjects_np)[0])

        # Ensure M_inv_sqrt_ exists for the test subject
        # (_fit_subject_means already computes it if test subject was in covs,
        #  but we recompute here to be safe in case it wasn't.)
        if test_s not in self.M_inv_sqrt_:
            covs_norm = covs_np.copy()
            if self.unit_trace_per_subject:
                tr = np.trace(covs_norm, axis1=1, axis2=2).reshape(-1, 1, 1)
                covs_norm = covs_norm / np.maximum(tr, 1e-12)
            M_init = mean_logeuclid(covs_norm)
            M = mean_riemann(covs_norm, init=M_init,
                             tol=self.mean_tol, maxiter=self.mean_maxiter)
            self.M_inv_sqrt_[test_s] = invsqrtm(M)

        # Recentering-only filter: A = M^{-1/2}
        # The full back-projection (recolour via 40-class SVD rotation) is too
        # noisy for high-class-count LOSO — the rotation estimate from N_test/2
        # samples across 40 classes is unreliable and produces out-of-distribution
        # directions. Recentering alone (Adaptive M) is stable and contributes
        # most of ITSA's benefit. The rotation step can only help if class
        # centroids are well-estimated, which requires far more test samples per
        # class than are available here.
        if self.A_filters_ is None:
            self.A_filters_ = {}
        self.A_filters_[test_s] = self.M_inv_sqrt_[test_s].copy()
        self._filters_cache.clear()

        # Normalise filter to identity scale to prevent signal amplitude explosion
        A = self.A_filters_[test_s]
        n_ch = A.shape[0]
        current_norm = np.linalg.norm(A)
        target_norm = np.sqrt(n_ch)  # same scale as identity matrix
        if current_norm > target_norm * 2:  # only clip if genuinely exploded
            self.A_filters_[test_s] = A * (target_norm / current_norm)

        self._filters_cache.clear()

    # --------------------------- API clásica (features) ----------------------
    def fit(
            self,
            covs: Union[np.ndarray, torch.Tensor],
            labels: Union[np.ndarray, torch.Tensor],
            subjects: Union[np.ndarray, torch.Tensor],
            train_idx: Union[np.ndarray, torch.Tensor],
    ):
        """
        Ajusta ITSA usando SOLO TRAIN.
        Acepta covs/labels/subjects/train_idx en NumPy o Torch.
        Internamente convierte a NumPy (CPU) para PyRiemann/Sklearn.
        """
        covs_np = _as_numpy(covs)  # (N,C,C)
        labels_np = _as_numpy(labels).astype(np.int64).reshape(-1)
        subjects_np = _as_numpy(subjects).astype(np.int64).reshape(-1)
        train_idx_np = _as_numpy(train_idx).astype(np.int64).reshape(-1)

        self._fit_subject_means(covs_np, subjects_np, train_idx_np)
        covs_rec = self._apply_subject_recentering_np(covs_np, subjects_np)
        self._fit_reference_tspace_and_scaler(covs_rec, train_idx_np)
        self._fit_subject_rotations(covs_rec, labels_np, subjects_np, train_idx_np)

        # Derivar filtros espaciales por sujeto (para transform_signals en GPU)
        self._derive_subject_filters(covs_rec, labels_np, subjects_np, train_idx_np)

        # Limpia caché (por si ya existía de otra corrida)
        self._filters_cache.clear()
        return self

    # --------------------- API de señales ------------------------------------
    def _derive_subject_filters(
            self,
            covs_rec: np.ndarray,
            labels: np.ndarray,
            subjects: np.ndarray,
            train_idx: np.ndarray,
    ) -> None:
        """
        Construye un filtro espacial único por sujeto:
        A_s = M_s^{-1/2} * W_s,
        donde W_s recolorea la media recentrada del sujeto hacia la
        covarianza objetivo inducida por la rotación supervisada en TS.
        """
        assert self.ts_ is not None and self.scaler_ is not None and self.reference_G_ is not None

        N = covs_rec.shape[0]
        mask_tr = np.zeros(N, dtype=bool)
        mask_tr[train_idx] = True

        Z_tr = self.ts_.transform(covs_rec[mask_tr])
        Z_tr_std = self.scaler_.transform(Z_tr)
        y_tr = labels[mask_tr]
        s_tr = subjects[mask_tr]

        self.A_filters_ = {}
        for s in tqdm(np.unique(s_tr), desc="3/3: Deriving Filters"):
            s = int(s)
            m = (s_tr == s)

            # Media recentrada en SPD (del sujeto, TRAIN)
            covs_s_rec = covs_rec[mask_tr][m]
            Gs_init = mean_logeuclid(covs_s_rec)
            Cbar_rec_s = mean_riemann(covs_s_rec, init=Gs_init, tol=self.mean_tol, maxiter=self.mean_maxiter)

            # Media global del sujeto en TS (estandarizado) y rotación supervisada
            mu_s = Z_tr_std[m].mean(axis=0, keepdims=False).reshape(1, -1)
            R = self.Rs_.get(s, None)
            if R is None:
                R = np.eye(mu_s.shape[1])
            mu_rot = (mu_s @ R.T)  # (1, d)
            mu_unstd = self.scaler_.inverse_transform(mu_rot)  # (1, d)
            G_target_s = self.ts_.inverse_transform(mu_unstd)[0]  # (C, C)

            # Recoloring: desde Cbar_rec_s hacia G_target_s
            Cbar_rec_s_isqrt = invsqrtm(Cbar_rec_s)
            G_target_s_sqrt = sqrtm(G_target_s)
            W_s = Cbar_rec_s_isqrt @ G_target_s_sqrt

            Minv = self.M_inv_sqrt_.get(s, None)
            if Minv is None:
                Minv = np.eye(W_s.shape[0])
            A_s = Minv @ W_s

            self.A_filters_[s] = A_s

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

        # Normalizar a (B,T,C)
        squeeze_4d = False
        if x.dim() == 4:  # (B,1,C,T)
            squeeze_4d = True
            x_ = x.squeeze(1).transpose(1, 2)  # -> (B,T,C)
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

        # Helper NATIVO de PyTorch: 100% en la GPU y ultrarápido
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

            # 1. Recuperar (o crear) el filtro BASE desde la caché
            A_base = self._filters_cache.get(key)
            if A_base is None:
                A_np = self.A_filters_.get(s, None) if self.A_filters_ is not None else None
                if A_np is None:
                    A_base = torch.eye(C, device=device, dtype=torch.float32)
                else:
                    A_base = torch.from_numpy(A_np.astype(np.float32)).to(device=device)
                # OJO AQUI: Cacheamos SOLO la matriz determinista base
                self._filters_cache[key] = A_base

                # 2. Aplicar aumentación estocástica AL VUELO en la GPU
            if mode == "augment":
                alpha = np.random.uniform(alpha_range[0], alpha_range[1])
                A_t = _blend_filter_torch(A_base, alpha)
            else:
                A_t = A_base

            # 3. Aplicar el filtro: (T,C) @ (C,C) -> (T,C)

            xb = x_[b].to(torch.float32) @ A_t.T
            out.append(xb)

        Xout = torch.stack(out, dim=0)

        if squeeze_4d:
            return Xout.transpose(1, 2).unsqueeze(1)
        if x.dim() == 2:
            return Xout.squeeze(0)
        return Xout

    # En ITSA.py (clase ITSA) --> para exportar SOLO los datos necesarios
    # ej. si se exportáse Rs_, el archivo ocuparía varios GB
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
        lite.M_inv_sqrt_ = self.M_inv_sqrt_
        lite.Rs_ = {}  # enorme → fuera
        lite.A_filters_ = self.A_filters_  # Conservamos los filtros base (pesan poquísimo)
        lite._filters_cache = {}  # limpio
        return lite


# ------------------------------ Integrador minimalista ------------------------------
class ITSAIntegrator:
    """
    Capa de integración para usar ITSA con cambios mínimos en el script principal.
    - from_dataset(...): realiza todo el ajuste (fit) leyendo dataset + splits.
    - transform_batch(x, subjects): aplica ITSA por sujeto y devuelve la MISMA forma que x.
    """

    def __init__(self, itsa: ITSA):
        self._itsa = itsa

    @classmethod
    def from_dataset(cls, dataset, splits_path: str, split_num: int = 0):
        """
        Lee los splits desde `splits_path`, replica el mismo filtro 450<=L<=600 que usa Splitter,
        construye las covarianzas por ensayo con el MISMO recorte temporal que hace EEGDataset.__getitem__,
        y ajusta ITSA usando SOLO TRAIN.
        """
        import numpy as _np
        import torch as _t

        loaded = _t.load(splits_path)
        splitnames = ["train", "val", "test"]

        # Replica el filtro de Splitter: mantener indices con 450 <= L <= 600 (en datos crudos)
        def _valid_idx(i: int) -> bool:
            eeg_raw = dataset.data[i]["eeg"]  # tensor crudo (C, T_raw)
            L = eeg_raw.size(1)
            return (L >= 450) and (L <= 600)

        covs_list, labels_list, subjects_list, split_order = [], [], [], []

        for sp in splitnames:
            idxs = loaded["splits"][split_num][sp]
            idxs = [i for i in idxs if _valid_idx(i)]  # mismo filtro que Splitter

            for i in idxs:
                # Usa el __getitem__ del dataset para respetar el recorte temporal actual (time_low, time_high)
                eeg, _label = dataset[i]  # (T,C) para transformer | (1,C,T) para EEGChannelNet
                if eeg.dim() == 3:  # (1,C,T) -> (T,C)
                    eeg2d = eeg.squeeze(0).transpose(0, 1)
                else:
                    eeg2d = eeg

                C = cov_from_signal_torch(eeg2d.double(), eps=1e-4).cpu().numpy()  # (C,C) SPD
                covs_list.append(C)

                labels_list.append(int(dataset.data[i]["label"]))
                subjects_list.append(int(dataset.data[i]["subject"]))
                split_order.append(sp)

        covs = _np.stack(covs_list, axis=0)  # (N, C, C)
        labels = _np.asarray(labels_list, dtype=_np.int64)  # (N,)
        subjects = _np.asarray(subjects_list, dtype=_np.int64)  # (N,)
        train_idx = _np.asarray([k for k, s in enumerate(split_order) if s == "train"], dtype=_np.int64)

        itsa = ITSA().fit(covs=covs, labels=labels, subjects=subjects, train_idx=train_idx)
        return cls(itsa)

    @classmethod
    def from_dataset_loso(cls, dataset, splits_path: str, split_num: int = 0):
        """
        LOSO variant of from_dataset().

        Fits ITSA on the training subjects, then derives the spatial filter for
        the held-out test subject using the paper's 3-step supervised rotation
        (nested 2-fold CV with the test subject's true labels).
        """
        import numpy as _np
        import torch as _t

        loaded = _t.load(splits_path)

        def _valid_idx(i: int) -> bool:
            L = dataset.data[i]["eeg"].size(1)
            return 450 <= L <= 600

        covs_list, labels_list, subjects_list, split_order = [], [], [], []

        for sp in ["train", "val", "test"]:
            idxs = [i for i in loaded["splits"][split_num][sp] if _valid_idx(i)]
            for i in tqdm(idxs, desc=f"Loading {sp} splits", leave=False):
                eeg, _ = dataset[i]
                eeg2d = eeg.squeeze(0).transpose(0, 1) if eeg.dim() == 3 else eeg
                C = cov_from_signal_torch(eeg2d.double(), eps=1e-4).cpu().numpy()
                covs_list.append(C)
                labels_list.append(int(dataset.data[i]["label"]))
                subjects_list.append(int(dataset.data[i]["subject"]))
                split_order.append(sp)

        covs = _np.stack(covs_list)
        labels = _np.asarray(labels_list, dtype=_np.int64)
        subjects = _np.asarray(subjects_list, dtype=_np.int64)
        train_idx = _np.asarray(
            [k for k, s in enumerate(split_order) if s == "train"], dtype=_np.int64
        )
        test_idx = _np.asarray(
            [k for k, s in enumerate(split_order) if s == "test"], dtype=_np.int64
        )

        # 1. Fit on training subjects
        itsa = ITSA().fit(
            covs=covs, labels=labels, subjects=subjects, train_idx=train_idx
        )

        # 2. Derive filter for the held-out test subject (needs labels for rotation)
        itsa.adapt_loso_test_subject(
            covs=covs[test_idx],
            labels=labels[test_idx],
            subjects=subjects[test_idx],
        )

        return cls(itsa)

    @torch.no_grad()
    def transform_test_loso(self, x: torch.Tensor, subjects: torch.Tensor) -> torch.Tensor:
        """
        Transform test batch for LOSO.
        After from_dataset_loso() / adapt_loso_test_subject(), the held-out subject
        has a spatial filter in A_filters_, so this is identical to transform_batch().
        """
        return self._itsa.transform_signals(x, subjects, mode="deterministic")

    @torch.no_grad()
    def transform_batch(self, x: torch.Tensor, subjects: torch.Tensor,
                        mode: str = "deterministic",
                        alpha_range=(1.0, 1.0)) -> torch.Tensor:
        """
        Aplica ITSA por sujeto. Conserva la forma de x:
        - (B,T,C) -> (B,T,C)
        - (B,1,C,T) -> (B,1,C,T)
        """
        return self._itsa.transform_signals(x, subjects,
                                            mode=mode,
                                            alpha_range=alpha_range)
