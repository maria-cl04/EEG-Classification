import numpy as np
import torch
from typing import Dict, Optional, Union

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
        for s in np.unique(subjects):
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

        for s in np.unique(s_tr):
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

    def adapt_subject(
            self,
            covs: np.ndarray,
            labels: np.ndarray,
            subjects: np.ndarray,
            train_idx: np.ndarray,
    ):
        """
        Adapta el espacio ITSA pre-entrenado a un NUEVO sujeto usando sus datos de calibración.
        """
        covs_np = _as_numpy(covs).astype(np.float64)
        labels_np = _as_numpy(labels).astype(np.int64).reshape(-1)
        subjects_np = _as_numpy(subjects).astype(np.int64).reshape(-1)
        train_idx_np = _as_numpy(train_idx).astype(np.int64).reshape(-1)

        mask_tr = np.zeros(covs_np.shape[0], dtype=bool)
        mask_tr[train_idx_np] = True

        for s in np.unique(subjects_np):
            s = int(s)
            m = (subjects_np == s)
            m_tr = m & mask_tr
            covs_s_tr = covs_np[m_tr] if np.any(m_tr) else covs_np[m]

            # 1. Media del sujeto nuevo y recentrado
            if self.unit_trace_per_subject:
                tr = np.trace(covs_s_tr, axis1=1, axis2=2).reshape(-1, 1, 1)
                covs_s_tr = covs_s_tr / np.maximum(tr, 1e-12)

            M_init = mean_logeuclid(covs_s_tr)
            M = mean_riemann(covs_s_tr, init=M_init, tol=self.mean_tol, maxiter=self.mean_maxiter)
            self.M_inv_sqrt_[s] = invsqrtm(M)

            Cc = np.empty_like(covs_s_tr)
            for i, C in enumerate(covs_s_tr):
                C_spd = to_spd_np(C, eps=self.subject_eps)
                Cc[i] = to_spd_np(self.M_inv_sqrt_[s] @ C_spd @ self.M_inv_sqrt_[s], eps=self.subject_eps)

            # 2. Proyección al TS CONGELADO de los sujetos base
            Z_tr = self.ts_.transform(Cc)
            Z_tr_std = self.scaler_.transform(Z_tr)
            ys = labels_np[m_tr] if np.any(m_tr) else labels_np[m]

            # 3. Rotación supervisada hacia los centroides CONGELADOS
            Ks = np.unique(ys)
            d = Z_tr_std.shape[1]
            valid_Ks = [k for k in Ks if int(k) in self.mu_global_]  # Solo clases que conocemos

            if len(valid_Ks) < 2:
                self.Rs_[s] = np.eye(d)
            else:
                A = np.stack([self.mu_global_[int(k)] for k in valid_Ks], 0)  # Destino (Global)
                B = np.stack([Z_tr_std[ys == k].mean(axis=0) for k in valid_Ks], 0)  # Origen (Sujeto Nuevo)
                R, _ = orthogonal_procrustes(B, A)
                self.Rs_[s] = R if R.shape == (d, d) else np.eye(d)

            # 4. Derivar el filtro final (A_s) para el Transformer
            Gs_init = mean_logeuclid(Cc)
            Cbar_rec_s = mean_riemann(Cc, init=Gs_init, tol=self.mean_tol, maxiter=self.mean_maxiter)

            mu_s = Z_tr_std.mean(axis=0, keepdims=False).reshape(1, -1)
            mu_rot = (mu_s @ self.Rs_[s].T)
            mu_unstd = self.scaler_.inverse_transform(mu_rot)
            G_target_s = self.ts_.inverse_transform(mu_unstd)[0]

            W_s = invsqrtm(Cbar_rec_s) @ sqrtm(G_target_s)
            self.A_filters_[s] = to_spd_np(self.M_inv_sqrt_[s] @ W_s)

        self._filters_cache.clear()  # Limpiamos caché GPU
        return self

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

    def transform(
        self,
        covs: Union[np.ndarray, torch.Tensor],
        subjects: Union[np.ndarray, torch.Tensor],
    ) -> np.ndarray:
        """
        Devuelve características ITSA (TS estandarizado y rotado).
        Acepta covs/subjects en NumPy o Torch, y retorna NumPy (para coherencia con TS/Sklearn).
        """
        assert self.ts_ is not None and self.scaler_ is not None

        covs_np = _as_numpy(covs)
        subjects_np = _as_numpy(subjects).astype(np.int64).reshape(-1)

        # Recentrado por sujeto en NumPy (robusto y coherente con fit)
        covs_rec = self._apply_subject_recentering_np(covs_np, subjects_np)

        # TS -> estandarización -> rotación por sujeto
        Z = self.ts_.transform(covs_rec)
        Z_std = self.scaler_.transform(Z)
        Z_rot = Z_std.copy()
        for s, R in self.Rs_.items():
            m = (subjects_np == s)
            if np.any(m):
                Z_rot[m] = Z_rot[m] @ R
        return Z_rot

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
        for s in np.unique(s_tr):
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
            mu_rot = (mu_s @ R.T)                  # (1, d)
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

            # Estabilización numérica (nota: esto impone simetría/SPD)
            self.A_filters_[s] = to_spd_np(A_s)

    @torch.no_grad()
    def transform_signals(self, x: torch.Tensor, subjects: torch.Tensor) -> torch.Tensor:
        """
        Aplica el filtro espacial por sujeto y devuelve la MISMA forma que x.
        x: torch.Tensor con forma (B,T,C), (B,1,C,T) o (T,C)
        subjects: torch.LongTensor con ids de sujeto (B,) o escalar si x es 2D
        """
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)

        device = x.device
        dtype = x.dtype

        # Normalizar a (B,T,C)
        squeeze_4d = False
        if x.dim() == 4:  # (B,1,C,T)
            squeeze_4d = True
            B, _, C, T = x.shape
            x_ = x.squeeze(1).transpose(1, 2)  # -> (B,T,C)
        elif x.dim() == 3:  # (B,T,C)
            x_ = x
            B, T, C = x_.shape
        elif x.dim() == 2:  # (T,C) -> (1,T,C)
            x_ = x.unsqueeze(0)
            B, T, C = x_.shape
        else:
            raise ValueError(f"Forma no soportada: {tuple(x.shape)}")

        if not torch.is_tensor(subjects):
            subjects = torch.tensor([int(subjects)], device=device)
        elif subjects.dim() == 0:
            subjects = subjects.view(1)

        # Aplicar A_s por lote con caché GPU
        out = []
        for b in range(B):
            s = int(subjects[b].item())
            key = (s, device, dtype)

            # Intentamos recuperar el filtro ya convertido y en el device correcto
            A_t = self._filters_cache.get(key)
            if A_t is None:
                # Cargamos el filtro desde el diccionario de NumPy (learned in fit)
                A_np = self.A_filters_.get(s, None) if self.A_filters_ is not None else None
                if A_np is None:
                    A_t = torch.eye(C, device=device, dtype=dtype)
                else:
                    A_t = torch.from_numpy(A_np).to(device=device, dtype=dtype)
                self._filters_cache[key] = A_t  # guardamos en caché para el próximo batch

            # Aplicamos el filtro: (T,C) @ (C,C) -> (T,C)
            xb = x_[b] @ A_t
            out.append(xb)

        Xout = torch.stack(out, dim=0)  # (B,T,C)

        # Volver a la forma original
        if squeeze_4d:
            return Xout.transpose(1, 2).unsqueeze(1)  # (B,1,C,T)
        if x.dim() == 2:
            return Xout.squeeze(0)  # (T,C)
        return Xout


# ------------------------------ Integrador minimalista ------------------------------
import torch as _torch  # alias local para evitar sombra de nombre

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
                if eeg.dim() == 3:        # (1,C,T) -> (T,C)
                    eeg2d = eeg.squeeze(0).transpose(0, 1)
                else:
                    eeg2d = eeg

                C = cov_from_signal_torch(eeg2d).cpu().numpy()  # (C,C) SPD
                covs_list.append(C)

                labels_list.append(int(dataset.data[i]["label"]))
                subjects_list.append(int(dataset.data[i]["subject"]))
                split_order.append(sp)

        covs = _np.stack(covs_list, axis=0)                       # (N, C, C)
        labels = _np.asarray(labels_list, dtype=_np.int64)         # (N,)
        subjects = _np.asarray(subjects_list, dtype=_np.int64)     # (N,)
        train_idx = _np.asarray([k for k, s in enumerate(split_order) if s == "train"], dtype=_np.int64)

        itsa = ITSA().fit(covs=covs, labels=labels, subjects=subjects, train_idx=train_idx)
        return cls(itsa)

    def adapt_from_dataset(self, dataset, splits_path: str, split_num: int = 0):
        """
        Lee los datos del nuevo sujeto y adapta los filtros de ITSA a él.
        """
        import numpy as _np
        import torch as _t

        loaded = _t.load(splits_path)
        splitnames = ["train", "val", "test"]

        def _valid_idx(i: int) -> bool:
            eeg_raw = dataset.data[i]["eeg"]
            L = eeg_raw.size(1)
            return (L >= 450) and (L <= 600)

        covs_list, labels_list, subjects_list, split_order = [], [], [], []

        for sp in splitnames:
            idxs = loaded["splits"][split_num][sp]
            idxs = [i for i in idxs if _valid_idx(i)]

            for i in idxs:
                eeg, _label = dataset[i]
                if eeg.dim() == 3:
                    eeg2d = eeg.squeeze(0).transpose(0, 1)
                else:
                    eeg2d = eeg

                C = cov_from_signal_torch(eeg2d.double(), eps=1e-4).cpu().numpy()
                covs_list.append(C)

                labels_list.append(int(dataset.data[i]["label"]))
                subjects_list.append(int(dataset.data[i]["subject"]))
                split_order.append(sp)

        covs = _np.stack(covs_list, axis=0)
        labels = _np.asarray(labels_list, dtype=_np.int64)
        subjects = _np.asarray(subjects_list, dtype=_np.int64)
        train_idx = _np.asarray([k for k, s in enumerate(split_order) if s == "train"], dtype=_np.int64)

        # Aplicamos la nueva función matemática
        self._itsa.adapt_subject(covs=covs, labels=labels, subjects=subjects, train_idx=train_idx)
        return self

    @_torch.no_grad()
    def transform_batch(self, x: _torch.Tensor, subjects: _torch.Tensor) -> _torch.Tensor:
        """
        Aplica ITSA por sujeto. Conserva la forma de x:
        - (B,T,C) -> (B,T,C)
        - (B,1,C,T) -> (B,1,C,T)
        """
        return self._itsa.transform_signals(x, subjects)