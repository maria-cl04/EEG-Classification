
# ITSA.py — Alineación Inter‑Sujeto en Espacio Tangente (GPU‑friendly)

**Propósito:** reducir el sesgo entre sujetos sin tocar tu modelo (*transformer*). ITSA aplica un preprocesado por sujeto en **tres pasos** (recentrado → escalado → rotación supervisada) y, además de la API clásica de *features*, expone una **API de señales** que devuelve **exactamente la misma forma** que espera tu red.

---

## Resumen del algoritmo

1. **Recentrado por sujeto**: para cada sujeto \(s\), se calcula su media de covarianza \(M_s\) y se transforma cada covarianza \(C\) como 
   \[ C' = M_s^{-1/2} \, C \, M_s^{-1/2} \]
   eliminando el sesgo medio del sujeto.
2. **Escalado de distribución**: se calcula una referencia global \(G\), se mapean las covarianzas al **espacio tangente** (TS) en \(G\) y se estandarizan (media 0, var 1) con `StandardScaler`.
3. **Alineación rotacional supervisada**: en TS estandarizado, se aprende por sujeto una rotación ortogonal (Procrustes) para alinear los centroides de clase del sujeto con los centroides globales.

Además, se deriva **un filtro espacial por sujeto** \(A_s\) para aplicar ITSA **directamente a las señales**: \(X_{out} = X \, A_s\). Así, el *transformer* sigue recibiendo `(B,T,128)` o `(B,1,128,T)` **sin cambios**.

---

## Arquitectura del archivo

- **CPU / NumPy / PyRiemann / Sklearn (fase *fit*)**: medias por sujeto (\(M_s\)), referencia global \(G\), `TangentSpace`, `StandardScaler` y rotaciones de Procrustes \(R_s\).
- **GPU / Torch (fase *forward* por batch)**: `transform_signals(...)` aplica el filtro \(A_s\) en GPU y devuelve la **misma forma** que entró. Incluye caché por `(s, device, dtype)` para evitar reconversiones NumPy→Torch.

---

## APIs públicas

### 1) `fit(covs, labels, subjects, train_idx)`
Ajusta todos los artefactos usando **solo TRAIN**. Acepta `np.ndarray` o `torch.Tensor` y convierte a NumPy internamente para PyRiemann/Sklearn. Aprende:
- `M_inv_sqrt_` (\(M_s^{-1/2}\)) por sujeto,
- `reference_G_` y `ts_` (espacio tangente),
- `scaler_` (estandarizador en TS),
- `Rs_` (rotaciones por sujeto en TS),
- y **deriva** `A_filters_` (filtros espaciales por sujeto para señales). Limpia `_filters_cache`.

### 2) `transform(covs, subjects)`
Devuelve **características** en TS (estandarizadas y rotadas). Acepta NumPy o Torch; retorna NumPy por coherencia con Sklearn.

### 3) `transform_signals(x, subjects)`
Aplica \(A_s\) a **señales** y devuelve la **misma forma** que `x`:
- Soporta `(B,T,C)`, `(B,1,C,T)` o `(T,C)` (Torch, CPU o GPU; también `np.ndarray`, que se convierte a Torch).
- Cachea filtros por `(s, device, dtype)` para evitar overhead.

---

## Funciones auxiliares

- `to_spd_np(A, eps)` / `to_spd_torch(A, eps)`: fuerzan **SPD** (simetrizan y elevan autovalores `< eps` con `eigh`) en **NumPy** / **Torch**. Útiles para robustez antes/después de productos tipo `A @ C @ A`.
- `cov_from_signal_torch(x, eps)`: covarianza **entre canales** desde una señal `(T, C)` en Torch (GPU si `x` está en GPU), con jitter `eps·I_C`. Devuelve `(C, C)`.

---

## Artefactos aprendidos (atributos)

- `M_inv_sqrt_`: dict `{sujeto: (C,C)}` con \(M_s^{-1/2}\) para recentrar.
- `reference_G_`: referencia `(C,C)` del espacio tangente.
- `ts_`: `TangentSpace` (referencia `G`).
- `scaler_`: `StandardScaler` en TS (TRAIN).
- `Rs_`: dict `{sujeto: (d,d)}` con rotaciones en TS.
- `A_filters_`: dict `{sujeto: (C,C)}` con filtros espaciales por sujeto (para señales).
- `_filters_cache`: dict `{(s, device, dtype): torch.Tensor (C,C)}` para reutilizar filtros en GPU.

---

## Formas y tipos esperados

- **Señales**: el `Dataset` entrega `(T, 128)` tras `.t()`, y el `DataLoader` apila a `(B, T, 128)`; el *transformer* espera `(B,T,128)` o `(B,1,128,T)`. `transform_signals` respeta estas formas.
- **Covarianzas**: `(N, C, C)` con `C=128` (puedes construirlas en GPU con `cov_from_signal_torch` y luego `.cpu().numpy()` para `fit`).
- **Subjects / Labels / train_idx**: vectores 1D alineados con `covs`/muestras (NumPy o Torch; se convierten a NumPy en `fit`).

---

## Integración mínima (ideas y ejemplos **funcionales**)

> A continuación hay dos ejemplos: (A) *setup* para `fit` y (B) uso por batch. **Sustituye** los nombres de variables según tu script si difieren.

### A) Setup — `fit` de ITSA (solo con TRAIN)

```python
import numpy as np
import torch
from ITSA import ITSA, cov_from_signal_torch

# Suponemos que ya tienes:
# - dataset (como en tu script)
# - opt.time_low, opt.time_high
# - splits cargados: loaded = torch.load(opt.splits_path)
# - idx_train: lista de índices de TRAIN

itsa = ITSA()
C_list = []
labels_tr = []
subjects_tr = []

for j in idx_train:
    x = dataset.data[j]["eeg"].float().t()                  # (T_all, 128)
    x = x[opt.time_low:opt.time_high, :]                      # (T, 128)
    if not opt.no_cuda:
        x = x.cuda()
    C = cov_from_signal_torch(x, eps=1e-6)                    # (128,128) torch
    C_list.append(C.detach().cpu().numpy())                   # -> NumPy
    labels_tr.append(int(dataset.data[j]["label"]))
    subjects_tr.append(int(dataset.data[j]["subject"]))

covs_tr = np.stack(C_list, axis=0)
labels_tr = np.asarray(labels_tr, dtype=np.int64)
subjects_tr = np.asarray(subjects_tr, dtype=np.int64)
train_idx = np.arange(len(covs_tr), dtype=np.int64)

itsa.fit(covs_tr, labels_tr, subjects_tr, train_idx)          # ajusta artefactos
```

### B) Uso por batch — `transform_signals` antes del modelo

```python
for split in ("train", "val", "test"):
    for input, target, subject in loaders[split]:
        if not opt.no_cuda:
            input = input.cuda(); target = target.cuda(); subject = subject.cuda()
        input = itsa.transform_signals(input, subject)        # MISMA forma que input
        output = model(input)
        # ... loss, backward (si split == 'train'), etc.
```

---

## Estabilidad numérica

- **Jitter** en covarianzas: `cov_from_signal_torch` añade `eps·I_C` para evitar autovalores demasiado pequeños.
- **SPD explícito**: `to_spd_np`/`to_spd_torch` simetrizan y elevan autovalores `< eps`. Recomendado si hay productos `A @ C @ A` sensibles.
- **Unit trace (opcional)** al estimar \(M_s\) estabiliza la media por sujeto (activado por defecto).

---

## Rendimiento

- **GPU**: `transform_signals` multiplica `(T,C) @ (C,C)` por muestra; con `C=128` es muy barato y paralelo. El caché evita reconversiones por batch.
- **CPU**: el *fit* (medias riemannianas, TS, scaler, Procrustes) ocurre una vez por ejecución y no suele ser el cuello de botella.

---

## Casos límite

- **Sujeto sin muestras en TRAIN**: el cálculo de \(M_s\) hace *fallback* a “todas las muestras del sujeto”. Si en Procrustes no hay ≥2 clases, se usa `R_s = I`.
- **Formas no soportadas** en `transform_signals`: solo `(B,T,C)`, `(B,1,C,T)` o `(T,C)`; de lo contrario se lanza `ValueError`.
- **Cambio de `dtype`/`device`**: `_filters_cache` genera entradas nuevas por `(s, device, dtype)` automáticamente.

---

## Dependencias

`pyriemann`, `scikit-learn`, `scipy`, `torch`. (TS/Scaler/Procrustes se hacen en NumPy/CPU; el preprocesado por batch es en Torch/GPU).

---

## Preguntas frecuentes (FAQ)

**1) ¿Por qué parte del flujo va en CPU y parte en GPU?**  
PyRiemann/Sklearn (medias riemannianas, `TangentSpace`, `StandardScaler`, Procrustes) trabajan con **NumPy/CPU**. El preprocesado por batch sobre **señales** sí se vectoriza en **Torch/GPU** (`transform_signals`).

**2) ¿Hay fuga de información en validación/test?**  
No. Todo se aprende **solo con TRAIN**. En val/test **solo se aplica** lo aprendido; no se usan etiquetas de val/test.

**3) ¿Qué pasa si un sujeto no aparece en TRAIN (p. ej., LOSO)?**  
Para \(M_s\) se hace *fallback* a todas las muestras del sujeto. En Procrustes, si no hay ≥2 clases, se usa `R_s = I`.

**4) ¿Cómo elijo `eps` (jitter)?**  
Valores típicos: `1e-6` a `1e-8`. Si hay inestabilidad (autovalores casi nulos), sube `eps`. Recuerda: el jitter es **(C,C)**, no `(T,T)`.

**5) ¿Puedo cambiar la ventana temporal (`low:high`)?**  
Sí, pero ajusta `max_seq_length` del *transformer* si cambia `T`.

**6) ¿Cómo activo/desactivo ITSA sin tocar el modelo?**  
Envuelve la llamada a `transform_signals(...)` con una bandera (p. ej., `--use-itsa`).

**7) ¿Cómo guardo y recargo ITSA?**  
Puedes serializar el objeto (o sus diccionarios `M_inv_sqrt_`, `A_filters_`, `Rs_`, etc.). Al recargar, reconstruye `ts_` y `scaler_` antes de usar `transform_signals`.

**8) ¿Qué ocurre si cambia el nº de canales `C`?**  
Los filtros y medias son `(C,C)`. Si cambia `C`, hay que volver a ejecutar `fit`.

**9) ¿Cómo depuro problemas de dimensiones?**  
Comprueba: señales `(B,T,128)` o `(B,1,128,T)` al entrar; `A_filters_[s]` `(128,128)`; salida con **misma forma**.

**10) ¿Cómo medir el impacto de ITSA?**  
Ejecuta A/B: **con** y **sin** ITSA (misma semilla/splits) y compara `val/test`. También puedes mirar la varianza inter‑sujeto de logits/embeddings.

---

> **Nota:** Este README describe el `ITSA.py` actualizado para usar GPU donde es posible, manteniendo la compatibilidad con tu *transformer*.
