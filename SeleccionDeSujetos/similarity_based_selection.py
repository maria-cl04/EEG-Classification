"""
similarity_based_selection.py
==============================
Similarity-Based Subject Selection for EEG Classification.

Three-phase pipeline:
  Phase 1 — Train a baseline model on all subjects (or load a pre-trained one).
  Phase 2 — Extract per-subject centroids from the frozen baseline; compute a
             pairwise Euclidean distance matrix.
  Phase 3 — For each target subject, select the 2 most similar subjects as
             the training set, 1 subject as validation, train a new model from
             scratch, and report TeA @ max VA.

Usage examples
--------------
# Run the full pipeline from scratch:
python similarity_based_selection.py \
    -ed path/to/eeg_55_95_std.pth

# Skip Phase 1 by supplying a pre-trained baseline:
python similarity_based_selection.py \
    -ed path/to/eeg_55_95_std.pth \
    --baseline-model baseline_model_all_subjects.pth

# Override training epochs for targeted phase only:
python similarity_based_selection.py \
    -ed path/to/eeg_55_95_std.pth \
    --baseline-model baseline_model_all_subjects.pth \
    --targeted-epochs 100
"""

import argparse
import importlib
# import os

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from sklearn.metrics import pairwise_distances # <-- ADD THIS

cudnn.benchmark = True

# ─────────────────────────────────────────────────────────────────────────────
# CLI Arguments
# ─────────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Similarity-Based Subject Selection")

# Data
parser.add_argument('-ed', '--eeg-dataset',
    default=r"C:\Users\maria\PycharmProjects\EEG-Classification\data\eeg_55_95_std.pth",
    help="Path to eeg_55_95_std.pth")
parser.add_argument('-tl', '--time_low',  default=20,  type=float, help="Start time sample")
parser.add_argument('-th', '--time_high', default=460, type=float, help="End time sample")
parser.add_argument('--num-subjects', default=6, type=int, help="Total number of subjects (default: 6)")

# Model
parser.add_argument('-mp', '--model_params',
    default=['num_heads=4', 'num_layers=1', 'd_ff=512', 'd_model=128', 'dropout=0.4'],
    nargs='*', help="Model hyperparameters as key=value pairs")

# Training (shared by Phase 1 and Phase 3)
parser.add_argument('-b',    '--batch-size',              default=128,  type=int)
parser.add_argument('-lr',   '--learning-rate',           default=0.001, type=float)
parser.add_argument('-lrdb', '--learning-rate-decay-by',  default=0.95,  type=float)
parser.add_argument('-lrde', '--learning-rate-decay-every', default=10,  type=int)
parser.add_argument('-e',    '--epochs',          default=1, type=int,
    help="Epochs for Phase 1 baseline training")
parser.add_argument('--targeted-epochs', default=1, type=int,
    help="Epochs for Phase 3 targeted training (defaults to --epochs)")

# Phase 1 shortcut
parser.add_argument('--baseline-model', default='baseline_model_all_subjects.pth',
    help="Path to a pre-trained baseline .pth — skips Phase 1 if provided")

# Subject split for Phase 3
parser.add_argument('--n-train', default=2, type=int,
    help="Number of closest subjects used for training (default: 2)")
parser.add_argument('--n-val', default=1, type=int,
    help="Number of subjects used for validation (default: 1)")

# Fine-tuning stage (Phase 3, Fix 1)
parser.add_argument('--finetune-ratio', default=0.2, type=float,
    help="Fraction of target subject data used for fine-tuning (rest is test). Default: 0.2")
parser.add_argument('--finetune-epochs', default=50, type=int,
    help="Epochs for the fine-tuning stage on the target subject")
parser.add_argument('--finetune-lr', default=0.0001, type=float,
    help="Learning rate for the fine-tuning stage (should be lower than pre-training LR)")
parser.add_argument('--freeze-encoder', default=False, action='store_true',
    help="If set, freeze all encoder layers and only fine-tune the classifier head")


# Misc
parser.add_argument('--no-cuda', default=True, action='store_true')
parser.add_argument('--seed', default=42, type=int)

opt = parser.parse_args()
opt.time_low  = int(opt.time_low)
opt.time_high = int(opt.time_high)
if opt.targeted_epochs == 200 and opt.epochs != 200:
    opt.targeted_epochs = opt.epochs  # inherit unless explicitly set

torch.manual_seed(opt.seed)
np.random.seed(opt.seed)

USE_CUDA = not opt.no_cuda and torch.cuda.is_available()
device   = torch.device("cuda" if USE_CUDA else "cpu")
print(f"Device: {device}")

# Parse model_params into a dict
model_options = {
    key: int(value) if value.isdigit()
         else (float(value) if value.replace('.', '', 1).isdigit() else value)
    for key, value in [x.split("=") for x in opt.model_params]
}


# ─────────────────────────────────────────────────────────────────────────────
# Dataset utilities
# ─────────────────────────────────────────────────────────────────────────────
class EEGDatasetFull:
    """
    Loads the entire EEG dataset (all subjects) without any subject filtering.
    Each call to __getitem__ returns the EEG tensor sliced to [time_low:time_high]
    along the time axis, yielding shape (time_steps, 128).
    """
    def __init__(self, eeg_signals_path):
        loaded = torch.load(eeg_signals_path, weights_only=False)
        self.data   = loaded['dataset']
        self.labels = loaded['labels']
        self.images = loaded['images']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        eeg   = self.data[i]["eeg"].float().t()        # (time, channels)
        eeg   = eeg[opt.time_low:opt.time_high, :]     # (440, 128)
        label = self.data[i]["label"]
        return eeg, label


class SubjectDataset(Dataset):
    """
    Wraps EEGDatasetFull and exposes only samples that belong to the
    given subject_ids, applying the standard EEG length validity check
    (raw eeg.size(1) must be in [450, 600]).
    """
    def __init__(self, full_dataset: EEGDatasetFull, subject_ids):
        self.full_dataset = full_dataset
        subject_set = set(subject_ids)
        self.indices = [
            i for i, sample in enumerate(full_dataset.data)
            if sample['subject'] in subject_set
            and 450 <= sample['eeg'].size(1) <= 600
        ]
        if len(self.indices) == 0:
            raise ValueError(f"No valid samples found for subjects {subject_ids}. "
                             "Check that subject IDs match those in the dataset.")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        return self.full_dataset[self.indices[i]]

class SubjectDatasetSplit(Dataset):
    """
    Splits a SubjectDataset into two non-overlapping partitions.
    Use split='finetune' to get the first `finetune_ratio` fraction,
    or split='test' to get the remainder.

    The split is done on the already-filtered indices inside SubjectDataset,
    so the validity check (eeg length 450-600) is inherited automatically.
    """
    def __init__(self, subject_dataset: SubjectDataset,
                 finetune_ratio: float = 0.2,
                 split: str = 'finetune',
                 seed: int = 42):
        assert 0.0 < finetune_ratio < 1.0, "finetune_ratio must be between 0 and 1"
        assert split in ('finetune', 'test'), "split must be 'finetune' or 'test'"

        rng = np.random.RandomState(seed)
        all_indices = np.arange(len(subject_dataset))
        rng.shuffle(all_indices)

        n_finetune = max(1, int(len(all_indices) * finetune_ratio))

        if split == 'finetune':
            self.indices = all_indices[:n_finetune]
        else:
            self.indices = all_indices[n_finetune:]

        self.subject_dataset = subject_dataset

        print(f"    SubjectDatasetSplit [{split}]: "
              f"{len(self.indices)}/{len(all_indices)} samples "
              f"({len(self.indices)/len(all_indices)*100:.1f}%)")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        return self.subject_dataset[self.indices[i]]

def make_loader(full_dataset: EEGDatasetFull, subject_ids, shuffle=True):
    """Creates a DataLoader filtered to the given subject IDs."""
    ds = SubjectDataset(full_dataset, subject_ids)
    return DataLoader(
        ds,
        batch_size=opt.batch_size,
        drop_last=True,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=USE_CUDA,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Model / optimizer factory
# ─────────────────────────────────────────────────────────────────────────────
def build_model() -> torch.nn.Module:
    """Instantiates a fresh transformer2.Model with the configured options."""
    module = importlib.import_module("models.transformer2")
    return module.Model(**model_options).to(device)


def build_optimizer(model):
    """Returns (optimizer, scheduler) for the given model."""
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=opt.learning_rate_decay_every,
        gamma=opt.learning_rate_decay_by,
    )
    return optimizer, scheduler


# ─────────────────────────────────────────────────────────────────────────────
# Training helpers
# ─────────────────────────────────────────────────────────────────────────────
def run_epoch(model, loader, optimizer=None, train=True):
    """
    Runs one epoch over `loader`.
    Returns (avg_loss, avg_accuracy).
    """
    model.train() if train else model.eval()
    torch.set_grad_enabled(train)

    total_loss = total_correct = total_samples = 0

    for inputs, targets in loader:
        inputs  = inputs.to(device)
        targets = targets.to(device)

        outputs = model(inputs)
        loss    = F.cross_entropy(outputs, targets)

        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss    += loss.item()
        total_correct += outputs.argmax(dim=1).eq(targets).sum().item()
        total_samples += inputs.size(0)

    torch.set_grad_enabled(True)   # restore default
    n_batches = max(len(loader), 1)
    return total_loss / n_batches, total_correct / max(total_samples, 1)


def train_model(model, train_loader, val_loader, test_loader, epochs, tag=""):
    """
    Full training loop with tracking of TeA @ max VA.

    Returns:
        model          — the trained model (weights at last epoch)
        best_val_acc   — highest validation accuracy achieved
        best_test_acc  — test accuracy at the epoch of best_val_acc
        best_epoch     — epoch index at which best_val_acc was reached
    """
    optimizer, scheduler = build_optimizer(model)
    best_val_acc  = 0.0
    best_test_acc = 0.0
    best_epoch    = 0

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, optimizer, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, train=False)
        te_loss,  te_acc  = run_epoch(model, test_loader,  train=False)
        scheduler.step()

        if val_acc >= best_val_acc:
            best_val_acc  = val_acc
            best_test_acc = te_acc
            best_epoch    = epoch

        print(
            f"[{tag}] Epoch {epoch:03d}: "
            f"TrL={tr_loss:.4f} TrA={tr_acc:.4f} | "
            f"VL={val_loss:.4f} VA={val_acc:.4f} | "
            f"TeL={te_loss:.4f} TeA={te_acc:.4f} | "
            f"TeA@maxVA={best_test_acc:.4f} (ep {best_epoch})"
        )

    return model, best_val_acc, best_test_acc, best_epoch

def finetune_model(model, finetune_loader, test_loader, epochs, lr,
                   freeze_encoder=False, tag=""):
    """
    Fine-tuning stage: adapts a pre-trained model to the target subject.

    If freeze_encoder=True, all parameters except the final classifier
    layer are frozen, so only the head is updated. This is safer when the
    fine-tuning set is very small.

    If freeze_encoder=False, the full network is trained at the lower
    fine-tuning LR, allowing the encoder to adapt as well.

    Returns:
        model          — fine-tuned model (weights at last epoch)
        best_test_acc  — test accuracy at the epoch of best fine-tune loss
        best_epoch     — epoch index of best fine-tune loss
    """
    if freeze_encoder:
        # Freeze everything except the classifier head
        for name, param in model.named_parameters():
            param.requires_grad = (name.startswith("classifier"))
        trainable = [p for p in model.parameters() if p.requires_grad]
        print(f"    [{tag}] Encoder frozen — training classifier head only "
              f"({sum(p.numel() for p in trainable)} params)")
    else:
        for param in model.parameters():
            param.requires_grad = True
        print(f"    [{tag}] Full network fine-tuning at LR={lr}")

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr
    )
    # No LR scheduler for fine-tuning — keep it simple and stable
    best_finetune_loss = float('inf')
    best_test_acc      = 0.0
    best_epoch         = 0

    for epoch in range(1, epochs + 1):
        ft_loss, ft_acc = run_epoch(model, finetune_loader, optimizer, train=True)
        te_loss, te_acc = run_epoch(model, test_loader, train=False)

        # Track best by fine-tune loss (not val acc, since fine-tune set IS target subject)
        if ft_loss < best_finetune_loss:
            best_finetune_loss = ft_loss
            best_test_acc      = te_acc
            best_epoch         = epoch

        print(
            f"    [{tag}] FT Epoch {epoch:03d}: "
            f"FtL={ft_loss:.4f} FtA={ft_acc:.4f} | "
            f"TeL={te_loss:.4f} TeA={te_acc:.4f} | "
            f"Best TeA={best_test_acc:.4f} (ep {best_epoch})"
        )

    # Unfreeze all params before returning (in case model is reused)
    for param in model.parameters():
        param.requires_grad = True

    return model, best_test_acc, best_epoch

# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 helpers
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def extract_subject_centroids(model, full_dataset: EEGDatasetFull, num_subjects=6):
    """
    Phase 2 — Feature Extraction & Centroid Calculation.

    Passes every sample through the *frozen* baseline model in eval mode,
    calling model.get_embeddings() to obtain the 128-dim pre-classifier
    representation (post-mean-pooling).  The per-subject mean over all
    embeddings is returned as the Subject Centroid.

    Args:
        model        : trained baseline Model (must expose get_embeddings())
        full_dataset : EEGDatasetFull with all subjects
        num_subjects : number of subjects in the dataset (default 6)

    Returns:
        centroids : dict {subject_id (int): centroid tensor (d_model,)}
    """
    model.eval()
    centroids = {}
    all_embeddings = {}

    for subject_id in range(1, num_subjects + 1):
        try:
            loader = make_loader(full_dataset, [subject_id], shuffle=False)
        except ValueError as e:
            print(f"  [Warning] {e}")
            continue

        embeddings = []
        for inputs, _ in loader:
            inputs = inputs.to(device)
            emb    = model.get_embeddings(inputs)   # (B, d_model)
            embeddings.append(emb.cpu())

        all_emb = torch.cat(embeddings, dim=0)      # (N_subject, d_model)
        centroid = all_emb.mean(dim=0)              # (d_model,)
        centroids[subject_id] = centroid

        all_embeddings[subject_id] = all_emb

        print(f"  Subject {subject_id}: {all_emb.shape[0]} samples  |  "
              f"centroid ‖·‖={centroid.norm().item():.4f}")

    return centroids, all_embeddings

def compute_pooled_covariance(centroids: dict, all_embeddings: dict, regularisation: float=1e-5):
    """
    Computes the pooled within-subject covariance matrix over the embedding
    space.  For each subject the per-sample deviations from that subject's
    centroid are accumulated; the result is then divided by the total degrees
    of freedom (total samples minus number of subjects).

    This gives the Mahalanobis distance its key advantage: distances between
    centroids are scaled by the natural within-subject variability in each
    direction of embedding space.  Dimensions along which subjects vary a
    lot internally are down-weighted; tight, consistent dimensions count more.

    A small regularisation term (epsilon * I) is added to the diagonal to
    ensure the matrix is invertible even if some embedding dimensions are
    nearly constant.

    Args:
    centroids       : dict {subject_id: centroid tensor (d_model,)}
    all_embeddings  : dict {subject_id: tensor (N_subject, d_model)}
    regularisation  : epsilon added to the diagonal for numerical stability

    Returns:
    cov_inv : torch.Tensor (d_model, d_model) — inverse of the pooled
             covariance matrix, ready for Mahalanobis computation
    """
    d_model = next(iter(centroids.values())).shape[0]
    scatter_sum = torch.zeros(d_model, d_model)
    total_df = 0        # total degrees of freedom

    for subjects_id, emb in all_embeddings.items():
        centroid = centroids[subjects_id].unsqueeze(0)      # (1, d_model)
        deviations = emb - centroid                         # (N, d_model)
        scatter_sum += deviations.T @ deviations            # (d_model, d_model)
        total_df    += emb.shape[0]-1                       # N - 1 per subject

    pooled_cov = scatter_sum / total_df                     # (d_model, d_model)

    # Regularise and invert
    # pooled_cov += regularisation * torch.eye(d_model)
    # cov_inv     = torch.linalg.inv(pooled_cov)

    # Symmetrise to cancel any floating-point asymmetry from linalg.inv
    # cov_inv = (cov_inv + cov_inv.T)/2.0

    #Sanity diagnostics
    eigenvalues = torch.linalg.eigvalsh(pooled_cov)
    print(f"  Pooled covariance — min eigenvalue : {eigenvalues.min().item():.6f}")
    print(f"  Pooled covariance — max eigenvalue : {eigenvalues.max().item():.6f}")

    eps = torch.tensor(1e-12, device=eigenvalues.device)
    min_eig = torch.maximum(eigenvalues.min(), eps)

    print(f"  Condition number                   : "
          f"{(eigenvalues.max() / min_eig).item():.2f}")
    # print(f"  Condition number                   : "
    #       f"{(eigenvalues.max() / eigenvalues.max(eigenvalues.min(), torch.tensor(1e-12))).item():.2f}")

    # ── THE FIX ────────────────────────────────────────────────────────────
    # Use Pseudo-Inverse (pinv) instead of adding diagonal regularisation.
    # rcond=1e-5 tells it to safely ignore any eigenvalues smaller than
    # (1e-5 * max_eigenvalue), preventing noise from blowing up.
    cov_inv = torch.linalg.pinv(pooled_cov, rcond=regularisation, hermitian=True)
    # ───────────────────────────────────────────────────────────────────────

    # Symmetrise to cancel any floating-point asymmetry
    cov_inv = (cov_inv + cov_inv.T) / 2.0

    return cov_inv


def compute_distance_matrix(centroids: dict, cov_inv: torch.Tensor = None):
    """
    Computes pairwise distances between all Subject Centroids.

    If cov_inv is provided, uses Mahalanobis distance:
        d_M(i,j) = sqrt( (μi - μj)^T  Σ^-1  (μi - μj) )

    If cov_inv is None, falls back to Euclidean distance (original behaviour).

    Returns:
        dist_matrix : np.ndarray (N, N)
        subject_ids : sorted list of subject IDs
    """
    subject_ids = sorted(centroids.keys())
    n           = len(subject_ids)
    dist_matrix = np.zeros((n, n), dtype=np.float32)
    # dist_matrix = np.zeros((n, n), dtype=np.float32)

    for i, si in enumerate(subject_ids):
        for j, sj in enumerate(subject_ids):
            if i != j:
                diff = centroids[si] - centroids[sj]
                if cov_inv is not None:
                    # Mahalanobis: sqrt(diff^T * Σ^-1 * diff)
                    dist_matrix[i, j] = torch.sqrt(
                        diff @ cov_inv @ diff
                    ).item()
                else:
                    # Euclidean fallback
                    dist_matrix[i, j] = diff.norm().item()

    metric = "Mahalanobis" if cov_inv is not None else "Euclidean"
    col_header = f"  [{metric}]\n       " + "   ".join(f"S{s:d}" for s in subject_ids)
    print(col_header)
    print("     " + "─" * (len(col_header) - 14))
    for i, si in enumerate(subject_ids):
        row = f"S{si:d} │  " + "  ".join(f"{dist_matrix[i, j]:5.3f}" for j in range(n))
        print(row)

    return dist_matrix, subject_ids


def select_training_subjects(target_subject, dist_matrix, subject_ids,
                             n_train=2, n_val=1):
    """
    Phase 3 — For a given target subject returns the training and validation
    subject sets based on centroid distance.

    The `n_train` subjects with the *lowest* distance to the target become the
    training set; the next `n_val` subjects become the validation set.

    Args:
        target_subject : int  — subject ID of the test subject
        dist_matrix    : np.ndarray (N, N)
        subject_ids    : list of subject IDs matching matrix rows/cols
        n_train        : number of training subjects  (default 2)
        n_val          : number of validation subjects (default 1)

    Returns:
        train_subjects : list[int]
        val_subjects   : list[int]
    """
    target_idx = subject_ids.index(target_subject)
    distances  = dist_matrix[target_idx].copy()
    distances[target_idx] = np.inf          # exclude the target itself

    sorted_idx     = np.argsort(distances)
    train_subjects = [subject_ids[i] for i in sorted_idx[:n_train]]
    val_subjects   = [subject_ids[i] for i in sorted_idx[n_train:n_train + n_val]]

    return train_subjects, val_subjects


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("  Similarity-Based Subject Selection — EEG Classification Pipeline")
    print("=" * 70)
    print(f"  Dataset      : {opt.eeg_dataset}")
    print(f"  Time window  : [{opt.time_low}, {opt.time_high}]")
    print(f"  Subjects     : {opt.num_subjects}")
    print(f"  Train subs   : {opt.n_train}  |  Val subs: {opt.n_val}")
    print(f"  Baseline ep  : {opt.epochs}   |  Targeted ep: {opt.targeted_epochs}")
    print()

    all_subjects = list(range(1, opt.num_subjects + 1))

    # ── Load dataset ──────────────────────────────────────────────────────────
    print("Loading dataset …")
    full_dataset = EEGDatasetFull(opt.eeg_dataset)
    print(f"  Total samples: {len(full_dataset)}\n")

    # ─────────────────────────────────────────────────────────────────────────
    # Phase 1 — Baseline training (all subjects)
    # ─────────────────────────────────────────────────────────────────────────
    print("─" * 70)
    print("PHASE 1 — Baseline Model")
    print("─" * 70)

    if opt.baseline_model:
        print(f"  Loading pre-trained baseline: {opt.baseline_model}")
        baseline_model = torch.load(opt.baseline_model, map_location=device, weights_only=False)
    else:
        print("  Training baseline on all subjects (Multi-Subject) …")
        print("  Split: 80% Train | 10% Val | 10% Test across the entire dataset\n")

        baseline_model = build_model()

        all_subjects_ds = SubjectDataset(full_dataset, all_subjects)

        # Calculate 80/10/10 global split sizes
        total_len = len(all_subjects_ds)
        train_len = int(0.8 * total_len)
        val_len = int(0.1 * total_len)
        test_len = total_len - train_len - val_len

        # Randomly split the pooled dataset
        train_ds, val_ds, test_ds = torch.utils.data.random_split(
            all_subjects_ds,
            [train_len, val_len, test_len],
            generator=torch.Generator().manual_seed(opt.seed)
        )

        # Generate DataLoaders (using standard PyTorch utilities instead of make_loader)
        baseline_train = DataLoader(train_ds, batch_size=opt.batch_size, drop_last=True, shuffle=True,
                                    pin_memory=USE_CUDA)
        baseline_val = DataLoader(val_ds, batch_size=opt.batch_size, drop_last=False, shuffle=False,
                                  pin_memory=USE_CUDA)
        baseline_test = DataLoader(test_ds, batch_size=opt.batch_size, drop_last=False, shuffle=False,
                                   pin_memory=USE_CUDA)


        baseline_model, bv, bt, be = train_model(
            baseline_model,
            baseline_train, baseline_val, baseline_test,
            epochs=opt.epochs,
            tag="Baseline",
        )

        save_path = "baseline_model_all_subjects.pth"
        torch.save(baseline_model, save_path)
        print(f"\n  Baseline saved → {save_path}")
        print(f"  Baseline result: VA={bv:.4f}  TeA@maxVA={bt:.4f}  (ep {be})")

    # ─────────────────────────────────────────────────────────────────────────
    # Phase 2 — Centroid extraction & distance matrix
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("PHASE 2 — Subject Centroids & Distance Matrix")
    print("─" * 70)

    print("\n  Extracting subject centroids from frozen baseline …")

    centroids, all_embeddings = extract_subject_centroids(
        baseline_model, full_dataset, opt.num_subjects
    )

    print("\n  Computing pooled within-subject covariance ...")
    cov_inv = compute_pooled_covariance(centroids, all_embeddings)

    print("\n  Pairwise Mahalanobis distance matrix:")
    dist_matrix, subject_ids = compute_distance_matrix(centroids, cov_inv)

    np.save("subject_distance_matrix.npy", dist_matrix)
    print("\n  Distance matrix saved → subject_distance_matrix.npy")

    # ─────────────────────────────────────────────────────────────────────────
    # Phase 3 — Targeted fine-tuning per subject
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("PHASE 3 — Targeted Training (similarity-selected subjects)")
    print("─" * 70)

    results = {}

    for target_subject in all_subjects:
        train_subs, val_subs = select_training_subjects(
            target_subject, dist_matrix, subject_ids,
            n_train=opt.n_train, n_val=opt.n_val,
        )
        # ── Stage A: build data loaders ───────────────────────────────────
        train_loader = make_loader(full_dataset, train_subs, shuffle=True)
        val_loader = make_loader(full_dataset, val_subs, shuffle=False)

        # Split target subject into fine-tune portion and held-out test portion
        target_subject_ds = SubjectDataset(full_dataset, [target_subject])
        finetune_ds = SubjectDatasetSplit(target_subject_ds,
                                          finetune_ratio=opt.finetune_ratio,
                                          split='finetune',
                                          seed=opt.seed)
        test_ds = SubjectDatasetSplit(target_subject_ds,
                                      finetune_ratio=opt.finetune_ratio,
                                      split='test',
                                      seed=opt.seed)

        finetune_loader = DataLoader(finetune_ds, batch_size=opt.batch_size,
                                     drop_last=False, shuffle=True,
                                     num_workers=0, pin_memory=USE_CUDA)
        test_loader = DataLoader(test_ds, batch_size=opt.batch_size,
                                 drop_last=False, shuffle=False,
                                 num_workers=0, pin_memory=USE_CUDA)

        # ── Stage B: pre-train on similar subjects ────────────────────────
        print(f"\n     [Stage B] Pre-training on subjects {train_subs} "
              f"(val: {val_subs}) for {opt.targeted_epochs} epochs …")
        targeted_model = build_model()

        _, pretrain_val, pretrain_test, pretrain_ep = train_model(
            targeted_model,
            train_loader, val_loader, test_loader,
            epochs=opt.targeted_epochs,
            tag=f"PreTrain-S{target_subject}",
        )
        print(f"     [Stage B] Done — VA={pretrain_val:.4f}  "
              f"TeA@maxVA={pretrain_test:.4f}  (ep {pretrain_ep})")

        # ── Stage C: fine-tune on target subject ──────────────────────────
        print(f"\n     [Stage C] Fine-tuning on Subject {target_subject} "
              f"({opt.finetune_ratio * 100:.0f}% of data, "
              f"{opt.finetune_epochs} epochs, "
              f"freeze_encoder={opt.freeze_encoder}) …")
        _, finetune_test, finetune_ep = finetune_model(
            targeted_model,
            finetune_loader, test_loader,
            epochs=opt.finetune_epochs,
            lr=opt.finetune_lr,
            freeze_encoder=opt.freeze_encoder,
            tag=f"FineTune-S{target_subject}",
        )
        print(f"     [Stage C] Done — Best TeA={finetune_test:.4f}  (ep {finetune_ep})")

        # ── Record both stages for comparison ─────────────────────────────
        results[target_subject] = {
            "train_subjects": train_subs,
            "val_subjects": val_subs,
            "pretrain_val_acc": pretrain_val,
            "pretrain_test_acc": pretrain_test,  # TeA@maxVA without fine-tuning
            "pretrain_best_epoch": pretrain_ep,
            "finetune_test_acc": finetune_test,  # TeA after fine-tuning
            "finetune_best_epoch": finetune_ep,
        }

        model_path = f"targeted_model_subject{target_subject}.pth"
        torch.save(targeted_model, model_path)
        print(f"     Model saved → {model_path}")

    # ─────────────────────────────────────────────────────────────────────────
    # Summary
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("FINAL RESULTS — TeA @ max VA per target subject")
    print("=" * 70)
    mean_acc = 0.0
    for subj, res in results.items():
        print(
            f"  Subject {subj} "
            f"[train={res['train_subjects']}, val={res['val_subjects']}]  "
            f"PreTrain TeA@maxVA={res['pretrain_test_acc']:.4f}  →  "
            f"FineTune TeA={res['finetune_test_acc']:.4f}"
        )
        mean_acc += res['finetune_test_acc']

    print(f"\n  Mean TeA@maxVA (all subjects): {mean_acc / len(results):.4f}")
    print("=" * 70)

    # Persist results dict for downstream analysis
    torch.save(results, "similarity_selection_results.pth")
    print("\nFull results saved → similarity_selection_results.pth")


if __name__ == "__main__":
    main()
