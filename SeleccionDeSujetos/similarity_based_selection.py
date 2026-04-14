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
import os

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

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
parser.add_argument('-e',    '--epochs',          default=200, type=int,
    help="Epochs for Phase 1 baseline training")
parser.add_argument('--targeted-epochs', default=200, type=int,
    help="Epochs for Phase 3 targeted training (defaults to --epochs)")

# Phase 1 shortcut
parser.add_argument('--baseline-model', default='',
    help="Path to a pre-trained baseline .pth — skips Phase 1 if provided")

# Subject split for Phase 3
parser.add_argument('--n-train', default=2, type=int,
    help="Number of closest subjects used for training (default: 2)")
parser.add_argument('--n-val', default=1, type=int,
    help="Number of subjects used for validation (default: 1)")

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

        print(f"  Subject {subject_id}: {all_emb.shape[0]} samples  |  "
              f"centroid ‖·‖={centroid.norm().item():.4f}")

    return centroids


def compute_distance_matrix(centroids: dict):
    """
    Phase 2 — Distance Matrix Calculation.

    Computes pairwise Euclidean distances between all Subject Centroids and
    prints the resulting N×N matrix.

    Returns:
        dist_matrix  : np.ndarray of shape (N, N)
        subject_ids  : sorted list of subject IDs (row/col index mapping)
    """
    subject_ids = sorted(centroids.keys())
    n = len(subject_ids)
    dist_matrix = np.zeros((n, n), dtype=np.float32)

    for i, si in enumerate(subject_ids):
        for j, sj in enumerate(subject_ids):
            if i != j:
                diff = centroids[si] - centroids[sj]
                dist_matrix[i, j] = diff.norm().item()

    # Pretty-print the matrix
    col_header = "       " + "   ".join(f"S{s:d}" for s in subject_ids)
    print(col_header)
    print("     " + "─" * (len(col_header) - 5))
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
        baseline_model = torch.load(opt.baseline_model, weights_only=False).to(device)
    else:
        print("  Training baseline on all subjects …")
        print("  Split: subjects 1–4 → train | subject 5 → val | subject 6 → test")
        print("  (Edit this block to use your pre-computed LOSO splits instead.)\n")

        baseline_model = build_model()

        # Default split: first four subjects train, fifth val, sixth test.
        # Replace make_loader calls below with your preferred split strategy,
        # e.g. load a LOSO .pth split file as in the original training script.
        baseline_train = make_loader(full_dataset, all_subjects[:4], shuffle=True)
        baseline_val   = make_loader(full_dataset, [all_subjects[4]], shuffle=False)
        baseline_test  = make_loader(full_dataset, [all_subjects[5]], shuffle=False)

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
    centroids = extract_subject_centroids(baseline_model, full_dataset, opt.num_subjects)

    print("\n  Pairwise Euclidean distance matrix:")
    dist_matrix, subject_ids = compute_distance_matrix(centroids)

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

        print(f"\n  ── Target: Subject {target_subject} ──────────────────────────")
        print(f"     Train subjects : {train_subs}")
        print(f"     Val   subjects : {val_subs}")
        print(f"     Test  subject  : [{target_subject}]")

        # Compute distances for display
        ti = subject_ids.index(target_subject)
        for s in train_subs + val_subs:
            si = subject_ids.index(s)
            print(f"       dist(S{target_subject}, S{s}) = "
                  f"{dist_matrix[ti, si]:.4f}")

        train_loader = make_loader(full_dataset, train_subs,       shuffle=True)
        val_loader   = make_loader(full_dataset, val_subs,         shuffle=False)
        test_loader  = make_loader(full_dataset, [target_subject], shuffle=False)

        targeted_model = build_model()   # fresh weights every time

        _, best_val, best_test, best_ep = train_model(
            targeted_model,
            train_loader, val_loader, test_loader,
            epochs=opt.targeted_epochs,
            tag=f"Target-S{target_subject}",
        )

        results[target_subject] = {
            "train_subjects" : train_subs,
            "val_subjects"   : val_subs,
            "best_val_acc"   : best_val,
            "best_test_acc"  : best_test,
            "best_epoch"     : best_ep,
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
            f"TeA@maxVA = {res['best_test_acc']:.4f}  (ep {res['best_epoch']})"
        )
        mean_acc += res['best_test_acc']

    print(f"\n  Mean TeA@maxVA (all subjects): {mean_acc / len(results):.4f}")
    print("=" * 70)

    # Persist results dict for downstream analysis
    torch.save(results, "similarity_selection_results.pth")
    print("\nFull results saved → similarity_selection_results.pth")


if __name__ == "__main__":
    main()
