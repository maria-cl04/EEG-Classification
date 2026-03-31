"""
transformer_loso_proto.py — Prototypical-Network inference for LOSO evaluation.

WHAT THIS DOES
--------------
After a model has been trained on 5 subjects (the standard LOSO training loop),
this script replaces the linear classifier with a nearest-prototype classifier:

1.  Extract embeddings (mean-pooled encoder output) for every TRAINING sample.
2.  Compute one prototype per class = mean embedding of all training samples
    in that class.  (This is exact prototypical-network inference.)
3.  For each TEST sample, find the nearest prototype (Euclidean distance in
    embedding space) and assign its label.
4.  Report accuracy.

WHY THIS HELPS FOR LOSO
------------------------
The linear head in the original model is fit on 5 subjects and evaluated on
a 6th.  Subject-specific biases in the classifier weights hurt cross-subject
transfer.  Prototypical inference is non-parametric: it only requires that
same-class embeddings cluster together, a property that the SupCon loss
(in the main training script) directly trains for.

HOW TO RUN
----------
  python transformer_loso_proto.py \
      --model  loso_model_subject1.pth \
      --eeg-dataset path/to/eeg_55_95_std.pth \
      --splits-path path/to/block_splits_LOSO_subject1.pth \
      --test-subject 1

The script runs BOTH standard softmax evaluation AND prototypical evaluation
so you can report the delta in your thesis.
"""

import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser()
parser.add_argument('--model',        required=True, help="Path to saved .pth model")
parser.add_argument('--eeg-dataset',  required=True)
parser.add_argument('--splits-path',  required=True)
parser.add_argument('--split-num',    default=0, type=int)
parser.add_argument('--test-subject', default=1, type=int,
                    help="The held-out subject (1–6)")
parser.add_argument('--time-low',     default=20,  type=int)
parser.add_argument('--time-high',    default=460, type=int)
parser.add_argument('--batch-size',   default=256, type=int)
parser.add_argument('--no-cuda',      action='store_true')
parser.add_argument('--distance',     default='euclidean',
                    choices=['euclidean', 'cosine'],
                    help="Distance metric for prototype matching")
opt = parser.parse_args()

device = torch.device('cpu' if opt.no_cuda or not torch.cuda.is_available() else 'cuda')
T = opt.time_high - opt.time_low


# ---------------------------------------------------------------------------
# Dataset helpers (same as main script, kept self-contained here)
# ---------------------------------------------------------------------------

class EEGDataset:
    def __init__(self, path, time_low, time_high):
        loaded = torch.load(path)
        self.data   = loaded['dataset']
        self.labels = loaded['labels']
        self.images = loaded['images']
        self.time_low  = time_low
        self.time_high = time_high

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        eeg = self.data[i]['eeg'].float().t()
        eeg = eeg[self.time_low:self.time_high, :]
        return eeg, self.data[i]['label']


class Splitter:
    def __init__(self, dataset, split_path, split_num, split_name):
        self.dataset = dataset
        loaded = torch.load(split_path)
        self.split_idx = loaded['splits'][split_num][split_name]
        self.split_idx = [i for i in self.split_idx
                          if 450 <= self.dataset.data[i]['eeg'].size(1) <= 600]

    def __len__(self):
        return len(self.split_idx)

    def __getitem__(self, i):
        return self.dataset[self.split_idx[i]]


dataset = EEGDataset(opt.eeg_dataset, opt.time_low, opt.time_high)

train_loader = DataLoader(
    Splitter(dataset, opt.splits_path, opt.split_num, 'train'),
    batch_size=opt.batch_size, shuffle=False, drop_last=False
)
test_loader = DataLoader(
    Splitter(dataset, opt.splits_path, opt.split_num, 'test'),
    batch_size=opt.batch_size, shuffle=False, drop_last=False
)


# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------

model = torch.load(opt.model, weights_only=False, map_location=device)
model.to(device)
model.eval()

num_classes = model.classifier.out_features
print(f"\nLoaded model from: {opt.model}")
print(f"num_classes={num_classes}  |  test-subject={opt.test_subject}")
print(f"distance metric: {opt.distance}\n")


# ---------------------------------------------------------------------------
# Embedding extraction helper
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_embeddings(loader):
    """Returns (embeddings, labels) as CPU tensors."""
    all_emb = []
    all_lbl = []
    for x, y in loader:
        x = x.to(device)
        # Use the get_embedding() method added to transformer2.Model.
        # This returns the mean-pooled encoder output (B, d_model)
        # WITHOUT going through the classifier or projection head.
        emb = model.get_embedding(x)         # (B, d_model)
        all_emb.append(emb.cpu())
        all_lbl.append(y.cpu())
    return torch.cat(all_emb, dim=0), torch.cat(all_lbl, dim=0)


# ---------------------------------------------------------------------------
# Standard softmax evaluation (baseline to compare against)
# ---------------------------------------------------------------------------

@torch.no_grad()
def softmax_accuracy(loader):
    correct = 0
    total   = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        pred = logits.argmax(dim=1)
        correct += pred.eq(y).sum().item()
        total   += y.size(0)
    return correct / total


# ---------------------------------------------------------------------------
# Prototypical evaluation
# ---------------------------------------------------------------------------

def compute_prototypes(embeddings, labels, num_classes):
    """
    Compute class prototypes as mean embedding per class.

    Args:
        embeddings : (N_train, d_model)
        labels     : (N_train,)
        num_classes: int

    Returns:
        prototypes : (num_classes, d_model)
    """
    d = embeddings.size(1)
    prototypes = torch.zeros(num_classes, d)
    counts     = torch.zeros(num_classes)

    for c in range(num_classes):
        mask = labels == c
        if mask.sum() > 0:
            prototypes[c] = embeddings[mask].mean(dim=0)
            counts[c]     = mask.sum()
        # If a class has no training samples (shouldn't happen) prototype stays 0

    n_empty = (counts == 0).sum().item()
    if n_empty > 0:
        print(f"  Warning: {n_empty} classes have no training samples in this split.")

    return prototypes


def proto_accuracy(test_emb, test_labels, prototypes, distance='euclidean'):
    """
    Classify test samples by nearest prototype.

    Args:
        test_emb   : (N_test, d)
        test_labels: (N_test,)
        prototypes : (num_classes, d)
        distance   : 'euclidean' or 'cosine'

    Returns:
        accuracy: float
    """
    if distance == 'euclidean':
        # Expand for broadcasting: (N_test, 1, d) - (1, num_classes, d)
        diff = test_emb.unsqueeze(1) - prototypes.unsqueeze(0)   # (N, K, d)
        dists = diff.pow(2).sum(dim=2)                           # (N, K)
        pred = dists.argmin(dim=1)                               # (N,)

    elif distance == 'cosine':
        # Cosine distance = 1 - cosine_similarity; minimise = maximise similarity
        test_norm  = F.normalize(test_emb,   dim=1)   # (N, d)
        proto_norm = F.normalize(prototypes, dim=1)   # (K, d)
        sim  = torch.matmul(test_norm, proto_norm.T)  # (N, K)
        pred = sim.argmax(dim=1)                       # (N,)

    correct = pred.eq(test_labels).sum().item()
    return correct / test_labels.size(0)


# ---------------------------------------------------------------------------
# Run evaluation
# ---------------------------------------------------------------------------

print("Extracting training embeddings …")
train_emb, train_lbl = extract_embeddings(train_loader)
print(f"  Train embeddings: {train_emb.shape}  |  labels range: {train_lbl.min()}–{train_lbl.max()}")

print("Extracting test embeddings …")
test_emb, test_lbl = extract_embeddings(test_loader)
print(f"  Test  embeddings: {test_emb.shape}")

print("\nComputing class prototypes …")
prototypes = compute_prototypes(train_emb, train_lbl, num_classes)
print(f"  Prototypes: {prototypes.shape}")

print("\n--- Results ---")
softmax_acc = softmax_accuracy(test_loader)
proto_acc   = proto_accuracy(test_emb, test_lbl, prototypes, distance=opt.distance)

print(f"  Softmax classifier accuracy : {softmax_acc*100:.2f}%")
print(f"  Prototypical ({opt.distance}) accuracy : {proto_acc*100:.2f}%")
print(f"  Delta                       : {(proto_acc - softmax_acc)*100:+.2f}%")
print(f"\n  Chance level (40 classes)   :  2.50%")
print()
