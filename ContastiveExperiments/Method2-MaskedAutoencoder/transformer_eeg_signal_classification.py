##### Define options
import argparse

parser = argparse.ArgumentParser(description="Template")
# Dataset options

### BLOCK DESIGN ###
parser.add_argument('-ed', '--eeg-dataset', default=r"C:\Users\maria\Documents\Beca\Transformer-22Abril\data\eeg_55_95_std.pth", help="EEG dataset path")
parser.add_argument('-sp', '--splits-path', default=r"C:\Users\maria\Documents\Beca\Splits-LOSO\block_splits_LOSO_subject1.pth", help="splits path")

parser.add_argument('-sn', '--split-num', default=0, type=int, help="split number")
parser.add_argument('-sub', '--subject', default=1, type=int,
                    help="choose a subject from 1 to 6, default is 0 (all subjects)")
parser.add_argument('-tl', '--time_low', default=20, type=float, help="lowest time value")
parser.add_argument('-th', '--time_high', default=460, type=float, help="highest time value")
parser.add_argument('-mt', '--model_type', default='transformer2',
                    help='specify which generator should be used: lstm|EEGChannelNet')
parser.add_argument('-mp', '--model_params',
                    default=['num_heads=4', 'num_layers=1', 'd_ff=512', 'd_model=128', 'dropout=0.4'], nargs='*',
                    help='list of key=value pairs of model options')
parser.add_argument('--pretrained_net', default='', help="path to pre-trained net (to continue training)")

# Training options
parser.add_argument("-b", "--batch_size", default=128, type=int, help="batch size")
parser.add_argument('-o', '--optim', default="Adam", help="optimizer")
parser.add_argument('-lr', '--learning-rate', default=0.001, type=float, help="learning rate")
parser.add_argument('-lrdb', '--learning-rate-decay-by', default=0.95, type=float, help="learning rate decay factor")
parser.add_argument('-lrde', '--learning-rate-decay-every', default=10, type=int, help="learning rate decay period")
parser.add_argument('-dw', '--data-workers', default=4, type=int, help="data loading workers")
parser.add_argument('-e', '--epochs', default=200, type=int, help="training epochs")
parser.add_argument('-sc', '--saveCheck', default=200, type=int, help="save checkpoint every N epochs")
parser.add_argument('--no-cuda', default=False, help="disable CUDA", action="store_true")

### NEW ### — contrastive learning arguments
parser.add_argument('--supcon', default=True, action=argparse.BooleanOptionalAction,
                    help="enable supervised contrastive loss (default: True)")
# lambda_supcon: weight of the contrastive term relative to cross-entropy.
# Formula: total_loss = CE_loss + lambda_supcon * SupCon_loss
# Recommended range: 0.1 (safe start) → 0.5 (stronger contrastive signal).
# If validation accuracy degrades vs. baseline, reduce to 0.05.
parser.add_argument('--lambda-supcon', default=0.1, type=float,
                    help="SupCon loss weight (default 0.1). Try 0.05–0.5.")
# Temperature for contrastive loss.  0.07 is the SupCon paper default.
# Lower = sharper / more discriminative; higher = softer / more stable.
parser.add_argument('--supcon-temp', default=0.07, type=float,
                    help="contrastive temperature (default 0.07)")
# proj_dim: dimensionality of the projection head output.
# 128 matches d_model and works well in practice.
parser.add_argument('--proj-dim', default=128, type=int,
                    help="projection head output dim for contrastive loss (default 128)")

opt = parser.parse_args()
print(opt)

opt.time_low = int(opt.time_low)
opt.time_high = int(opt.time_high)

# Imports
import sys
import os
import random
import math
import time
import torch

torch.utils.backcompat.broadcast_warning.enabled = True
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import torch.backends.cudnn as cudnn
import torch.serialization

cudnn.benchmark = True
from scipy.fftpack import fft, rfft, fftfreq, irfft, ifft, rfftfreq
from scipy import signal
import numpy as np
import models
import importlib



# ---------------------------------------------------------------------------
# Dataset / Splitter (unchanged)
# ---------------------------------------------------------------------------

class EEGDataset:
    def __init__(self, eeg_signals_path):
        loaded = torch.load(eeg_signals_path)
        if opt.subject != 0:
            self.data = [loaded['dataset'][i] for i in range(len(loaded['dataset'])) if
                         loaded['dataset'][i]['subject'] == opt.subject]
        else:
            self.data = loaded['dataset']
        self.labels = loaded["labels"]
        self.images = loaded["images"]
        self.size = len(self.data)

    def __len__(self):
        return self.size

    def __getitem__(self, i):
        eeg = self.data[i]["eeg"].float().t()
        eeg = eeg[opt.time_low:opt.time_high, :]
        if opt.model_type == "EEGChannelNet":
            eeg = eeg.t()
            eeg = eeg.view(1, 128, opt.time_high - opt.time_low)
        label = self.data[i]["label"]
        return eeg, label


class Splitter:
    def __init__(self, dataset, split_path, split_num=0, split_name="train"):
        self.dataset = dataset
        loaded = torch.load(split_path)
        self.split_idx = loaded["splits"][split_num][split_name]
        self.split_idx = [i for i in self.split_idx if 450 <= self.dataset.data[i]["eeg"].size(1) <= 600]
        self.size = len(self.split_idx)

    def __len__(self):
        return self.size

    def __getitem__(self, i):
        eeg, label = self.dataset[self.split_idx[i]]
        return eeg, label


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

dataset = EEGDataset(opt.eeg_dataset)
loaders = {
    split: DataLoader(
        Splitter(dataset, split_path=opt.splits_path, split_num=opt.split_num, split_name=split),
        batch_size=opt.batch_size, drop_last=True, shuffle=True
    )
    for split in ["train", "val", "test"]
}

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

model_options = {
    key: int(value) if value.isdigit() else (float(value) if value[0].isdigit() else value)
    for (key, value) in [x.split("=") for x in opt.model_params]
}

module = importlib.import_module("models." + opt.model_type)

### CHANGED ###
# Pass proj_dim to the model so it builds a projection head.
# If --no-supcon is set, proj_dim is still passed but the head will simply
# never be called (zero overhead at inference time).
model = module.Model(**model_options, proj_dim=opt.proj_dim)

optimizer = getattr(torch.optim, opt.optim)(model.parameters(), lr=opt.learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=opt.learning_rate_decay_every,
    gamma=opt.learning_rate_decay_by
)

if not opt.no_cuda:
    model.cuda()
    print("Copied to CUDA")

if opt.pretrained_net != '':
    model = torch.load(opt.pretrained_net, weights_only=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=opt.learning_rate_decay_every,
        gamma=opt.learning_rate_decay_by
    )
    print(model)

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

losses_per_epoch = {"train": [], "val": [], "test": []}
accuracies_per_epoch = {"train": [], "val": [], "test": []}

best_accuracy = 0
best_accuracy_val = 0
best_epoch = 0

predicted_labels = []
correct_labels = []

for epoch in range(1, opt.epochs + 1):
    losses = {"train": 0, "val": 0, "test": 0}
    accuracies = {"train": 0, "val": 0, "test": 0}
    counts = {"train": 0, "val": 0, "test": 0}

    if opt.optim == "SGD":
        lr = opt.learning_rate * (opt.learning_rate_decay_by ** (epoch // opt.learning_rate_decay_every))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    for split in ("train", "val", "test"):
        if split == "train":
            model.train()
            torch.set_grad_enabled(True)
        else:
            model.eval()
            torch.set_grad_enabled(False)

        for i, (input, target) in enumerate(loaders[split]):
            if not opt.no_cuda:
                input = input.to("cuda")
                target = target.to("cuda")


            output = model(input)
            loss = F.cross_entropy(output, target)

            losses[split] += loss.item()

            _, pred = output.data.max(1)
            correct = pred.eq(target.data).sum().item()
            accuracies[split] += correct / input.data.size(0)
            counts[split] += 1

            if split == "train":
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    if accuracies["val"] / counts["val"] >= best_accuracy_val:
        best_accuracy_val = accuracies["val"] / counts["val"]
        best_accuracy = accuracies["test"] / counts["test"]
        best_epoch = epoch

    TrL = losses["train"] / counts["train"]
    TrA = accuracies["train"] / counts["train"]
    VL  = losses["val"]   / counts["val"]
    VA  = accuracies["val"]   / counts["val"]
    TeL = losses["test"]  / counts["test"]
    TeA = accuracies["test"]  / counts["test"]

    print(
        "Model: {11} - Subject {12} - Time interval: [{9}-{10}] [{9}-{10} Hz] - "
        "Epoch {0}: TrL={1:.4f}, TrA={2:.4f}, VL={3:.4f}, VA={4:.4f}, "
        "TeL={5:.4f}, TeA={6:.4f}, TeA at max VA = {7:.4f} at epoch {8:d}".format(
            epoch, TrL, TrA, VL, VA, TeL, TeA,
            best_accuracy, best_epoch,
            opt.time_low, opt.time_high, opt.model_type, opt.subject
        )
    )

    losses_per_epoch['train'].append(TrL)
    losses_per_epoch['val'].append(VL)
    losses_per_epoch['test'].append(TeL)
    accuracies_per_epoch['train'].append(TrA)
    accuracies_per_epoch['val'].append(VA)
    accuracies_per_epoch['test'].append(TeA)

    scheduler.step()

    if epoch % opt.saveCheck == 0:
        torch.save(model, '%s__subject%d_epoch_%d.pth' % (opt.model_type, opt.subject, epoch))
