##### Define options
import argparse

parser = argparse.ArgumentParser(description="Template")
# Dataset options

# Data - Data needs to be pre-filtered and filtered data is available

### BLOCK DESIGN ###
# Data
parser.add_argument('-ed', '--eeg-dataset', default=r"/kaggle/input/datasets/marii04/eeg-datasets/eeg_55_95_std.pth", help="EEG dataset path") #55-95Hz
# parser.add_argument('-ed', '--eeg-dataset', default=r"data\block\eeg_5_95_std.pth", help="EEG dataset path")  # 5-95Hz
# parser.add_argument('-ed', '--eeg-dataset', default=r"data\block\eeg_14_70_std.pth", help="EEG dataset path") #14-70Hz
# Splits
#parser.add_argument('-sp', '--splits-path', default=r"/kaggle/input/datasets/marii04/single-subject-splits/block_splits_by_single_subject_1.pth", help="splits path") #Subject 1
#parser.add_argument('-sp', '--splits-path', default=r"/kaggle/input/datasets/marii04/single-subject-splits/block_splits_by_single_subject23456.pth", help="splits path") #Subjects 2,3,4,5,6
parser.add_argument('-sp', '--splits-path', default=r"/kaggle/input/datasets/marii04/splits-all-subjects/block_splits_by_image_all.pth", help="splits path") #All subjects
#parser.add_argument('-sp', '--splits-path', default=r"/kaggle/input/datasets/marii04/splits-loso/block_splits_LOSO_subject6.pth", help="splits path") #LOSO

#parser.add_argument('-sp', '--splits-path', default=r"/kaggle/input/datasets/marii04/splits-experiments-with-fine-tuning/splits_fineTuning_subject1.pth", help="splits path") #Subject 1
#parser.add_argument('-sp', '--splits-path', default=r"/kaggle/input/datasets/marii04/splits-by-percent-for-fine-tuning-experiments/splits_fineTuning_subject1_70percent.pth", help="splits path") #Subjects 2,3,4,5,6

### BLOCK DESIGN ###

parser.add_argument('-sn', '--split-num', default=0, type=int, help="split number")  # leave this always to zero.

# Subject selecting
parser.add_argument('-sub', '--subject', default=1, type=int,
                    help="choose a subject from 1 to 6, default is 0 (all subjects)")

# Time options: select from 20 to 460 samples from EEG data
parser.add_argument('-tl', '--time_low', default=20, type=float, help="lowest time value")
parser.add_argument('-th', '--time_high', default=460, type=float, help="highest time value")

# Model type/options
parser.add_argument('-mt', '--model_type', default='transformer2',
                    help='specify which generator should be used: lstm|EEGChannelNet')
# It is possible to test out multiple deep classifiers:
# - lstm is the model described in the paper "Deep Learning Human Mind for Automated Visual Classification”, in CVPR 2017
# - model10 is the model described in the paper "Decoding brain representations by multimodal learning of neural activity and visual features", TPAMI 2020
parser.add_argument('-mp', '--model_params', default=['num_heads=4', 'num_layers=1', 'd_ff=512', 'd_model=128', 'dropout=0.4'], nargs='*', help='list of key=value pairs of model options')
#parser.add_argument('--pretrained_net', default='', help="path to pre-trained net (to continue training)")
parser.add_argument('--pretrained_net', default='', help="path to pre-trained net (to continue training)")

# Training options
parser.add_argument("-b", "--batch_size", default=128, type=int, help="batch size")
parser.add_argument('-o', '--optim', default="Adam", help="optimizer")
parser.add_argument('-lr', '--learning-rate', default=0.001, type=float, help="learning rate")
parser.add_argument('-lrdb', '--learning-rate-decay-by', default=0.95, type=float, help="learning rate decay factor")
parser.add_argument('-lrde', '--learning-rate-decay-every', default=10, type=int, help="learning rate decay period")
parser.add_argument('-dw', '--data-workers', default=4, type=int, help="data loading workers")
parser.add_argument('-e', '--epochs', default=200, type=int, help="training epochs")
#parser.add_argument('-do', '--dropout', default=0.2, type=float, help="dropout probability (overwrites model default)")
# Save options
parser.add_argument('-sc', '--saveCheck', default=200, type=int, help="learning rate")
# Backend options
parser.add_argument('--no-cuda', default=False, help="disable CUDA", action="store_true")

# cargar objeto ITSA guardado
parser.add_argument('--pretrained_itsa', default='', help="path to pre-trained itsa")

# Parse arguments
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
import torch;

torch.utils.backcompat.broadcast_warning.enabled = True
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import torch.backends.cudnn as cudnn;

import torch.serialization

cudnn.benchmark = True
from scipy.fftpack import fft, rfft, fftfreq, irfft, ifft, rfftfreq
from scipy import signal
import numpy as np
import models
import importlib

from ITSA import ITSAIntegrator

# Dataset class
class EEGDataset:

    # Constructor
    def __init__(self, eeg_signals_path):
        # Load EEG signals
        loaded = torch.load(eeg_signals_path)
        if opt.subject != 0:
            self.data = [loaded['dataset'][i] for i in range(len(loaded['dataset'])) if
                         loaded['dataset'][i]['subject'] == opt.subject]
        else:
            self.data = loaded['dataset']
        self.labels = loaded["labels"]
        self.images = loaded["images"]

        # Compute size
        self.size = len(self.data)

    # Get size
    def __len__(self):
        return self.size

    # Get item
    def __getitem__(self, i):
        # Process EEG
        eeg = self.data[i]["eeg"].float().t()
        eeg = eeg[opt.time_low:opt.time_high, :]

        if opt.model_type == "EEGChannelNet":
            eeg = eeg.t()
            eeg = eeg.view(1, 128, opt.time_high - opt.time_low)
        # Get label
        label = self.data[i]["label"]
        # Return
        return eeg, label


# Splitter class
class Splitter:

    def __init__(self, dataset, split_path, split_num=0, split_name="train"):
        # Set EEG dataset
        self.dataset = dataset
        # Load split
        loaded = torch.load(split_path)
        self.split_idx = loaded["splits"][split_num][split_name]
        # Filter data
        self.split_idx = [i for i in self.split_idx if 450 <= self.dataset.data[i]["eeg"].size(1) <= 600]
        # Compute size
        self.size = len(self.split_idx)

    # Get size
    def __len__(self):
        return self.size

    # Get item
    def __getitem__(self, i):
        # Get sample from dataset
        eeg, label = self.dataset[self.split_idx[i]]
        subj = self.dataset.data[self.split_idx[i]]["subject"]
        # Return
        return eeg, label, subj

# Load dataset
dataset = EEGDataset(opt.eeg_dataset)
# Create loaders
loaders = {split: DataLoader(Splitter(dataset, split_path=opt.splits_path, split_num=opt.split_num, split_name=split),
                             batch_size=opt.batch_size, drop_last=True, shuffle=True) for split in
           ["train", "val", "test"]}

if opt.pretrained_itsa != '':
    itsa = torch.load(opt.pretrained_itsa)
    itsa.adapt_from_dataset(dataset, splits_path=opt.splits_path, split_num=opt.split_num)
else:
    itsa = ITSAIntegrator.from_dataset(dataset, splits_path=opt.splits_path, split_num=opt.split_num)

# Load model
model_options = {key: int(value) if value.isdigit() else (float(value) if value[0].isdigit() else value) for
                 (key, value) in [x.split("=") for x in opt.model_params]}
# Create discriminator model/optimizer
module = importlib.import_module("models." + opt.model_type)
model = module.Model(**model_options)


optimizer = getattr(torch.optim, opt.optim)(model.parameters(), lr=opt.learning_rate)
# Learning rate scheduler (equivalente al learning_rate_decay_by para Adam)
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=opt.learning_rate_decay_every,
    gamma=opt.learning_rate_decay_by
)
# Setup CUDA
if not opt.no_cuda:
    model.cuda()
    print("Copied to CUDA")

if opt.pretrained_net != '':
    model = torch.load(opt.pretrained_net, weights_only=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=opt.learning_rate)  # nuevo LR
    # Learning rate scheduler (equivalente al learning_rate_decay_by para Adam)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=opt.learning_rate_decay_every,
        gamma=opt.learning_rate_decay_by
    )


    # probar LR 0.005
    # disminuir LR_decay_by

    
    #for name, param in model.named_parameters():
     #   print(f"Parameter name: {name}, Trainable: {param.requires_grad}")
    
    print(model)

# initialize training,validation, test losses and accuracy list
losses_per_epoch = {"train": [], "val": [], "test": []}
accuracies_per_epoch = {"train": [], "val": [], "test": []}

best_accuracy = 0
best_accuracy_val = 0
best_epoch = 0
# Start training

predicted_labels = []
correct_labels = []

for epoch in range(1, opt.epochs + 1):
    # Initialize loss/accuracy variables
    losses = {"train": 0, "val": 0, "test": 0}
    accuracies = {"train": 0, "val": 0, "test": 0}
    counts = {"train": 0, "val": 0, "test": 0}
    # Adjust learning rate for SGD
    if opt.optim == "SGD":
        lr = opt.learning_rate * (opt.learning_rate_decay_by ** (epoch // opt.learning_rate_decay_every))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    # Process each split
    for split in ("train", "val", "test"):
        # Set network mode
        if split == "train":
            model.train()
            torch.set_grad_enabled(True)
        else:
            model.eval()
            torch.set_grad_enabled(False)
        # Process all split batches
        for i, (input, target, batch_subjects) in enumerate(loaders[split]):
            # Check CUDA
            if not opt.no_cuda:
                input = input.to("cuda") 
                target = target.to("cuda")
                batch_subjects = batch_subjects.to("cuda")
            # Forward
            input = itsa.transform_bath(input, batch_subjects)
            output = model(input)

            # Compute loss
            loss = F.cross_entropy(output, target)
            losses[split] += loss.item()
            # Compute accuracy
            _, pred = output.data.max(1)
            correct = pred.eq(target.data).sum().item()
            accuracy = correct / input.data.size(0)
            accuracies[split] += accuracy
            counts[split] += 1
            # Backward and optimize
            if split == "train":
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    # Print info at the end of the epoch
    if accuracies["val"] / counts["val"] >= best_accuracy_val:
        best_accuracy_val = accuracies["val"] / counts["val"]
        best_accuracy = accuracies["test"] / counts["test"]
        best_epoch = epoch

    TrL, TrA, VL, VA, TeL, TeA = losses["train"] / counts["train"], accuracies["train"] / counts["train"], losses[
        "val"] / counts["val"], accuracies["val"] / counts["val"], losses["test"] / counts["test"], accuracies["test"] / \
                                 counts["test"]
    print(
        "Model: {11} - Subject {12} - Time interval: [{9}-{10}]  [{9}-{10} Hz] - Epoch {0}: TrL={1:.4f}, TrA={2:.4f}, VL={3:.4f}, VA={4:.4f}, TeL={5:.4f}, TeA={6:.4f}, TeA at max VA = {7:.4f} at epoch {8:d}".format(
            epoch,
            losses["train"] / counts["train"],
            accuracies["train"] / counts["train"],
            losses["val"] / counts["val"],
            accuracies["val"] / counts["val"],
            losses["test"] / counts["test"],
            accuracies["test"] / counts["test"],
            best_accuracy, best_epoch, opt.time_low, opt.time_high, opt.model_type, opt.subject))

    losses_per_epoch['train'].append(TrL)
    losses_per_epoch['val'].append(VL)
    losses_per_epoch['test'].append(TeL)
    accuracies_per_epoch['train'].append(TrA)
    accuracies_per_epoch['val'].append(VA)
    accuracies_per_epoch['test'].append(TeA)

     # Update learning rate after each epoch
    scheduler.step()

    if epoch % opt.saveCheck == 0:
        torch.save(model, '%s__subject%d_epoch_%d.pth' % (opt.model_type, opt.subject, epoch))

        torch.save(itsa, 'itsa_pretrained_space.pth')
        print(f"Modelo y espacio ITSA guardados en la época {epoch}.")
