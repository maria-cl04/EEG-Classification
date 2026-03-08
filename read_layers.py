import torch
loaded = torch.load(r'C:\Users\maria\Documents\Beca\Transformer-22Abril\data\eeg_55_95_std.pth')
print("Label set:", loaded["labels"])
print("Number of classes:", len(loaded["labels"]))

# Check what labels appear in the data
all_labels = [loaded["dataset"][i]["label"] for i in range(len(loaded["dataset"]))]
import numpy as np
print("Unique labels in data:", np.unique(all_labels))
print("Class distribution:", np.bincount(all_labels))