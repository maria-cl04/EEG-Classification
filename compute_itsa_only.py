import argparse
import torch
from ITSA import ITSAIntegrator

# ---- Minimal settings (match exactly what you use on Kaggle) ----
class Opt :
    eeg_dataset = r"C:\Users\maria\Documents\Beca\Transformer-22Abril\data\eeg_55_95_std.pth"       # ← change this
    splits_path = r"data\splits_preTraining_subject23456.pth"  # ← change this
    split_num   = 0
    subject     = 1
    time_low    = 20
    time_high   = 460
    model_type  = "transformer2"

opt = Opt()

# ---- Replicate EEGDataset (same as main script) ----
class EEGDataset:
    def __init__(self, eeg_signals_path):
        loaded = torch.load(eeg_signals_path, map_location="cpu")
        self.data = [loaded['dataset'][i] for i in range(len(loaded['dataset'])) if
                     loaded['dataset'][i]['subject'] != opt.subject]
        self.labels = loaded["labels"]
        self.images = loaded["images"]
        self.size   = len(self.data)

    def __len__(self):
        return self.size

    def __getitem__(self, i):
        eeg = self.data[i]["eeg"].float().t()
        eeg = eeg[opt.time_low:opt.time_high, :]
        return eeg, self.data[i]["label"]

for sub in [2,3,4]:
    opt.subject = sub
    print(f"Current subject: {opt.subject}")
    # ---- Run ITSA fit (this is the slow part, ~30-60 min on CPU) ----
    print("Loading dataset...")
    dataset = EEGDataset(opt.eeg_dataset)

    print("Computing ITSA space (this will take a while on CPU)...")
    itsa = ITSAIntegrator.from_dataset(
        dataset,
        splits_path=opt.splits_path,
        split_num=opt.split_num
    )

    print("Saving lightweight ITSA file...")
    itsa_lite = itsa._itsa.export_light()
    torch.save(itsa_lite, f"itsa_pretrained_space_light_subject{opt.subject}.pth")
    print(f"Done! File saved as itsa_pretrained_space_light_subject{opt.subject}.pth")