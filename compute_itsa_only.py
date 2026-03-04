import torch
from LOSO_experiments.ITSA import ITSAIntegrator

# ============================================================
#  CONFIGURATION — edit these paths and experiments
# ============================================================

EEG_DATASET_PATH = r"C:\Users\maria\Documents\Beca\Transformer-22Abril\data\eeg_55_95_std.pth"
TIME_LOW  = 20
TIME_HIGH = 460

# Each experiment: (left_out_subject, splits_file)
# The splits file must only contain the subjects you want to train on.
# If you have a single "all subjects" splits file, see the note below.
EXPERIMENTS = [
    {
        "splits_path": r"C:\Users\maria\Documents\Beca\Splits-LOSO\block_splits_LOSO_subject1.pth",
        "output_file": "itsa_loso_subject1.pth",
    },
]

# ============================================================
#  Dataset class — same as your main script but loads ALL subjects
# ============================================================

class EEGDataset:
    def __init__(self, eeg_signals_path):
        loaded = torch.load(eeg_signals_path, map_location="cpu")
        # Only keep the subjects we want for this experiment
        self.data = loaded['dataset']
        self.labels = loaded["labels"]
        self.images = loaded["images"]
        self.size   = len(self.data)

    def __len__(self):
        return self.size

    def __getitem__(self, i):
        eeg = self.data[i]["eeg"].float().t()
        eeg = eeg[TIME_LOW:TIME_HIGH, :]
        return eeg, self.data[i]["label"]


# ============================================================
#  Run all experiments
# ============================================================

print("Loading EEG dataset file once...")
raw = torch.load(EEG_DATASET_PATH, map_location="cpu")

for exp_num, exp in enumerate(EXPERIMENTS, start=1):
    splits_path = exp["splits_path"]
    output_file = exp["output_file"]

    print(f"\n{'='*60}")
    print(f"Experiment {exp_num}/{len(EXPERIMENTS)}")
    print(f"  Output file       : {output_file}")
    print(f"{'='*60}")

    # Build dataset filtered to only the training subjects
    dataset = EEGDataset(EEG_DATASET_PATH)
    print(f"  Dataset size: {len(dataset)} segments")

    # Fit ITSA
    print("  Computing ITSA space (this may take 30-90 min on CPU)...")
    itsa = ITSAIntegrator.from_dataset_loso(
        dataset,
        splits_path=splits_path,
        split_num=0
    )

    # Save lightweight version
    print(f" Saving to {output_file}...")
    itsa_lite = itsa._itsa.export_light()
    torch.save(itsa_lite, output_file)
    print(f" Done! Saved {output_file}")

print(f"\n{'='*60}")
print("All experiments completed!")
print(f"{'='*60}")
