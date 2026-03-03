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
        "left_out":    3,
        "subjects":    [1, 2, 4, 5, 6],
        "splits_path": r"data\splits_preTraining_subject23456.pth",
        "output_file": "itsa_pretrained_excl_subject3.pth",
    },
]
r"""
EXPERIMENTS = [
    {
        "left_out":    1,
        "subjects":    [2, 3, 4, 5, 6],
        "splits_path": r"data\splits_preTraining_subject1.pth",
        "output_file": "itsa_pretrained_excl_subject1.pth",
    },
    {
        "left_out":    2,
        "subjects":    [1, 3, 4, 5, 6],
        "splits_path": r"data\splits_preTraining_subject23456.pth",
        "output_file": "itsa_pretrained_excl_subject2.pth",
    },
    {
        "left_out":    3,
        "subjects":    [1, 2, 4, 5, 6],
        "splits_path": r"data\splits_preTraining_subject23456.pth",
        "output_file": "itsa_pretrained_excl_subject3.pth",
    },
    {
        "left_out":    4,
        "subjects":    [1, 2, 3, 5, 6],
        "splits_path": r"data\splits_preTraining_subject23456.pth",
        "output_file": "itsa_pretrained_excl_subject4.pth",
    },
]
"""
# ============================================================
#  Dataset class — same as your main script but loads ALL subjects
# ============================================================

class EEGDataset:
    def __init__(self, eeg_signals_path, subjects_to_keep):
        loaded = torch.load(eeg_signals_path, map_location="cpu")
        # Only keep the subjects we want for this experiment
        self.data = [
            loaded['dataset'][i]
            for i in range(len(loaded['dataset']))
            if loaded['dataset'][i]['subject'] in subjects_to_keep
        ]
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
    left_out = exp["left_out"]
    subjects = exp["subjects"]
    splits_path = exp["splits_path"]
    output_file = exp["output_file"]

    print(f"\n{'='*60}")
    print(f"Experiment {exp_num}/{len(EXPERIMENTS)}")
    print(f"  Training subjects : {subjects}")
    print(f"  Left-out subject  : {left_out}")
    print(f"  Output file       : {output_file}")
    print(f"{'='*60}")

    # Build dataset filtered to only the training subjects
    dataset = EEGDataset(EEG_DATASET_PATH, subjects_to_keep=subjects)
    print(f"  Dataset size: {len(dataset)} segments")

    # Fit ITSA
    print("  Computing ITSA space (this may take 30-90 min on CPU)...")
    itsa = ITSAIntegrator.from_dataset(
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
