import torch
import numpy as np

file_path = r'C:\Users\maria\PycharmProjects\EEG-Classification\SeleccionDeSujetos\baseline_model_all_subjects.pth'

splits = torch.load(file_path, map_location='cpu', weights_only=False)


def summarize_structure(data, indent=0, max_keys=5):
    spacing = "  " * indent

    if isinstance(data, dict):
        print(f"{spacing}Dict (keys={len(data)}):")

        for key, value in data.items():
            print(f"\n{spacing} -- Key: '{key}' --")
            summarize_structure(value, indent + 1)

    elif isinstance(data, (list, tuple)):
        if data and isinstance(data[0], (int, float, np.number)):
            preview = str(data[:5])[:-1] + ", ...]" if len(data) > 5 else str(data)
            # ver todos los splits --> preview = str(data)
            print(f"{spacing}List of indexes (len={len(data)}) -> {preview}")

        else:
            print(f"{spacing}List of components (len={len(data)}):")
            for i, item in enumerate(data):
                summarize_structure(item, indent + 1)

    else:
        print(f"{spacing}{data}")

if __name__ == "__main__":
    summarize_structure(splits)