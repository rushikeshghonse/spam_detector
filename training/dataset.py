# training/dataset.py

import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from typing import Tuple

class EmailDataset(Dataset):
    """
    PyTorch Dataset wrapper around our numpy feature arrays.

    WHY DO WE NEED THIS?
    PyTorch's DataLoader needs a Dataset object to:
    - Know how many samples exist   → __len__
    - Fetch one sample by index     → __getitem__
    - Batch samples automatically   → DataLoader handles this
    """

    def __init__(self, features: np.ndarray, labels: np.ndarray):
        assert len(features) == len(labels), \
            f"Features ({len(features)}) and labels ({len(labels)}) must match"

        # Convert numpy → float32 tensors
        # float32 is standard for neural network inputs
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels   = torch.tensor(labels,   dtype=torch.float32)

        print(f"  Dataset created: {len(self.features):,} samples, "
              f"{self.features.shape[1]:,} features")

    def __len__(self) -> int:
        """Total number of emails in this dataset."""
        return len(self.features)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (feature_vector, label) for one email."""
        return self.features[idx], self.labels[idx]

    @property
    def input_size(self) -> int:
        return self.features.shape[1]


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from torch.utils.data import DataLoader
    from preprocessing.pipeline import PreprocessingPipeline

    print("="*50)
    print("Testing EmailDataset + DataLoader")
    print("="*50)

    # Load small sample
    train_df = pd.read_csv("data/processed/train.csv").head(200)

    # Build pipeline
    pipeline = PreprocessingPipeline(max_features=500)
    X = pipeline.fit_transform(train_df['text'].tolist())
    y = train_df['label'].values

    # Create dataset
    print("\nCreating dataset...")
    dataset = EmailDataset(X, y)

    print(f"  Total samples : {len(dataset)}")
    print(f"  Input size    : {dataset.input_size}")

    # Test single item
    feat, label = dataset[0]
    print(f"  Sample feature shape : {feat.shape}")
    print(f"  Sample label         : {label.item()} "
          f"({'spam' if label.item()==1 else 'ham'})")

    # Test DataLoader
    print("\nTesting DataLoader (batch_size=32)...")
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    batch_features, batch_labels = next(iter(loader))
    print(f"  Batch features shape : {batch_features.shape}")
    print(f"  Batch labels shape   : {batch_labels.shape}")
    print(f"  Spam in batch        : {batch_labels.sum().int().item()}")
    print(f"  Ham in batch         : {(batch_labels==0).sum().int().item()}")

    print("\n✅ Dataset and DataLoader working correctly!")