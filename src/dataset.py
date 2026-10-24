
"""
dataset.py

Task 2: Data preprocessing and pipeline preparation.
Prepares the image datasets and data loaders for training, validation and
making predictions on the Kaggle test set. The same resizing and
normalisation are used for validation and test images, while data
augmentation is only applied during training.
"""

import os
import sys
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

sys.path.append(str(Path(__file__).resolve().parent))
from config import (  # noqa: E402
    BATCH_SIZE, CLASS_TO_IDX, IMG_EXTS, IMG_SIZE, SEED, TEST_DIR, VAL_FRACTION,
)
from explore_data import build_index  # noqa: E402

# ImageNet mean and standard deviation are used because the pretrained
# models expect images to be normalised using these values.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Use the main process to avoid multiprocessing issues.
NUM_WORKERS = 0 if os.name == "nt" else min(4, os.cpu_count() or 1)


def get_transforms(train: bool):
    """Set up image transformations for training or evaluation."""
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def list_images(folder: Path):
    """Find all supported image files in a folder and its subfolders."""
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMG_EXTS)


def load_train_index() -> pd.DataFrame:
    """Load the training image paths and convert their class names to labels."""
    # Use the same function as explore_data.py to load the image labels.
    df = build_index()

    # Assign each class a numeric label based on config.py.
    df["target"] = df.label.map(CLASS_TO_IDX)
    return df


def split_train_val(df: pd.DataFrame):
    """Split the dataset into training and validation sets.

    Stratification keeps the class proportions similar in both sets.
    Using a fixed seed also makes the split reproducible.
    """
    train_df, val_df = train_test_split(
        df, test_size=VAL_FRACTION, stratify=df.target, random_state=SEED
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


class ImageDataset(Dataset):
    """Load images and apply the required transformations.

    Returns an image and its label for training and validation.
    For test images, returns the image and its filename instead.
    """

    def __init__(self, paths, targets=None, transform=None):
        self.paths = list(paths)
        self.targets = None if targets is None else list(targets)
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        # Convert every image to RGB so they all have three colour channels.
        with Image.open(self.paths[idx]) as im:
            image = im.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        if self.targets is None:
            return image, Path(self.paths[idx]).name
        return image, self.targets[idx]


def get_loaders(batch_size: int = BATCH_SIZE):
    """Create the training and validation data loaders."""
    df = load_train_index()
    train_df, val_df = split_train_val(df)

    train_ds = ImageDataset(train_df.path, train_df.target, get_transforms(train=True))
    val_ds = ImageDataset(val_df.path, val_df.target, get_transforms(train=False))

    # Use a fixed seed so the training data shuffle can be reproduced.
    generator = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=NUM_WORKERS, generator=generator)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=NUM_WORKERS)
    return train_loader, val_loader, train_df, val_df


def get_test_loader(batch_size: int = BATCH_SIZE):
    """Create a data loader for the Kaggle test images, which have no labels."""
    test_ds = ImageDataset(list_images(TEST_DIR), None, get_transforms(train=False))
    return DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS)


if __name__ == "__main__":
    # Check that the datasets and data loaders are set up correctly.
    train_loader, val_loader, train_df, val_df = get_loaders()
    print(f"Training images:   {len(train_df)}")
    print(f"Validation images: {len(val_df)}")

    split_counts = pd.DataFrame({
        "train": train_df.label.value_counts(),
        "val": val_df.label.value_counts(),
    }).reindex(list(CLASS_TO_IDX))
    print("\nImages per class in each split:")
    print(split_counts.to_string())

    images, labels = next(iter(train_loader))
    print(f"\nOne training batch: images {tuple(images.shape)}, labels {labels[:8].tolist()}")

    if TEST_DIR.exists():
        test_loader = get_test_loader()
        print(f"Test images: {len(test_loader.dataset)}")
