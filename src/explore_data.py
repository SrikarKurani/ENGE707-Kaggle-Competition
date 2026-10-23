"""
explore_data.py

Task 1: Dataset exploration.

Checks the training dataset before model training begins. This includes
checking the number of images in each class, image sizes, colour modes,
corrupted files and duplicate images. It also creates plots in
outputs/figures/ that can be used in the report.

"""
import hashlib
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

# Add src/ to the Python path so config.py can be imported when
# the script is run from the main project folder.
sys.path.append(str(Path(__file__).resolve().parent))
from config import CLASSES, DATA_DIR, FIG_DIR, IMG_EXTS, OUTPUT_DIR, SEED, TEST_DIR, TRAIN_DIR  
# The Kaggle data stores the class of each training image in this CSV file.
LABELS_CSV = DATA_DIR / "train_labels.csv"


def list_images(folder: Path):
    """Find all image files in a folder and its subfolders, sorted by path."""
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMG_EXTS)


def build_index() -> pd.DataFrame:
    """Create a table containing the path and label for each training image."""
    labels = pd.read_csv(LABELS_CSV, dtype=str)

    # Check whether the csv is missing its header row. 
    if labels.columns[1].strip() in CLASSES:
        labels = pd.read_csv(LABELS_CSV, dtype=str, header=None)
    labels.columns = ["image_id", "label"]
    labels["label"] = labels.label.str.strip()

    # Convert image IDs to numbers so they can match filenames with leading zeros.
    files = {int(p.stem): str(p) for p in list_images(TRAIN_DIR) if p.stem.isdigit()}
    labels["path"] = labels.image_id.astype(int).map(files)

    # Warning for any labelled images that cannot be found in the training folder.
    missing = int(labels.path.isna().sum())
    if missing:
        print(f"WARNING: {missing} labelled ids have no matching image file")
    # Keep only rows with a matching image, and return the required columns.
    return labels.dropna(subset=["path"])[["path", "label"]].reset_index(drop=True)


def inspect_image(path: str) -> dict:
    """Collect image details and check whether the file can be opened."""
    info = {"width": None, "height": None, "mode": None, "corrupt": False, "md5": None}
    try:
        # Create a hash from the file contents to identify exact duplicates.
        with open(path, "rb") as f:
            info["md5"] = hashlib.md5(f.read()).hexdigest()

        # Check for damaged or incomplete image files.
        # The image must be opened again after verify() has been called.
        with Image.open(path) as im:
            im.verify()

        with Image.open(path) as im:
            info["width"], info["height"] = im.size
            info["mode"] = im.mode
    except Exception:
        info["corrupt"] = True

    return info


def plot_class_distribution(counts: pd.Series):
    """Create a bar chart showing the number of training images per class."""
    fig, ax = plt.subplots(figsize=(9, 4))
    counts.reindex(CLASSES).plot.bar(ax=ax, color="steelblue")
    ax.set_ylabel("Number of images")
    ax.set_title("Training images per class")

    # Display the image count above each bar.
    for i, v in enumerate(counts.reindex(CLASSES).fillna(0)):
        ax.text(i, v, str(int(v)), ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "class_distribution.png", dpi=150)
    plt.close(fig)


def plot_sample_grid(df: pd.DataFrame, per_class: int = 5):
    """Display a grid of random example images, with one row per class."""
    rng = random.Random(SEED)  # Keep the selected samples consistent between runs.
    fig, axes = plt.subplots(len(CLASSES), per_class, figsize=(per_class * 2, len(CLASSES) * 2))

    for r, cls in enumerate(CLASSES):
        paths = df.loc[(df.label == cls) & (~df.corrupt), "path"].tolist()
        picks = rng.sample(paths, min(per_class, len(paths)))

        for c in range(per_class):
            ax = axes[r, c]
            ax.axis("off")

            if c < len(picks):
                with Image.open(picks[c]) as im:
                    ax.imshow(im.convert("RGB"))

            if c == 0:
                ax.set_title(cls, fontsize=9, loc="left")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "sample_grid.png", dpi=120)
    plt.close(fig)


def plot_sizes(df: pd.DataFrame):
    """Plot image widths against heights to check the variation in image sizes."""
    ok = df[~df.corrupt]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(ok.width, ok.height, s=5, alpha=0.4)
    ax.set_xlabel("Width (px)")
    ax.set_ylabel("Height (px)")
    ax.set_title("Image size distribution (train)")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "size_distribution.png", dpi=150)
    plt.close(fig)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    if not TRAIN_DIR.exists():
        sys.exit(f"Training folder not found: {TRAIN_DIR}. Download the Kaggle data into data/ first.")

    df = build_index()
    print(f"Training images found: {len(df)}")
    print(f"Test images found:     {len(list_images(TEST_DIR)) if TEST_DIR.exists() else 0}")

    # Check whether all labels match the expected class names.
    unknown = set(df.label) - set(CLASSES)
    if unknown:
        print(f"WARNING: labels not in CLASSES list: {sorted(unknown)}")

    # Check each image and save the collected information to a CSV file.
    meta = pd.DataFrame([inspect_image(p) for p in df.path])
    df = pd.concat([df.reset_index(drop=True), meta], axis=1)
    df.to_csv(OUTPUT_DIR / "data_report.csv", index=False)

    # Check the number of images in each class.
    counts = df.label.value_counts()
    print("\nImages per class:")
    print(counts.reindex(CLASSES).fillna(0).astype(int).to_string())

    ratio = counts.max() / max(counts.min(), 1)
    print(f"Imbalance ratio (largest / smallest class): {ratio:.2f}")

    # Summarise image quality and format information.
    print(f"\nCorrupt / unreadable files: {int(df.corrupt.sum())}")
    print("Colour modes:", df["mode"].value_counts(dropna=False).to_dict())

    ok = df[~df.corrupt]
    print(f"Width  min/median/max: {ok.width.min()} / {ok.width.median():.0f} / {ok.width.max()}")
    print(f"Height min/median/max: {ok.height.min()} / {ok.height.median():.0f} / {ok.height.max()}")

    # Files with matching hashes are exact duplicates.
    # If duplicates have different labels, this may indicate a labelling error.
    dup = df[df.duplicated("md5", keep=False) & df.md5.notna()]
    print(f"Exact duplicate files: {len(dup)} images in {dup.md5.nunique()} groups")

    cross = dup.groupby("md5").label.nunique()
    print(f"Duplicate groups with conflicting labels: {int((cross > 1).sum())}")

    # Generate the plots for the report.
    plot_class_distribution(counts)
    plot_sample_grid(df)
    plot_sizes(df)

    print(f"\nFigures saved to {FIG_DIR}")


if __name__ == "__main__":
    main()