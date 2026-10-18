"""
config.py

Stores the shared settings used throughout the project, including file paths,
class names and the random seed. Keeping these in one file means the same
values can be used across different scripts without repeating them.
"""
from pathlib import Path

# Define the main project folders
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"
OUTPUT_DIR = ROOT / "outputs"
FIG_DIR = OUTPUT_DIR / "figures"

# The 8 target classes and their corresponding labels.
# The order of this list determines the index assigned to each class.
CLASSES = [
    "CatEye", "DogEye", "CatFace", "DogFace",
    "CatMouth", "DogMouth", "CatNose", "DogNose",
]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

# Use a fixed seed to help produce consistent results between runs.
SEED = 42

# Image file formats to look for when scanning the data folders.
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Initial settings for model training. These can be adjusted
# later based on the results of model experiments.
IMG_SIZE = 224        # Input size commonly used by pretrained CNNs
BATCH_SIZE = 32
VAL_FRACTION = 0.2    # Reserve 20% of the training data for validation