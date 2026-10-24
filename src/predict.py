
"""
predict.py

Task 5: Kaggle test predictions.

Loads the trained model and predicts the class of each test image.
The predictions are saved to outputs/submission.csv.

"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import timm
import torch

sys.path.append(str(Path(__file__).resolve().parent))
from config import DATA_DIR, OUTPUT_DIR  
from dataset import get_test_loader  


def find_sample_submission():
    """Find the Kaggle sample submission file inside data/."""
    matches = sorted(DATA_DIR.rglob("sample_submission*.csv"))
    return matches[0] if matches else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="efficientnet_b0")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(OUTPUT_DIR / "models" / f"best_{args.model}.pt", map_location=device)
    classes = checkpoint["classes"]

    # Create the same model structure and load the saved weights.
    model = timm.create_model(checkpoint["model_name"], pretrained=False, num_classes=len(classes))
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()

    test_loader = get_test_loader(args.batch_size)
    print(f"Predicting {len(test_loader.dataset)} test images on {device}")

    names, preds = [], []
    with torch.no_grad():
        for images, filenames in test_loader:
            outputs = model(images.to(device))
            preds.extend(outputs.argmax(dim=1).cpu().tolist())
            names.extend(filenames)

    # Convert the image filenames to numbers so they match the IDs as sample.
    pred_by_id = {int(Path(n).stem): classes[p] for n, p in zip(names, preds) if Path(n).stem.isdigit()}

    sample_path = find_sample_submission()
    if sample_path is not None:
        # Use the same columns and row order as the sample.
        sample = pd.read_csv(sample_path)
        id_col, label_col = sample.columns[0], sample.columns[1]
        sample[label_col] = sample[id_col].astype(int).map(pred_by_id)

        missing = sample[label_col].isna().sum()
        if missing:
            print(f"WARNING: {missing} ids in the sample submission had no matching test image")
        submission = sample
        print(f"Using format from {sample_path.name}: columns {id_col}, {label_col}")
    else:
        submission = pd.DataFrame({"image_id": list(pred_by_id), "class": list(pred_by_id.values())})
        print("No sample submission found, writing columns image_id,class")

    out_path = OUTPUT_DIR / "submission.csv"
    submission.to_csv(out_path, index=False)
    print(f"Saved {len(submission)} predictions to {out_path}")
    print("\nPredicted class counts:")
    print(submission.iloc[:, 1].value_counts().to_string())


if __name__ == "__main__":
    main()
