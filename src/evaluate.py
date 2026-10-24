
"""
evaluate.py

Task 5 and 6: Model evaluation.

Tests the saved model using the validation set and reports its accuracy,
loss, precision, recall and F1-score for each class. It also saves a
confusion matrix and shows which classes the model confuses most often.
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix

sys.path.append(str(Path(__file__).resolve().parent))
from config import FIG_DIR, OUTPUT_DIR
from dataset import get_loaders


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="efficientnet_b0")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(OUTPUT_DIR / "models" / f"best_{args.model}.pt", map_location=device)
    classes = checkpoint["classes"]

    model = timm.create_model(checkpoint["model_name"], pretrained=False, num_classes=len(classes))
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()

    # Use the same data split as training to evaluate images the model has not seen.
    _, val_loader, _, val_df = get_loaders(args.batch_size)
    print(f"Evaluating on {len(val_df)} validation images")

    criterion = nn.CrossEntropyLoss()
    total_loss, y_true, y_pred = 0.0, [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            total_loss += criterion(outputs, labels).item() * labels.size(0)
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(outputs.argmax(dim=1).cpu().tolist())

    y_true, y_pred = np.array(y_true), np.array(y_pred)
    accuracy = (y_true == y_pred).mean()
    print(f"\nValidation accuracy: {accuracy:.4f}")
    print(f"Validation loss:     {total_loss / len(y_true):.4f}")

    # Generate precision, recall and F1-scores for each class.
    report = classification_report(y_true, y_pred, labels=range(len(classes)),
                                   target_names=classes, digits=4, zero_division=0)
    print("\n" + report)
    report_df = pd.DataFrame(classification_report(
        y_true, y_pred, labels=range(len(classes)), target_names=classes,
        output_dict=True, zero_division=0)).T
    report_df.to_csv(OUTPUT_DIR / f"classification_report_{args.model}.csv")

    # Show the actual classes against the classes predicted by the model.
    cm = confusion_matrix(y_true, y_pred, labels=range(len(classes)))
    fig, ax = plt.subplots(figsize=(8, 7))
    ConfusionMatrixDisplay(cm, display_labels=classes).plot(ax=ax, xticks_rotation=45,
                                                            cmap="Blues", colorbar=False)
    ax.set_title(f"Validation confusion matrix ({args.model})")
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"confusion_matrix_{args.model}.png", dpi=150)
    plt.close(fig)

    # Find the most common incorrect predictions, excluding correct ones.
    mistakes = [(classes[t], classes[p], cm[t, p])
                for t in range(len(classes)) for p in range(len(classes))
                if t != p and cm[t, p] > 0]
    mistakes.sort(key=lambda m: m[2], reverse=True)
    print("Most confused pairs (true -> predicted):")
    for true_cls, pred_cls, n in mistakes[:8]:
        print(f"  {true_cls:>9} -> {pred_cls:<9} {n}")

    print(f"\nSaved confusion matrix and report to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
