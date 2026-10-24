
"""
train.py

Task 3 and 4: Model training.

Trains a pretrained CNN, using EfficientNet-B0 by default, to classify
images into eight facial feature classes. Training and validation loss
and accuracy are recorded after each epoch. The model with the lowest
validation loss is saved, and early stopping is used if the validation
loss stops improving.

"""

import argparse
import random
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn

sys.path.append(str(Path(__file__).resolve().parent))
from config import CLASSES, FIG_DIR, OUTPUT_DIR, SEED  
from dataset import get_loaders  

MODEL_DIR = OUTPUT_DIR / "models"


def set_seed(seed: int):
    """Set random seeds to make training results more consistent."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(name: str, pretrained: bool = True):
    """Load a model and set its output layer for the eight classes."""
    return timm.create_model(name, pretrained=pretrained, num_classes=len(CLASSES))


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None):
    """Process one epoch of data for training or validation."""
    training = optimizer is not None
    model.train(training)
    total_loss, correct, count = 0.0, 0, 0

    # Use mixed precision on the GPU to help speed up training.
    use_amp = device.type == "cuda"

    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(images)
                loss = criterion(outputs, labels)

            if training:
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            total_loss += loss.item() * labels.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            count += labels.size(0)

    return total_loss / count, correct / count


def plot_curves(history: pd.DataFrame, tag: str):
    """Plot and save the loss and accuracy for training and validation."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(history.epoch, history.train_loss, label="Train")
    ax1.plot(history.epoch, history.val_loss, label="Validation")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss")
    ax1.legend()

    ax2.plot(history.epoch, history.train_acc, label="Train")
    ax2.plot(history.epoch, history.val_acc, label="Validation")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Accuracy")
    ax2.legend()

    fig.tight_layout()
    fig.savefig(FIG_DIR / f"learning_curves_{tag}.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="efficientnet_b0", help="any timm model name")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=4,
                        help="epochs to wait for validation loss to improve before stopping")
    parser.add_argument("--no-pretrained", action="store_true",
                        help="train from random weights instead of ImageNet weights")
    args = parser.parse_args()

    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, train_df, val_df = get_loaders(args.batch_size)
    print(f"Training images: {len(train_df)}, validation images: {len(val_df)}")

    model = build_model(args.model, pretrained=not args.no_pretrained).to(device)

    # Cross-entropy calculates the loss for the eight-class classification task.
    # Label smoothing helps prevent the model from becoming too confident.
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # AdamW updates the model weights and uses weight decay to help reduce overfitting.
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Gradually reduce the learning rate as training progresses.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Scale the loss when using mixed precision to help prevent numerical underflow.
    # This is disabled when training on the CPU.
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    history = []
    model_path = MODEL_DIR / f"best_{args.model}.pt"

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_loss, train_acc = run_epoch(model, train_loader, criterion, device, optimizer, scaler)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, device)
        scheduler.step()

        history.append({
            "epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_loss, "val_acc": val_acc,
            "lr": optimizer.param_groups[0]["lr"],
        })
        print(f"Epoch {epoch:2d}/{args.epochs} | "
              f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
              f"val loss {val_loss:.4f} acc {val_acc:.4f} | {time.time() - start:.0f}s")

        # Save the model whenever the validation loss reaches a new lowest value.
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save({"model_name": args.model, "classes": CLASSES,
                        "state_dict": model.state_dict()}, model_path)
            print("  -> saved new best model")
        else:
            epochs_without_improvement += 1

        # Stop training if the validation loss does not improve for the set number of epochs.
        if epochs_without_improvement >= args.patience:
            print(f"Early stopping: no improvement for {args.patience} epochs.")
            break

    history = pd.DataFrame(history)
    history.to_csv(OUTPUT_DIR / f"history_{args.model}.csv", index=False)
    plot_curves(history, args.model)

    best = history.loc[history.val_loss.idxmin()]
    print(f"\nBest epoch: {int(best.epoch)} | val loss {best.val_loss:.4f} | val acc {best.val_acc:.4f}")
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    main()
