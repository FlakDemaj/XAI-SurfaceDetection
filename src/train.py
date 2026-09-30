import time
from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm

from model import DefectCNN
from dataset import create_dataloaders

# Configuration

NUM_EPOCHS = 30
LEARNING_RATE = 0.001

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT_PATH = CHECKPOINT_DIR / "best_model.pth"

# Training

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device
):
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        logits = model(images)

        loss = criterion(logits, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item() * images.size(0)

        predictions = logits.argmax(dim=1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_accuracy = correct / total

    return epoch_loss, epoch_accuracy

def validate(
        model,
        loader,
        criterion,
        device
):
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)

            loss = criterion(logits, labels)

            running_loss += loss.item() * images.size(0)

            predictions = logits.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

        epoch_loss = running_loss / total
        epoch_accuracy = correct / total

        return epoch_loss, epoch_accuracy

# Main

def main():

    start = time.time()

    print("Starting training...")

    # data
    train_loader, val_loader, test_loader, class_names = create_dataloaders()

    print("Classes:")
    print(class_names)

    print()
    print("Train images:", len(train_loader.dataset))
    print("Validation images:", len(val_loader.dataset))
    print("Test images:", len(test_loader.dataset))

    # device
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print()
    print("Device:", device)

    # model
    model = DefectCNN(
        num_classes=len(class_names),
    ).to(device)

    # Loss
    criterion = nn.CrossEntropyLoss()

    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4)

    best_val_loss = float("inf")

    # Training Loop
    for epoch in tqdm(range(NUM_EPOCHS)):
        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device
        )

        val_loss, val_accuracy = validate(
            model,
            val_loader,
            criterion,
            device
        )
        print()
        print(
            f"Epoch {epoch + 1:02d}/{NUM_EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_accuracy:.2%} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_accuracy:.2%}"
        )

        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss

            print(
                f"Epoch {epoch + 1:02d}/{NUM_EPOCHS} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Train Acc: {train_accuracy:.2%} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Acc: {val_accuracy:.2%}"
            )

            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_accuracy": val_accuracy,
                    "class_names": class_names
                },
                CHECKPOINT_PATH
            )

    end = time.time() - start
    print(f"Training complete in {end} seconds.")

if __name__ == "__main__":
    main()