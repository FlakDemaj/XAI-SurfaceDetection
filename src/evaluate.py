from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from dataset import create_dataloaders
from model import DefectCNN

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "checkpoints"
    / "best_model.pth"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    # Data

    _, _, test_loader, class_names = (
        create_dataloaders())

    print("Classes:")
    print(class_names)

    # device

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu")

    print("Device:", device)

    #model

    model = DefectCNN(
        num_classes=len(class_names),
    ).to(device)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location = device)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    # prediction

    all_labels = []
    all_predictions = []
    all_probabilities = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)

            probabilities = torch.softmax(
                logits,
                dim=1
            )

            predictions = probabilities.argmax(
                dim=1
            )

            all_labels.extend(
                labels.cpu().numpy()
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_probabilities.extend(
                probabilities.cpu().numpy()
            )

    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)
    all_probabilities = np.array(all_probabilities)

    # Accuracy

    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    print()
    print(
        f"Test Accuracy: {accuracy:.2%}"
    )

    # Classification Report

    print()
    print("Classification Report:")

    print(
        classification_report(
            all_labels,
            all_predictions,
            target_names=class_names,
            digits=4
        )
    )

    # Confusion Matrix

    cm = confusion_matrix(
        all_labels,
        all_predictions
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names
    )

    fig, ax = plt.subplots(
        figsize=(9, 8)
    )

    display.plot(
        ax=ax,
        cmap="Blues",
        xticks_rotation=45,
        colorbar=False
    )

    plt.tight_layout()

    output_path = (
            OUTPUT_DIR
            / "confusion_matrix.png"
    )

    plt.savefig(
        output_path,
        dpi=200
    )

    plt.show()

    print(
        f"Confusion matrix saved to: "
        f"{output_path}"
    )

if __name__ == "__main__":
    main()