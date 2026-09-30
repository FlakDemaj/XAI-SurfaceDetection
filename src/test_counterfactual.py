from pathlib import Path

import torch
import matplotlib.pyplot as plt

from dataset import create_dataloaders
from model import DefectCNN

from xai.counterfactual import (
    generate_counterfactual_explanation
)

def find_reference_image(
    dataset,
    target_class
):
    """
    Find one image belonging to target_class.
    """

    for i in range(len(dataset)):

        image, label = dataset[i]

        if label == target_class:

            return (
                image.unsqueeze(0),
                label
            )

    raise ValueError(
        f"No image found for class "
        f"{target_class}"
    )

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "checkpoints"
    / "best_model.pth"
)

def find_reference_image(
    dataset,
    target_class
):

    for i in range(len(dataset)):

        image, label = dataset[i]

        if label == target_class:

            return (
                image.unsqueeze(0),
                label
            )

    raise ValueError(
        f"No reference image found "
        f"for class {target_class}"
    )

if __name__ == "__main__":

    # ----------------------------------------
    # Device
    # ----------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    # ----------------------------------------
    # Data
    # ----------------------------------------

    train_loader, val_loader, test_loader, class_names = (
        create_dataloaders()
    )

    # Query from test set
    images, labels = next(
        iter(test_loader)
    )

    query_image = images[0].unsqueeze(0)

    true_class = labels[0].item()

    # ----------------------------------------
    # Model
    # ----------------------------------------

    model = DefectCNN(
        num_classes=len(class_names)
    ).to(device)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    with torch.no_grad():

        logits = model(
            query_image.to(device)
        )

        probabilities = torch.softmax(
            logits,
            dim=1
        )

    top_probs, top_indices = torch.topk(
        probabilities,
        k=2,
        dim=1
    )

    predicted_class = (
        top_indices[0, 0].item()
    )

    counter_class = (
        top_indices[0, 1].item()
    )

    predicted_probability = (
        top_probs[0, 0].item()
    )

    counter_probability = (
        top_probs[0, 1].item()
    )

    print(
        "True:",
        class_names[true_class]
    )

    print(
        "Prediction:",
        class_names[predicted_class],
        f"{predicted_probability:.2%}"
    )

    print(
        "Counter class:",
        class_names[counter_class],
        f"{counter_probability:.2%}"
    )

    counter_image, _ = find_reference_image(
        val_loader.dataset,
        counter_class
    )

    explanation = (
        generate_counterfactual_explanation(
            model=model,
            query_image=query_image,
            counter_image=counter_image,
            predicted_class=predicted_class,
            counter_class=counter_class,
            device=device,
            quantile=0.90
        )
    )

    query_explanation = explanation["query"]
    counter_explanation = explanation["counter"]

    query_display = (
        query_image
        .squeeze()
        .detach()
        .cpu()
        * 0.5
        + 0.5
    )

    counter_display = (
        counter_image
        .squeeze()
        .detach()
        .cpu()
        * 0.5
        + 0.5
    )

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(12, 8)
    )

    # Query original
    axes[0, 0].imshow(
        query_display,
        cmap="gray"
    )

    axes[0, 0].set_title(
        f"Query\n"
        f"{class_names[predicted_class]}"
    )

    axes[0, 1].imshow(
        query_display,
        cmap="gray"
    )

    axes[0, 1].imshow(
        query_explanation[
            "discriminant_map"
        ],
        cmap="jet",
        alpha=0.50
    )

    axes[0, 1].set_title(
        f"Why "
        f"{class_names[predicted_class]}\n"
        f"not "
        f"{class_names[counter_class]}?"
    )

    axes[0, 1].imshow(
        query_display,
        cmap="gray"
    )

    axes[0, 1].imshow(
        query_explanation[
            "discriminant_map"
        ],
        cmap="jet",
        alpha=0.50
    )

    axes[0, 1].set_title(
        f"Why "
        f"{class_names[predicted_class]}\n"
        f"not "
        f"{class_names[counter_class]}?"
    )

    axes[1, 0].imshow(
        counter_display,
        cmap="gray"
    )

    axes[1, 0].set_title(
        f"Counter example\n"
        f"{class_names[counter_class]}"
    )

    axes[1, 1].imshow(
        counter_display,
        cmap="gray"
    )

    axes[1, 1].imshow(
        counter_explanation[
            "discriminant_map"
        ],
        cmap="jet",
        alpha=0.50
    )

    axes[1, 1].set_title(
        f"Why "
        f"{class_names[counter_class]}\n"
        f"not "
        f"{class_names[predicted_class]}?"
    )

    axes[1, 2].imshow(
        counter_display,
        cmap="gray"
    )

    axes[1, 2].imshow(
        counter_explanation["mask"],
        cmap="Greens",
        alpha=0.50
    )

    axes[1, 2].set_title(
        f"Specific for\n"
        f"{class_names[counter_class]}"
    )

    for ax in axes.flatten():
        ax.axis("off")

    plt.tight_layout()
    plt.show()