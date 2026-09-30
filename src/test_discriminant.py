from pathlib import Path

import torch
import matplotlib.pyplot as plt

from dataset import create_dataloaders
from model import DefectCNN

from xai.discriminant import (
    generate_discriminant_explanation
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "checkpoints"
    / "best_model.pth"
)


if __name__ == "__main__":

    # --------------------------------------------------
    # Device
    # --------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    # --------------------------------------------------
    # Dataset
    # --------------------------------------------------

    _, _, test_loader, class_names = (
        create_dataloaders()
    )

    images, labels = next(
        iter(test_loader)
    )

    image = images[0].unsqueeze(0)

    true_class = labels[0].item()

    # --------------------------------------------------
    # Model
    # --------------------------------------------------

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

    # --------------------------------------------------
    # Prediction
    # --------------------------------------------------

    with torch.no_grad():

        logits = model(
            image.to(device)
        )

        probabilities = torch.softmax(
            logits,
            dim=1
        )

    # --------------------------------------------------
    # Top-1 and Top-2
    # --------------------------------------------------

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

    # --------------------------------------------------
    # Discriminant explanation
    #
    # IMPORTANT:
    # No torch.no_grad() here.
    # Attribution requires gradients.
    # --------------------------------------------------

    explanation = generate_discriminant_explanation(
        model=model,
        image=image,
        predicted_class=predicted_class,
        counter_class=counter_class,
        device=device,
        quantile=0.90
    )

    print(
        "Discriminant strength:",
        f"{explanation['strength']:.3f}"
    )

    # --------------------------------------------------
    # Display image
    # --------------------------------------------------

    display_image = (
        image
        .squeeze()
        .detach()
        .cpu()
        * 0.5
        + 0.5
    )

    # --------------------------------------------------
    # Visualization
    # --------------------------------------------------

    fig, axes = plt.subplots(
        1,
        6,
        figsize=(24, 4)
    )

    # Original
    axes[0].imshow(
        display_image,
        cmap="gray"
    )

    axes[0].set_title(
        "Original"
    )

    # Predicted class attribution
    axes[1].imshow(
        display_image,
        cmap="gray"
    )

    axes[1].imshow(
        explanation["predicted_map"],
        cmap="jet",
        alpha=0.45
    )

    axes[1].set_title(
        f"{class_names[predicted_class]}\n"
        f"{predicted_probability:.1%}"
    )

    # Counter class attribution
    axes[2].imshow(
        display_image,
        cmap="gray"
    )

    axes[2].imshow(
        explanation["counter_map"],
        cmap="jet",
        alpha=0.45
    )

    axes[2].set_title(
        f"{class_names[counter_class]}\n"
        f"{counter_probability:.1%}"
    )

    # Confidence
    axes[3].imshow(
        display_image,
        cmap="gray"
    )

    axes[3].imshow(
        explanation["confidence_map"],
        cmap="jet",
        alpha=0.45
    )

    axes[3].set_title(
        "Confidence"
    )

    # Discriminant heatmap
    axes[4].imshow(
        display_image,
        cmap="gray"
    )

    axes[4].imshow(
        explanation["discriminant_map"],
        cmap="jet",
        alpha=0.55
    )

    axes[4].set_title(
        f"Why {class_names[predicted_class]}\n"
        f"not {class_names[counter_class]}?"
    )

    # Discriminant region
    axes[5].imshow(
        display_image,
        cmap="gray"
    )

    axes[5].imshow(
        explanation["mask"],
        cmap="Greens",
        alpha=0.50
    )

    axes[5].set_title(
        "Discriminant Region\n"
        f"Strength: "
        f"{explanation['strength']:.2f}"
    )

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.show()