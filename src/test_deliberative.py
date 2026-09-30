from pathlib import Path

import torch
import matplotlib.pyplot as plt

from dataset import create_dataloaders
from model import DefectCNN

from xai.deliberative import (
    generate_deliberative_explanation,
    generate_multiple_insecurities
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
    # Data
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

    # Here no gradients are needed
    with torch.no_grad():

        logits = model(
            image.to(device)
        )

        probabilities = torch.softmax(
            logits,
            dim=1
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
    # Top-2
    # --------------------------------------------------

    top_probs, top_indices = torch.topk(
        probabilities,
        k=2,
        dim=1
    )

    class_a = top_indices[0, 0].item()
    class_b = top_indices[0, 1].item()

    prob_a = top_probs[0, 0].item()
    prob_b = top_probs[0, 1].item()

    print(
        "True:",
        class_names[true_class]
    )

    print(
        "Prediction:",
        class_names[class_a],
        f"{prob_a:.2%}"
    )

    print(
        "Alternative:",
        class_names[class_b],
        f"{prob_b:.2%}"
    )

    # --------------------------------------------------
    # ONE deliberative explanation
    # --------------------------------------------------

    # IMPORTANT:
    # No torch.no_grad() here.
    # Grad-CAM needs gradients.

    explanation = generate_deliberative_explanation(
        model=model,
        image=image,
        class_a=class_a,
        class_b=class_b,
        device=device,
        quantile=0.90
    )

    mask = explanation["mask"]
    threshold = explanation["threshold"]
    strength = explanation["strength"]

    print(
        "Threshold:",
        f"{threshold:.3f}"
    )

    print(
        "Strength:",
        f"{strength:.3f}"
    )

    # --------------------------------------------------
    # Visualization: single insecurity
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

    # Top-1 attribution
    axes[1].imshow(
        display_image,
        cmap="gray"
    )

    axes[1].imshow(
        explanation["class_a_map"],
        cmap="jet",
        alpha=0.45
    )

    axes[1].set_title(
        f"{class_names[class_a]}\n"
        f"{prob_a:.1%}"
    )

    # Top-2 attribution
    axes[2].imshow(
        display_image,
        cmap="gray"
    )

    axes[2].imshow(
        explanation["class_b_map"],
        cmap="jet",
        alpha=0.45
    )

    axes[2].set_title(
        f"{class_names[class_b]}\n"
        f"{prob_b:.1%}"
    )

    # Difficulty
    axes[3].imshow(
        display_image,
        cmap="gray"
    )

    axes[3].imshow(
        explanation["difficulty_map"],
        cmap="jet",
        alpha=0.45
    )

    axes[3].set_title(
        "Difficulty"
    )

    # Ambiguity
    axes[4].imshow(
        display_image,
        cmap="gray"
    )

    axes[4].imshow(
        explanation["ambiguity_map"],
        cmap="jet",
        alpha=0.55
    )

    axes[4].set_title(
        f"Ambiguity\n"
        f"{class_names[class_a]}"
        f" vs "
        f"{class_names[class_b]}"
    )

    # Insecurity region
    axes[5].imshow(
        display_image,
        cmap="gray"
    )

    axes[5].imshow(
        mask,
        cmap="Reds",
        alpha=0.50
    )

    axes[5].set_title(
        f"Insecurity Region\n"
        f"{class_names[class_a]}"
        f" vs "
        f"{class_names[class_b]}"
        f"\nStrength: {strength:.2f}"
    )

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.show()

    # --------------------------------------------------
    # MULTIPLE insecurities
    # --------------------------------------------------

    # Again:
    # Gradients are needed here.
    insecurities = generate_multiple_insecurities(
        model=model,
        image=image,
        probabilities=probabilities,
        device=device,
        top_k=3,
        quantile=0.90
    )

    print("\nDetected insecurities:")

    for i, insecurity in enumerate(
        insecurities,
        start=1
    ):
        pair_a = insecurity["class_a"]
        pair_b = insecurity["class_b"]

        print(
            f"{i}. "
            f"{class_names[pair_a]} "
            f"<-> "
            f"{class_names[pair_b]} "
            f"| strength = "
            f"{insecurity['strength']:.3f}"
        )

    # --------------------------------------------------
    # Visualization: multiple insecurities
    # --------------------------------------------------

    fig, axes = plt.subplots(
        1,
        len(insecurities) + 1,
        figsize=(5 * (len(insecurities) + 1), 5)
    )

    # Original
    axes[0].imshow(
        display_image,
        cmap="gray"
    )

    axes[0].set_title(
        "Original"
    )

    axes[0].axis("off")

    # Insecurities
    for i, insecurity in enumerate(
        insecurities,
        start=1
    ):

        pair_a = insecurity["class_a"]
        pair_b = insecurity["class_b"]

        insecurity_mask = insecurity["mask"]
        insecurity_strength = insecurity["strength"]

        axes[i].imshow(
            display_image,
            cmap="gray"
        )

        axes[i].imshow(
            insecurity_mask,
            cmap="Reds",
            alpha=0.45
        )

        axes[i].set_title(
            f"{class_names[pair_a]}\n"
            f"vs\n"
            f"{class_names[pair_b]}\n"
            f"Strength: "
            f"{insecurity_strength:.2f}"
        )

        axes[i].axis("off")

    plt.tight_layout()
    plt.show()