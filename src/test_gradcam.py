from pathlib import Path

import torch
import matplotlib.pyplot as plt

from dataset import create_dataloaders
from model import DefectCNN
from xai.gradcam import generate_gradcam

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "checkpoints"
    / "best_model.pth"
)

if __name__ == "__main__":

    def generate_all_gradcams(
        model,
        image,
        num_classes,
        device
    ):
        heatmaps = {}

        for class_id in range(num_classes):

            heatmaps[class_id] = generate_gradcam(
                model=model,
                image=image,
                target_class=class_id,
                device=device
            )

        return heatmaps

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    _, _, test_loader, class_names = (
        create_dataloaders()
    )

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

    images, labels = next(
        iter(test_loader)
    )

    image = images[0].unsqueeze(0)

    true_class = labels[0].item()

    with torch.no_grad():

        logits = model(
            image.to(device)
        )

        probabilities = torch.softmax(
            logits,
            dim=1
        )

    predicted_class = probabilities.argmax(
        dim=1
    ).item()

    confidence = probabilities[
        0,
        predicted_class
    ].item()

    print(
        "True:",
        class_names[true_class]
    )

    print(
        "Prediction:",
        class_names[predicted_class]
    )

    print(
        "Confidence:",
        f"{confidence:.2%}"
    )

    heatmap = generate_gradcam(
        model=model,
        image=image,
        target_class=predicted_class,
        device=device
    )

    print("Heatmap shape:", heatmap.shape)

    display_image = (
            image
            .squeeze()
            .detach()
            .cpu()
            * 0.5
            + 0.5
    )

    plt.figure(
        figsize=(6, 6)
    )

    plt.imshow(
        display_image,
        cmap="gray"
    )

    plt.imshow(
        heatmap,
        cmap="jet",
        alpha=0.45
    )

    plt.title(
        f"{class_names[predicted_class]} "
        f"({confidence:.1%})"
    )

    plt.axis("off")
    plt.show()

    print("True:",
          class_names[true_class])

    print("Prediction:",
          class_names[predicted_class])

    print("Confidence:",
          f"{confidence:.2%}")

    heatmaps = generate_all_gradcams(
        model=model,
        image=image,
        num_classes=len(class_names),
        device=device
    )

    display_image = (
            image
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

    axes = axes.flatten()

    for class_id, class_name in enumerate(class_names):
        ax = axes[class_id]

        ax.imshow(
            display_image,
            cmap="gray"
        )

        ax.imshow(
            heatmaps[class_id],
            cmap="jet",
            alpha=0.45
        )

        probability = probabilities[
            0,
            class_id
        ].item()

        ax.set_title(
            f"{class_name}\n"
            f"{probability:.1%}"
        )

        ax.axis("off")

    plt.tight_layout()
    plt.show()

    top_probs, top_indices = torch.topk(
        probabilities,
        k=2,
        dim=1
    )

    top1_idx = top_indices[0, 0].item()
    top2_idx = top_indices[0, 1].item()

    top1_prob = top_probs[0, 0].item()
    top2_prob = top_probs[0, 1].item()

    print()
    print(
        "Top 1:",
        class_names[top1_idx],
        f"{top1_prob:.2%}"
    )

    print(
        "Top 2:",
        class_names[top2_idx],
        f"{top2_prob:.2%}"
    )