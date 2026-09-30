from pathlib import Path
import argparse

import torch
import matplotlib.pyplot as plt
from PIL import Image

from dataset import (
    create_dataloaders,
    eval_transform
)

from model import DefectCNN

from xai.deliberative import (
    generate_deliberative_explanation
)

from xai.discriminant import (
    generate_discriminant_explanation
)

from xai.counterfactual import (
    generate_counterfactual_explanation
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "checkpoints"
    / "best_model.pth"
)

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "analysis"
)


# ============================================================
# Load arbitrary image
# ============================================================

def load_image(image_path):
    """
    Load arbitrary image using the same preprocessing
    that is used for evaluation.
    """

    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image does not exist: {image_path}"
        )

    image = Image.open(image_path)

    tensor = eval_transform(image)

    # [C, H, W] -> [1, C, H, W]
    tensor = tensor.unsqueeze(0)

    return tensor


# ============================================================
# Find counterfactual reference image
# ============================================================

def find_reference_image(
    dataset,
    target_class
):
    """
    Find a reference image from the requested class.

    Currently returns the first matching validation image.
    """

    for index in range(len(dataset)):

        image, label = dataset[index]

        if label == target_class:

            return image.unsqueeze(0)

    raise ValueError(
        f"No reference image found "
        f"for class {target_class}"
    )


# ============================================================
# Undo normalization for visualization
# ============================================================

def prepare_display_image(image):
    """
    Undo Normalize(mean=0.5, std=0.5).
    """

    return (
        image
        .squeeze()
        .detach()
        .cpu()
        * 0.5
        + 0.5
    )


# ============================================================
# Analyze one image
# ============================================================

def analyze_image(
    image_path,
    model,
    class_names,
    reference_dataset,
    device,
    output_dir=None,
    show_plot=True
):
    """
    Classify and explain a single image.
    """

    image_path = Path(image_path)

    print("\n")
    print("=" * 60)
    print("Analyzing:", image_path.name)
    print("=" * 60)

    # --------------------------------------------------------
    # Load image
    # --------------------------------------------------------

    image = load_image(
        image_path
    )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    with torch.no_grad():

        logits = model(
            image.to(device)
        )

        probabilities = torch.softmax(
            logits,
            dim=1
        )

    # --------------------------------------------------------
    # Top-2 classes
    # --------------------------------------------------------

    top_probs, top_indices = torch.topk(
        probabilities,
        k=2,
        dim=1
    )

    predicted_class = (
        top_indices[0, 0].item()
    )

    alternative_class = (
        top_indices[0, 1].item()
    )

    predicted_probability = (
        top_probs[0, 0].item()
    )

    alternative_probability = (
        top_probs[0, 1].item()
    )

    # --------------------------------------------------------
    # Print prediction
    # --------------------------------------------------------

    print(
        "\nPrediction:",
        class_names[predicted_class],
        f"{predicted_probability:.2%}"
    )

    print(
        "Alternative:",
        class_names[alternative_class],
        f"{alternative_probability:.2%}"
    )

    print("\nAll classes:")

    for class_id, class_name in enumerate(
        class_names
    ):

        probability = (
            probabilities[0, class_id].item()
        )

        print(
            f"  {class_name}: "
            f"{probability:.2%}"
        )

    # ========================================================
    # DELIBERATIVE EXPLANATION
    # ========================================================

    deliberative = (
        generate_deliberative_explanation(
            model=model,
            image=image,
            class_a=predicted_class,
            class_b=alternative_class,
            device=device,
            quantile=0.90
        )
    )

    # ========================================================
    # DISCRIMINANT EXPLANATION
    # ========================================================

    discriminant = (
        generate_discriminant_explanation(
            model=model,
            image=image,
            predicted_class=predicted_class,
            counter_class=alternative_class,
            device=device,
            quantile=0.90
        )
    )

    # ========================================================
    # COUNTERFACTUAL REFERENCE
    # ========================================================

    counter_image = find_reference_image(
        reference_dataset,
        alternative_class
    )

    # ========================================================
    # COUNTERFACTUAL EXPLANATION
    # ========================================================

    counterfactual = (
        generate_counterfactual_explanation(
            model=model,
            query_image=image,
            counter_image=counter_image,
            predicted_class=predicted_class,
            counter_class=alternative_class,
            device=device,
            quantile=0.90
        )
    )

    counter_explanation = (
        counterfactual["counter"]
    )

    # --------------------------------------------------------
    # Print XAI information
    # --------------------------------------------------------

    print(
        "\nInsecurity strength:",
        f"{deliberative['strength']:.3f}"
    )

    print(
        "Discriminant strength:",
        f"{discriminant['strength']:.3f}"
    )

    print(
        "Counterfactual reference:",
        class_names[alternative_class]
    )

    # ========================================================
    # Prepare images
    # ========================================================

    display_image = prepare_display_image(
        image
    )

    counter_display = prepare_display_image(
        counter_image
    )

    # ========================================================
    # Visualization
    # ========================================================

    fig, axes = plt.subplots(
        2,
        5,
        figsize=(20, 8)
    )

    # --------------------------------------------------------
    # Row 1
    # --------------------------------------------------------

    # Original
    axes[0, 0].imshow(
        display_image,
        cmap="gray"
    )

    axes[0, 0].set_title(
        f"Input\n"
        f"{class_names[predicted_class]} "
        f"{predicted_probability:.1%}"
    )

    # Predicted class attribution
    axes[0, 1].imshow(
        display_image,
        cmap="gray"
    )

    axes[0, 1].imshow(
        deliberative["class_a_map"],
        cmap="jet",
        alpha=0.45
    )

    axes[0, 1].set_title(
        f"Attribution\n"
        f"{class_names[predicted_class]}"
    )

    # Alternative attribution
    axes[0, 2].imshow(
        display_image,
        cmap="gray"
    )

    axes[0, 2].imshow(
        deliberative["class_b_map"],
        cmap="jet",
        alpha=0.45
    )

    axes[0, 2].set_title(
        f"Attribution\n"
        f"{class_names[alternative_class]}"
    )

    # Difficulty
    axes[0, 3].imshow(
        display_image,
        cmap="gray"
    )

    axes[0, 3].imshow(
        deliberative["difficulty_map"],
        cmap="jet",
        alpha=0.45
    )

    axes[0, 3].set_title(
        "Difficulty"
    )

    # Ambiguity
    axes[0, 4].imshow(
        display_image,
        cmap="gray"
    )

    axes[0, 4].imshow(
        deliberative["ambiguity_map"],
        cmap="jet",
        alpha=0.50
    )

    axes[0, 4].set_title(
        f"Ambiguity\n"
        f"{class_names[predicted_class]}"
        f" vs "
        f"{class_names[alternative_class]}"
    )

    # --------------------------------------------------------
    # Row 2
    # --------------------------------------------------------

    # Insecurity region
    axes[1, 0].imshow(
        display_image,
        cmap="gray"
    )

    axes[1, 0].imshow(
        deliberative["mask"],
        cmap="Reds",
        alpha=0.50
    )

    axes[1, 0].set_title(
        f"Insecurity Region\n"
        f"Strength: "
        f"{deliberative['strength']:.2f}"
    )

    # Discriminant map
    axes[1, 1].imshow(
        display_image,
        cmap="gray"
    )

    axes[1, 1].imshow(
        discriminant["discriminant_map"],
        cmap="jet",
        alpha=0.50
    )

    axes[1, 1].set_title(
        f"Why "
        f"{class_names[predicted_class]}\n"
        f"not "
        f"{class_names[alternative_class]}?"
    )

    # Discriminant region
    axes[1, 2].imshow(
        display_image,
        cmap="gray"
    )

    axes[1, 2].imshow(
        discriminant["mask"],
        cmap="Greens",
        alpha=0.50
    )

    axes[1, 2].set_title(
        f"Specific for\n"
        f"{class_names[predicted_class]}"
    )

    # Counterfactual reference
    axes[1, 3].imshow(
        counter_display,
        cmap="gray"
    )

    axes[1, 3].set_title(
        f"Counter Example\n"
        f"{class_names[alternative_class]}"
    )

    # Counterfactual discriminant region
    axes[1, 4].imshow(
        counter_display,
        cmap="gray"
    )

    axes[1, 4].imshow(
        counter_explanation["mask"],
        cmap="Greens",
        alpha=0.50
    )

    axes[1, 4].set_title(
        f"Specific for\n"
        f"{class_names[alternative_class]}"
    )

    # --------------------------------------------------------
    # Hide axes
    # --------------------------------------------------------

    for ax in axes.flatten():
        ax.axis("off")

    fig.suptitle(
        f"{image_path.name}\n"
        f"Prediction: "
        f"{class_names[predicted_class]} "
        f"({predicted_probability:.1%}) "
        f"| Alternative: "
        f"{class_names[alternative_class]} "
        f"({alternative_probability:.1%})",
        fontsize=14
    )

    plt.tight_layout()

    # ========================================================
    # Save result
    # ========================================================

    if output_dir is not None:

        output_dir = Path(
            output_dir
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        output_path = (
            output_dir
            / f"{image_path.stem}_analysis.png"
        )

        plt.savefig(
            output_path,
            dpi=200,
            bbox_inches="tight"
        )

        print(
            "\nSaved:",
            output_path
        )

    # ========================================================
    # Show or close
    # ========================================================

    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    # ========================================================
    # Return structured result
    # ========================================================

    return {
        "file": str(image_path),

        "predicted_class": (
            class_names[predicted_class]
        ),

        "predicted_probability": (
            predicted_probability
        ),

        "alternative_class": (
            class_names[alternative_class]
        ),

        "alternative_probability": (
            alternative_probability
        ),

        "insecurity_strength": (
            deliberative["strength"]
        ),

        "discriminant_strength": (
            discriminant["strength"]
        )
    }


# ============================================================
# Analyze folder
# ============================================================

def analyze_folder(
    folder_path,
    model,
    class_names,
    reference_dataset,
    device,
    output_dir
):
    """
    Analyze all supported images in a folder.
    """

    folder_path = Path(
        folder_path
    )

    if not folder_path.exists():
        raise FileNotFoundError(
            f"Folder does not exist: "
            f"{folder_path}"
        )

    supported_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".tif",
        ".tiff"
    }

    image_paths = [
        path
        for path in folder_path.iterdir()
        if (
            path.is_file()
            and path.suffix.lower()
            in supported_extensions
        )
    ]

    image_paths.sort()

    if len(image_paths) == 0:

        print(
            "No supported images found."
        )

        return

    print(
        f"\nFound "
        f"{len(image_paths)} images."
    )

    results = []

    for image_path in image_paths:

        try:

            result = analyze_image(
                image_path=image_path,
                model=model,
                class_names=class_names,
                reference_dataset=reference_dataset,
                device=device,
                output_dir=output_dir,
                show_plot=False
            )

            results.append(
                result
            )

        except Exception as error:

            print(
                f"\nERROR with "
                f"{image_path.name}:"
            )

            print(
                error
            )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n")
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)

    for result in results:

        print(
            f"{Path(result['file']).name}: "
            f"{result['predicted_class']} "
            f"({result['predicted_probability']:.1%})"
        )


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Surface defect classification "
            "and GALORE-style XAI."
        )
    )

    input_group = (
        parser.add_mutually_exclusive_group(
            required=True
        )
    )

    input_group.add_argument(
        "--image",
        type=str,
        help="Path to a single image."
    )

    input_group.add_argument(
        "--folder",
        type=str,
        help=(
            "Analyze all images "
            "inside a folder."
        )
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(
            DEFAULT_OUTPUT_DIR
        ),
        help=(
            "Directory where analysis "
            "figures are saved."
        )
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "Device:",
        device
    )

    # --------------------------------------------------------
    # Dataset / classes / reference pool
    # --------------------------------------------------------

    (
        _,
        val_loader,
        _,
        class_names
    ) = create_dataloaders()

    reference_dataset = (
        val_loader.dataset
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = DefectCNN(
        num_classes=len(class_names)
    ).to(device)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model.eval()

    # --------------------------------------------------------
    # Single image
    # --------------------------------------------------------

    if args.image is not None:

        analyze_image(
            image_path=args.image,
            model=model,
            class_names=class_names,
            reference_dataset=reference_dataset,
            device=device,
            output_dir=args.output_dir,
            show_plot=True
        )

    # --------------------------------------------------------
    # Folder
    # --------------------------------------------------------

    elif args.folder is not None:

        analyze_folder(
            folder_path=args.folder,
            model=model,
            class_names=class_names,
            reference_dataset=reference_dataset,
            device=device,
            output_dir=args.output_dir
        )


if __name__ == "__main__":
    main()