import torch

from itertools import combinations

from xai.gradcam import (
    generate_class_attribute,
    generate_difficulty_attribution,
    resize_heatmap,
    normalize_heatmap
)

from xai.region import (
    region_strength,
    create_quantile_mask
)


def generate_deliberative_explanation(
    model,
    image,
    class_a,
    class_b,
    device,
    quantile=0.90
):
    """
    Generate one GALORE-style insecurity for two classes.

    I(x, {a,b}) =
        attribution(a)
        * attribution(b)
        * difficulty
    """

    # ----------------------------------------
    # Class attribution
    # ----------------------------------------

    attribution_a = generate_class_attribute(
        model=model,
        image=image,
        class_id=class_a,
        device=device,
        resize_to_input=False
    )

    attribution_b = generate_class_attribute(
        model=model,
        image=image,
        class_id=class_b,
        device=device,
        resize_to_input=False
    )

    # ----------------------------------------
    # Difficulty attribution
    # ----------------------------------------

    difficulty = generate_difficulty_attribution(
        model=model,
        image=image,
        device=device,
        resize_to_input=False
    )

    # ----------------------------------------
    # GALORE ambiguity map
    # ----------------------------------------

    ambiguity = (
        attribution_a
        * attribution_b
        * difficulty
    )

    ambiguity = normalize_heatmap(
        ambiguity
    )

    # ----------------------------------------
    # Resize only after combination
    # ----------------------------------------

    output_size = image.shape[-2:]

    class_a_map = resize_heatmap(
        attribution_a,
        output_size
    )

    class_b_map = resize_heatmap(
        attribution_b,
        output_size
    )

    difficulty_map = resize_heatmap(
        difficulty,
        output_size
    )

    ambiguity_map = resize_heatmap(
        ambiguity,
        output_size
    )

    # ----------------------------------------
    # Insecurity region
    # ----------------------------------------

    mask, threshold = create_quantile_mask(
        ambiguity_map,
        quantile=quantile
    )

    strength = region_strength(
        ambiguity_map,
        mask
    )

    # ----------------------------------------
    # Result
    # ----------------------------------------

    return {
        "class_a": class_a,
        "class_b": class_b,

        "class_a_map": class_a_map,
        "class_b_map": class_b_map,

        "difficulty_map": difficulty_map,
        "ambiguity_map": ambiguity_map,

        "mask": mask,
        "threshold": threshold,
        "strength": strength
    }


def generate_multiple_insecurities(
    model,
    image,
    probabilities,
    device,
    top_k=3,
    quantile=0.90
):
    """
    Generate GALORE-style insecurities for all
    pairs among the top-k predicted classes.
    """

    # ----------------------------------------
    # Top-k classes
    # ----------------------------------------

    _, top_indices = torch.topk(
        probabilities,
        k=top_k,
        dim=1
    )

    class_ids = top_indices[0].tolist()

    # ----------------------------------------
    # Difficulty only once
    # ----------------------------------------

    difficulty = generate_difficulty_attribution(
        model=model,
        image=image,
        device=device,
        resize_to_input=False
    )

    # ----------------------------------------
    # Attribution for each top-k class
    # ----------------------------------------

    class_maps = {}

    for class_id in class_ids:

        class_maps[class_id] = generate_class_attribute(
            model=model,
            image=image,
            class_id=class_id,
            device=device,
            resize_to_input=False
        )

    # ----------------------------------------
    # All class pairs
    # ----------------------------------------

    insecurities = []

    for class_a, class_b in combinations(
        class_ids,
        2
    ):

        ambiguity = (
            class_maps[class_a]
            * class_maps[class_b]
            * difficulty
        )

        ambiguity = normalize_heatmap(
            ambiguity
        )

        ambiguity_map = resize_heatmap(
            ambiguity,
            image.shape[-2:]
        )

        mask, threshold = create_quantile_mask(
            ambiguity_map,
            quantile=quantile
        )

        strength = region_strength(
            ambiguity_map,
            mask
        )

        insecurities.append({
            "class_a": class_a,
            "class_b": class_b,
            "ambiguity_map": ambiguity_map,
            "mask": mask,
            "threshold": threshold,
            "strength": strength
        })

    # ----------------------------------------
    # Strongest insecurity first
    # ----------------------------------------

    insecurities.sort(
        key=lambda item: item["strength"],
        reverse=True
    )

    return insecurities