from xai.gradcam import (
    generate_class_attribute,
    generate_confidence_attribution,
    normalize_heatmap,
    resize_heatmap
)

from xai.region import (
    create_quantile_mask,
    region_strength
)


def generate_discriminant_explanation(
    model,
    image,
    predicted_class,
    counter_class,
    device,
    quantile=0.90
):
    """
    GALORE-style discriminant explanation.

    D(x, y*, yc) =
        attribution(y*)
        * complement(attribution(yc))
        * confidence_attribution
    """

    # --------------------------------------------------
    # Attribution for predicted class
    # --------------------------------------------------

    predicted_map = generate_class_attribute(
        model=model,
        image=image,
        class_id=predicted_class,
        device=device,
        resize_to_input=False
    )

    # --------------------------------------------------
    # Attribution for counter class
    # --------------------------------------------------

    counter_map = generate_class_attribute(
        model=model,
        image=image,
        class_id=counter_class,
        device=device,
        resize_to_input=False
    )

    # --------------------------------------------------
    # Confidence attribution
    # --------------------------------------------------

    confidence_map = generate_confidence_attribution(
        model=model,
        image=image,
        device=device,
        resize_to_input=False
    )

    # --------------------------------------------------
    # Complement of counter-class attribution
    #
    # Because our attribution maps are normalized to [0, 1],
    # this corresponds to the complement used in GALORE.
    # --------------------------------------------------

    counter_complement = 1.0 - counter_map

    # --------------------------------------------------
    # GALORE discriminant map
    # --------------------------------------------------

    discriminant = (
        predicted_map
        * counter_complement
        * confidence_map
    )

    discriminant = normalize_heatmap(
        discriminant
    )

    # --------------------------------------------------
    # Resize only after combination
    # --------------------------------------------------

    output_size = image.shape[-2:]

    predicted_map_resized = resize_heatmap(
        predicted_map,
        output_size
    )

    counter_map_resized = resize_heatmap(
        counter_map,
        output_size
    )

    confidence_map_resized = resize_heatmap(
        confidence_map,
        output_size
    )

    discriminant_map = resize_heatmap(
        discriminant,
        output_size
    )

    # --------------------------------------------------
    # Region mask
    # --------------------------------------------------

    mask, threshold = create_quantile_mask(
        discriminant_map,
        quantile=quantile
    )

    strength = region_strength(
        discriminant_map,
        mask
    )

    # --------------------------------------------------
    # Result
    # --------------------------------------------------

    return {
        "predicted_class": predicted_class,
        "counter_class": counter_class,

        "predicted_map": predicted_map_resized,
        "counter_map": counter_map_resized,
        "confidence_map": confidence_map_resized,

        "discriminant_map": discriminant_map,

        "mask": mask,
        "threshold": threshold,
        "strength": strength
    }