from xai.discriminant import (
    generate_discriminant_explanation
)


def generate_counterfactual_explanation(
    model,
    query_image,
    counter_image,
    predicted_class,
    counter_class,
    device,
    quantile=0.90
):
    """
    GALORE-style counterfactual explanation.

    Query:
        Why predicted_class instead of counter_class?

    Counter example:
        Why counter_class instead of predicted_class?
    """

    # ----------------------------------------
    # Query image
    # D(x, y*, yc)
    # ----------------------------------------

    query_explanation = (
        generate_discriminant_explanation(
            model=model,
            image=query_image,
            predicted_class=predicted_class,
            counter_class=counter_class,
            device=device,
            quantile=quantile
        )
    )

    # ----------------------------------------
    # Counter-class image
    # D(xc, yc, y*)
    # ----------------------------------------

    counter_explanation = (
        generate_discriminant_explanation(
            model=model,
            image=counter_image,
            predicted_class=counter_class,
            counter_class=predicted_class,
            device=device,
            quantile=quantile
        )
    )

    return {
        "predicted_class": predicted_class,
        "counter_class": counter_class,

        "query": query_explanation,
        "counter": counter_explanation
    }