import torch

def create_quantile_mask(
        heatmap,
        quantile = 0.90,
):
    heatmap = heatmap.squeeze()

    threshold = torch.quantile(
        heatmap,
        quantile,
    )

    mask = heatmap >= threshold

    return mask, threshold.item()

def region_strength(
    heatmap,
    mask
):
    heatmap = heatmap.squeeze()
    mask = mask.squeeze()

    values = heatmap[mask]

    if values.numel() == 0:
        return 0.0

    return values.mean().item()