import torch
import torch.nn.functional as F

def generate_gradcam(
        model,
        image,
        target_class,
        device,
):
    model.eval()

    image = image.to(device)

    image.requires_grad = True

    logits, features,_ = model(
        image,
        return_metadata=True,
    )

    features.retain_grad()

    target_score = logits[:, target_class]

    model.zero_grad()

    target_score.backward()

    gradients = features.grad

    weights = gradients.mean(
        dim = (2,3),
        keepdim = True,
    )

    cam = (
        weights * features
    ).sum(
        dim = 1,
        keepdim = True,
    )

    cam = F.relu(cam)

    cam = F.interpolate(
        cam,
        size = image.shape[-2:],
        mode = "bilinear",
        align_corners = False,
    )

    cam = cam.squeeze()

    cam_min = cam.min()
    cam_max = cam.max()

    cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)

    return cam.detach().cpu()

def normalize_heatmap(heatmap):
    minimum = heatmap.min()
    maximum = heatmap.max()

    return (heatmap - minimum) / (maximum - minimum) + 1e-8

def resize_heatmap(heatmap, size):

    # Remove all dimensions of size 1
    heatmap = heatmap.squeeze()

    if heatmap.ndim != 2:
        raise ValueError(
            f"Expected 2D heatmap, got shape {heatmap.shape}"
        )

    # [H, W] -> [1, 1, H, W]
    heatmap = heatmap.unsqueeze(0).unsqueeze(0)

    heatmap = F.interpolate(
        heatmap,
        size=size,
        mode="bilinear",
        align_corners=False
    )

    # [1, 1, H, W] -> [H, W]
    heatmap = heatmap[0, 0]

    return heatmap

def generate_score_gradcam(
        model,
        image,
        score_function,
        device,
        resize_to_input=True,
):
    model.eval()
    image_copy = (image
                  .to(device)
                  .clone()
                  .detach()
                  .requires_grad_(True))

    logits, features,_ = model(image_copy, return_metadata=True)

    features.retain_grad()

    score = score_function(logits)

    model.zero_grad(set_to_none=True)

    score.sum().backward()
    gradients = features.grad
    weights = gradients.mean(
        dim = (2,3),
        keepdim = True,
    )

    cam = (
        weights * features
    ).sum(dim = 1, keepdim = True)

    cam = F.relu(cam)

    cam = cam[0, 0]

    cam = normalize_heatmap(cam)

    if resize_to_input:
        cam = resize_heatmap(cam, image.shape[-2:])

    return cam.detach().cpu()

def generate_class_attribute(
        model,
        image,
        class_id,
        device,
        resize_to_input=True,
):
    def score_function(logits):
        probs = torch.softmax(
            logits,
            dim = 1,
        )

        return probs[:, class_id]

    return generate_score_gradcam(
        model = model,
        image = image,
        score_function = score_function,
        device = device,
        resize_to_input = resize_to_input,
    )

def generate_confidence_attribution(
        model,
        image,
        device,
        resize_to_input=True,
):
    def score_function(logits):

        probs = torch.softmax(
            logits,
            dim = 1,
        )

        confidence = probs.max(dim = 1).values

        return confidence

    return generate_score_gradcam(
        model=model,
        image=image,
        score_function=score_function,
        device=device,
        resize_to_input=resize_to_input
    )

def generate_difficulty_attribution(
    model,
    image,
    device,
    resize_to_input=True
):
    def score_function(logits):

        probabilities = torch.softmax(
            logits,
            dim=1
        )

        confidence = probabilities.max(
            dim=1
        ).values

        difficulty = 1.0 - confidence

        return difficulty

    return generate_score_gradcam(
        model=model,
        image=image,
        score_function=score_function,
        device=device,
        resize_to_input=resize_to_input
    )