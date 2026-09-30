import torch
from src.model import DefectCNN

if __name__ == '__main__':
    model = DefectCNN(num_classes=6)

    dummy_image = torch.randn(
        4,
        1,
        200,
        200
    )

    logits, features, parameters = model(
        dummy_image,
        return_metadata=True
    )

    print("Logits:")
    print(logits.shape)

    print("Features:")
    print(features.shape)

    print("parameters:")
    print(model.parameters)