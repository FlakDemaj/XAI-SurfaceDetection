import torch
import torch.nn as nn

class DefectCNN(nn.Module):

    def __init__(self, num_classes = 6):
        super().__init__()

        self.features = nn.Sequential(

            # Input: 1 x 200 x 200
            nn.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            # 32 x 100 x 100
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            # 64 x 50 x 50
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            # 128 x 25 x 25
            nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True)
        )

        self.global_pool = nn.AdaptiveAvgPool2d((1,1))
        self.classifier = nn.Linear(
            in_features=256,
            out_features=num_classes
        )

    def forward(self, x, return_metadata = False):
        features = self.features(x)
        pooled = self.global_pool(features)

        pooled = torch.flatten(
            pooled,
            start_dim = 1
        )

        logits = self.classifier(pooled)

        if return_metadata:
            num_parameters = sum(
                p.numel()
                for p in self.parameters()
            )
            return logits, features, num_parameters

        return logits