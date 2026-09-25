"""
Siamese network architecture used to reconstruct the model from a saved state_dict.

This class is used when the model artifact is stored as weights only
(`torch.save(model.state_dict(), ...)`) instead of a TorchScript archive.
"""

import torch
from torch import nn


class Backbone(nn.Module):
    def __init__(self, input_channels: int = 1, embedding_dim: int = 128) -> None:
        super().__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=False),

            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=False),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=False),

            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=False),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2),
        )

        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4096, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_layers(x)
        x = self.adaptive_pool(x)
        x = self.fc_layers(x)
        return nn.functional.normalize(x, p=2, dim=1)


class SiameseNineNet(nn.Module):
    def __init__(self, embedding_dim: int = 128, input_channels: int = 1) -> None:
        super().__init__()
        self.backbone = Backbone(input_channels=input_channels, embedding_dim=embedding_dim)

    def forward(
        self,
        g1: torch.Tensor,
        g2: torch.Tensor,
        g3: torch.Tensor,
        g4: torch.Tensor,
        g5: torch.Tensor,
        f1: torch.Tensor,
        f2: torch.Tensor,
        f3: torch.Tensor,
        f4: torch.Tensor,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        genuine_embeddings = [self.backbone(g) for g in (g1, g2, g3, g4, g5)]
        forged_embeddings = [self.backbone(f) for f in (f1, f2, f3, f4)]
        return genuine_embeddings, forged_embeddings

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
