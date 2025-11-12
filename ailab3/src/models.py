import torch
import torch.nn as nn
import torch.nn.functional as F


def get_activation(name: str):
    name = (name or 'relu').lower()
    if name == 'relu':
        return nn.ReLU(inplace=True)
    if name == 'leaky_relu':
        return nn.LeakyReLU(negative_slope=0.1, inplace=True)
    if name == 'elu':
        return nn.ELU(inplace=True)
    return nn.ReLU(inplace=True)


class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int, activation: str = 'relu', in_channels: int = 1):
        super().__init__()
        act = get_activation(activation)
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            act,
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            act,
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            act,
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x