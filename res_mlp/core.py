import torch
from torch import nn
from torch.nn import functional as F


class AffineTransform(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, 1, num_features))
        self.beta = nn.Parameter(torch.zeros(1, 1, num_features))

    def forward(self, x):
        return self.alpha * x + self.beta


class CommunicationLayer(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.aff1 = AffineTransform(num_features)
        self.fc1 = nn.Linear(num_features, num_features)
        self.aff2 = AffineTransform(num_features)

    def forward(self, x):
        x = self.aff1(x)
        residual = x
        x = self.fc1(x.transpose(1, 2)).transpose(1, 2)
        x = self.aff2(x)
        out = x + residual
        return out


class TwoLayerResidualPerceptron(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.aff1 = AffineTransform(num_features)
        self.fc1 = nn.Linear(num_features, num_features)
        self.fc2 = nn.Linear(num_features, num_features)
        self.aff2 = AffineTransform(num_features)

    def forward(self, x):
        x = self.aff1(x)
        residual = x
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        x = self.aff2(x)
        out = x + residual
        return out


class ResidualMultiLayerPerceptron(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.cl = CommunicationLayer(num_features)
        self.ff = TwoLayerResidualPerceptron(num_features)

    def forward(self, x):
        x = self.cl(x)
        out = self.ff(x)
        return out


class ResMLP(nn.Module):
    def __init__(
        self,
        image_size=256,
        patch_size=16,
        in_channels=3,
        num_features=128,
        num_layers=6,
        num_classes=10,
    ):
        sqrt_num_patches, remainder = divmod(image_size, patch_size)
        assert remainder == 0, "`image_size` must be divisibe by `patch_size`"
        num_patches = sqrt_num_patches ** 2
        super().__init__()
        self.patcher = nn.Conv2d(
            in_channels, num_features, kernel_size=patch_size, stride=patch_size
        )
        self.mlps = nn.Sequential(
            *[ResidualMultiLayerPerceptron() for _ in range(num_layers)]
        )
        self.classifier = nn.Linear(num_features, num_features)

    def forward(self, x):
        patches = self.patcher(x)
        embedding = self.mlps(patches)
        embedding = torch.mean(embedding, dim=1)
        logits = self.classifier(embedding)
        return logits

