import torch
import torch.nn as nn

from models.backbone import CNNBackbone
from models.multiscale import MultiScaleModule
from models.attention import CBAM


class MSLFER(nn.Module):

    def __init__(self, num_classes=7):
        super(MSLFER, self).__init__()

        self.backbone = CNNBackbone()

        self.multiscale = MultiScaleModule(
            in_channels=128,
            out_channels=64
        )

        self.attention = CBAM(192)

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(192, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):

        x = self.backbone(x)

        x = self.multiscale(x)

        x = self.attention(x)

        x = self.pool(x)

        x = self.classifier(x)

        return x