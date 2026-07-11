import torch
import torch.nn as nn


class MultiScaleModule(nn.Module):
    def __init__(self, in_channels=128, out_channels=64):
        super(MultiScaleModule, self).__init__()

        self.conv3 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1
        )

        self.conv5 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=5,
            padding=2
        )

        self.conv7 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=7,
            padding=3
        )

        self.relu = nn.ReLU()

    def forward(self, x):

        x1 = self.relu(self.conv3(x))
        x2 = self.relu(self.conv5(x))
        x3 = self.relu(self.conv7(x))

        output = torch.cat([x1, x2, x3], dim=1)

        return output