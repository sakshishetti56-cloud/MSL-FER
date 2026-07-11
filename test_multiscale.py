import torch

from models.backbone import CNNBackbone
from models.multiscale import MultiScaleModule

backbone = CNNBackbone()
msl = MultiScaleModule()

x = torch.randn(1, 1, 48, 48)

features = backbone(x)

print("Backbone Output:", features.shape)

output = msl(features)

print("MultiScale Output:", output.shape)