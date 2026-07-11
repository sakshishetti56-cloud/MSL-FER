import torch

from models.backbone import CNNBackbone
from models.multiscale import MultiScaleModule
from models.attention import CBAM

backbone = CNNBackbone()
msl = MultiScaleModule()
attention = CBAM(192)

x = torch.randn(1, 1, 48, 48)

features = backbone(x)
features = msl(features)

print("Before Attention:", features.shape)

features = attention(features)

print("After Attention :", features.shape)