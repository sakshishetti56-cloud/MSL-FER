import torch
from models.backbone import CNNBackbone

model = CNNBackbone()

x = torch.randn(1, 1, 48, 48)

y = model(x)

print("Input Shape :", x.shape)
print("Output Shape:", y.shape)