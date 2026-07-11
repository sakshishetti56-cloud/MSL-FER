import torch

from models.mslfer import MSLFER

model = MSLFER()

x = torch.randn(4, 1, 48, 48)

y = model(x)

print("Input :", x.shape)
print("Output:", y.shape)