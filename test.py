import torch
import torchvision
import cv2

print("PyTorch:", torch.__version__)
print("Torchvision:", torchvision.__version__)
print("OpenCV:", cv2.__version__)
print("CUDA Available:", torch.cuda.is_available())