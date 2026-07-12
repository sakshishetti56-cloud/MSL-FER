import torch

# Dataset
IMAGE_SIZE = 48
NUM_CLASSES = 7

# Training
BATCH_SIZE = 64
EPOCHS = 30
LEARNING_RATE = 0.001

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Checkpoints
CHECKPOINT_PATH = "checkpoints/best_model.pth"