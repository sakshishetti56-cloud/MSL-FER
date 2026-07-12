from configs.config import BATCH_SIZE, IMAGE_SIZE
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Training transformations
train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48, 48)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
])


# Testing transformations
test_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
])

# Load datasets
train_dataset = datasets.ImageFolder(
    root="dataset/train",
    transform=train_transform
)

test_dataset = datasets.ImageFolder(
    root="dataset/test",
    transform=test_transform
)

# Create DataLoaders
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=False
)

classes = train_dataset.classes