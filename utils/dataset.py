from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Training transformations
train_transform = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
])

# Testing transformations
test_transform = transforms.Compose([
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
    batch_size=64,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False
)

print("Classes:", train_dataset.classes)
print("Training Images:", len(train_dataset))
print("Testing Images:", len(test_dataset))