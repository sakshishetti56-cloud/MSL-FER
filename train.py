import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from configs.config import (
    DEVICE,
    EPOCHS,
    LEARNING_RATE,
    CHECKPOINT_PATH
)

from utils.dataset import train_loader, test_loader
from models.mslfer import MSLFER
from utils.seed import set_seed
from utils.early_stopping import EarlyStopping
from utils.visualize import plot_curves


# ===========================
# Create folders
# ===========================
os.makedirs("checkpoints", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# ===========================
# Seed
# ===========================
set_seed(42)

# ===========================
# Model
# ===========================
model = MSLFER().to(DEVICE)

# ===========================
# Loss
# ===========================
criterion = nn.CrossEntropyLoss()

# ===========================
# Optimizer
# ===========================
optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

# ===========================
# Scheduler
# ===========================
scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=EPOCHS
)

# ===========================
# Early Stopping
# ===========================
early_stopping = EarlyStopping(patience=5)

# ===========================
# Training Info
# ===========================
print("=" * 50)
print("MSL-FER Training Setup")
print("=" * 50)
print(f"Device          : {DEVICE}")
print(f"Epochs          : {EPOCHS}")
print(f"Learning Rate   : {LEARNING_RATE}")
print(f"Train Images    : {len(train_loader.dataset)}")
print(f"Test Images     : {len(test_loader.dataset)}")
print(f"Classes         : {train_loader.dataset.classes}")
print("=" * 50)

# ===========================
# History
# ===========================
train_losses = []
train_accuracies = []

val_losses = []
val_accuracies = []

# ===========================
# Training
# ===========================
for epoch in range(EPOCHS):

    model.train()

    running_loss = 0
    correct = 0
    total = 0

    progress_bar = tqdm(
        train_loader,
        desc=f"Epoch {epoch+1}/{EPOCHS}"
    )

    for images, labels in progress_bar:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()

        progress_bar.set_postfix(
            loss=f"{loss.item():.4f}",
            acc=f"{100*correct/total:.2f}%"
        )

    train_loss = running_loss / len(train_loader)
    train_acc = 100 * correct / total

    train_losses.append(train_loss)
    train_accuracies.append(train_acc)

    # ===========================
    # Validation
    # ===========================
    model.eval()

    val_loss = 0
    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            loss = criterion(outputs, labels)

            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)

            val_total += labels.size(0)

            val_correct += (predicted == labels).sum().item()

    val_loss /= len(test_loader)
    val_acc = 100 * val_correct / val_total

    val_losses.append(val_loss)
    val_accuracies.append(val_acc)

    scheduler.step()

    early_stopping(val_loss, model)

    print()

    print(f"Epoch {epoch+1}/{EPOCHS}")

    print(f"Train Loss : {train_loss:.4f}")

    print(f"Train Acc  : {train_acc:.2f}%")

    print(f"Val Loss   : {val_loss:.4f}")

    print(f"Val Acc    : {val_acc:.2f}%")

    print("-" * 50)

    if early_stopping.stop:

        print("Early stopping triggered.")

        break

# ===========================
# Save Graphs
# ===========================
plot_curves(
    train_losses,
    train_accuracies,
    val_losses,
    val_accuracies
)

print("\nTraining Completed!")

print(f"Best model saved to: {CHECKPOINT_PATH}")