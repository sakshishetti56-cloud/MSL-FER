import torch

from utils.metrics import accuracy


def train_one_epoch(model,
                    loader,
                    optimizer,
                    criterion,
                    device):

    model.train()

    running_loss = 0
    running_acc = 0

    for images, labels in loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        running_acc += accuracy(outputs, labels)

    epoch_loss = running_loss / len(loader)
    epoch_acc = running_acc / len(loader)

    return epoch_loss, epoch_acc