import torch

class EarlyStopping:

    def __init__(self, patience=5):
        self.patience = patience
        self.best_loss = float("inf")
        self.counter = 0
        self.stop = False

    def __call__(self, loss, model):

        if loss < self.best_loss:
            self.best_loss = loss
            self.counter = 0
            torch.save(model.state_dict(), "checkpoints/best_model.pth")

        else:
            self.counter += 1

            if self.counter >= self.patience:
                self.stop = True