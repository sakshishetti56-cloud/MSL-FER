import matplotlib.pyplot as plt

def plot_curves(train_loss, train_acc, val_loss, val_acc):

    plt.figure(figsize=(8,5))

    plt.plot(train_loss, label="Train Loss")
    plt.plot(val_loss, label="Validation Loss")

    plt.legend()
    plt.savefig("outputs/loss.png")
    plt.close()

    plt.figure(figsize=(8,5))

    plt.plot(train_acc, label="Train Accuracy")
    plt.plot(val_acc, label="Validation Accuracy")

    plt.legend()
    plt.savefig("outputs/accuracy.png")
    plt.close()