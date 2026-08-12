import matplotlib.pyplot as plt
import numpy as np

def plot_learning_curves(history, config, file_name='learning_curve'):
    epochs = range(1, len(history["train_losses"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].plot(epochs, history["train_losses"],label="Training loss")
    axes[0].plot(epochs, history["val_losses"],label="Validation loss")
    axes[0].set_title("NN: Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    
    axes[1].plot(epochs, history["train_accuracies"],label="Training accuracy")
    axes[1].plot(epochs, history["val_accuracies"],label="Validation accuracy")
    axes[1].set_title("NN: Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(f"{config.plot_dir}/{file_name}.png", dpi=150)
    plt.show()

