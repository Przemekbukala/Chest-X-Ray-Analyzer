import os
import logging
import torch
from typing import List
import matplotlib.pyplot as plt


def save_model(model, run_dir: str) -> str:
    model_path = os.path.join(run_dir, "model.pth")
    torch.save(model.state_dict(), model_path)
    logging.info(f"Model saved to: {model_path}")
    return model_path

def save_training_plots(
        train_losses: List[float],
        val_accuracies: List[float],
        run_dir: str
        ) -> str:
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training loss")
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Validation accuracy")
    plt.grid(True)

    plt.tight_layout()
    plot_path = os.path.join(run_dir, "training_plots.png")
    plt.savefig(plot_path, dpi=150)
    logging.info(f"Training plots saved to: {plot_path}")
    plt.close()
    
    return plot_path

def save_training_metrics(
        train_losses: List[float],
        val_accuracies: List[float],
        epochs: int,
        timestamp: str,
        run_dir: str
        ) -> str:
    metrics_path = os.path.join(run_dir, "metrics.txt")
    
    with open(metrics_path, "w") as f:
        f.write(f"Training Results - {timestamp}\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Number of epochs: {epochs}\n")
        f.write(f"Final training loss: {train_losses[-1]:.4f}\n")
        f.write(f"Final validation accuracy: {val_accuracies[-1]:.4f}\n\n")
        f.write("Epoch-by-epoch metrics:\n")
        f.write("-" * 50 + "\n")
        for i, (loss, acc) in enumerate(zip(train_losses, val_accuracies), 1):
            f.write(f"Epoch {i}: Loss={loss:.4f}, Val_Acc={acc:.4f}\n")
    
    logging.info(f"Metrics saved to: {metrics_path}")
    return metrics_path
