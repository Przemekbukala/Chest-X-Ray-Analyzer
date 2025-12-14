import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import logging
from datetime import datetime

def train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs=3, save_dir="runs") -> None:
    model.to(device)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(save_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    logging.info(f"Saving training results to: {run_dir}")

    train_losses = []
    val_accuracies = []

    for epoch in range(epochs):
        logging.info(f"EPOCH {epoch+1}/{epochs}")

        model.train()
        total_loss = 0

        for images, labels in tqdm(train_loader, desc="Training"):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        logging.info(f"Train loss: {total_loss/len(train_loader):.4f}")

        model.eval()
        correct, total = 0.0, 0.0

        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Validation"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, preds = outputs.max(1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()

        train_losses.append(total_loss / len(train_loader))
        val_accuracies.append(correct / total)

        logging.info(f"Validation accuracy: {correct/total:.2f}")

    model_path = os.path.join(run_dir, "model.pth")
    torch.save(model.state_dict(), model_path)
    logging.info(f"Model saved to: {model_path}")

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
