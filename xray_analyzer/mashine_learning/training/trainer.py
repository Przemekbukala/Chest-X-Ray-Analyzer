import os
import torch
from typing import List, Tuple
from torch.utils.data import DataLoader
from torch.nn import Module
from torch.optim import Optimizer
from tqdm import tqdm
import logging
from datetime import datetime
from helpers.training_utils import save_model, save_training_plots, save_training_metrics


def train_epoch(
        model: Module,
        train_loader: DataLoader,
        criterion: Module,
        optimizer: Optimizer,
        device: torch.device
        ) -> float:
    model.train()
    total_loss = 0.0

    for images, labels in tqdm(train_loader, desc="Training"):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)

def validate_epoch(
        model: Module,
        val_loader: DataLoader,
        device: torch.device
        ) -> Tuple[float, int, int]:
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validation"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = outputs.max(1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

    accuracy = correct / total if total > 0 else 0.0
    return accuracy, correct, total

def train_model(
        model: Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: Module,
        optimizer: Optimizer,
        device: torch.device,
        epochs: int = 3,
        save_dir: str = "runs"
        ) -> None:
    model.to(device)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(save_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    logging.info(f"Saving training results to: {run_dir}")

    train_losses: List[float] = []
    val_accuracies: List[float] = []
    
    for epoch in range(epochs):
        logging.info(f"EPOCH {epoch+1}/{epochs}")

        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        logging.info(f"Train loss: {train_loss:.4f}")

        val_accuracy, correct, total = validate_epoch(model, val_loader, device)
        val_accuracies.append(val_accuracy)
        logging.info(f"Validation accuracy: {val_accuracy:.4f} ({correct}/{total})")

    save_model(model, run_dir)
    save_training_plots(train_losses, val_accuracies, run_dir)
    save_training_metrics(train_losses, val_accuracies, epochs, timestamp, run_dir)
