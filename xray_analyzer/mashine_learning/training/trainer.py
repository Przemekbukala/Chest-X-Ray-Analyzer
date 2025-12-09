import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging

def train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs=3):
    model.to(device)

    for epoch in range(epochs):
        logging.info(f"\nEPOCH {epoch+1}/{epochs}")

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
        correct, total = 0, 0

        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Validation"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, preds = outputs.max(1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()

        logging.info(f"Validation accuracy: {correct/total:.2f}")

    return model
