import torch
from torch.utils.data import DataLoader
import logging
from ..dataset.chest_xray_dataset import ChestXrayDataset
from .transforms import train_transforms, test_transforms

def get_dataloaders(batch_size=32):
    train_ds = ChestXrayDataset(split="train", transform=train_transforms)
    val_ds   = ChestXrayDataset(split="val",   transform=test_transforms)
    test_ds  = ChestXrayDataset(split="test",  transform=test_transforms)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader, test_loader

def train_model(model, train_loader, val_loader, epochs=5, device="cpu"):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            preds = model(images)
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()

        logging.info(f"Epoch {epoch+1}/{epochs} - train loss: {loss.item():.4f}")

    torch.save(model.state_dict(), "model.pth")
    logging.info("Model saved to model.pth")
    return model
