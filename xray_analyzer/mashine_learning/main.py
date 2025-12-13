import os
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import logging
from dataset.chest_xray_dataset import ChestXrayDataset
from models.simple_cnn import SimpleCNN
from training.trainer import train_model

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    ROOT = os.path.abspath("../data/chest_xray")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    train_ds = ChestXrayDataset(ROOT, "train", transform)
    val_ds   = ChestXrayDataset(ROOT, "val",   transform)

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=16)

    model = SimpleCNN(num_classes=3)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    train_model(model, train_loader, val_loader, criterion, optimizer, device)
