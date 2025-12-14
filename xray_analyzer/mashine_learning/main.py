import os
import sys
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import logging
from dataset.chest_xray_dataset import ChestXrayDataset
from dataset.init_dataset import download_dataset
from models.simple_cnn import SimpleCNN
from training.trainer import train_model

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    model_path = None
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
        
        if model_path.startswith("run_"):
            model_path = os.path.join("runs", model_path, "model.pth")
        elif os.path.isdir(model_path):
            model_path = os.path.join(model_path, "model.pth")
        
        if not os.path.exists(model_path):
            logging.error(f"Model file not found: {model_path}")
            sys.exit(1)
        
        logging.info(f"Using existing model: {model_path}")
    else:
        logging.info("No model path provided - training new model")

    dataset_path = download_dataset()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    train_ds = ChestXrayDataset(dataset_path, "train", transform)
    val_ds   = ChestXrayDataset(dataset_path, "val",   transform)

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=16)

    model = SimpleCNN(num_classes=3)

    if model_path:
        model.load_state_dict(torch.load(model_path, map_location=device))
        logging.info("Model loaded successfully")
    else:

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        train_model(model, train_loader, val_loader, criterion, optimizer, device)
