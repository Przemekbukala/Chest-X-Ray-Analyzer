import sys
import torch
import logging
from models.simple_cnn import SimpleCNN
from training.trainer import train_model
from helpers.model_utils import parse_model_path, get_device
from helpers.data_utils import get_dataloaders

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] : %(message)s')

    try:
        model_path = parse_model_path(sys.argv)
        device = get_device()
        train_loader, val_loader, _ = get_dataloaders(batch_size=16)
        model = SimpleCNN(num_classes=3)
        model.to(device)
        
        if model_path:
            model.load_state_dict(torch.load(model_path, map_location=device))
            logging.info("Model loaded successfully - skipping training")
        else:
            logging.info("Training new model...")
            criterion = torch.nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            train_model(model, train_loader, val_loader, criterion, optimizer, device)
            
    except Exception as e:
        logging.error(f"Training failed: {e}")
        sys.exit(1)
