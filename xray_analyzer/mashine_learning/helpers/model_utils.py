import os
import sys
import logging
import torch
from typing import List, Optional, Tuple
from torch.nn import Module
from mashine_learning.models.simple_cnn import SimpleCNN
from mashine_learning.config import MODELS_DIR

def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    logging.info(f"Using device: {device}")
    return device

def parse_model_path_or_epochs(argv: List[str]) -> Tuple[Optional[str], Optional[int]]:
    if len(argv) <= 1:
        logging.info("No arguments provided - training new model with default epochs")
        return None, None
    
    arg = argv[1]
    
    if arg.isdigit():
        epochs = int(arg)
        logging.info(f"Training new model with {epochs} epochs")
        return None, epochs
    
    model_path = arg
    
    if model_path.startswith("run_"):
        model_path = os.path.join("runs", model_path, "model.pth")
    elif os.path.isdir(model_path):
        model_path = os.path.join(model_path, "model.pth")
    
    if not os.path.exists(model_path):
        logging.error(f"Model file not found: {model_path}")
        sys.exit(1)
    
    logging.info(f"Using model: {model_path}")
    return model_path, None

def parse_model_path(argv: List[str]) -> Optional[str]:
    if len(argv) <= 1:
        logging.info("No model path provided")
        return None
    
    model_path = argv[1]
    
    if model_path.startswith("run_"):
        model_path = os.path.join("runs", model_path, "model.pth")
    elif os.path.isdir(model_path):
        model_path = os.path.join(model_path, "model.pth")
    
    if not os.path.exists(model_path):
        logging.error(f"Model file not found: {model_path}")
        sys.exit(1)
    
    logging.info(f"Using model: {model_path}")
    return model_path


def load_model(
        model_path: str,
        device: torch.device,
        num_classes: int = 3
        ) -> Module:
    try:
        model = SimpleCNN(num_classes=num_classes)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        logging.info("Model loaded successfully")
        return model
    except Exception as e:
        logging.error(f"Failed to load model: {e}")
        raise RuntimeError(f"Model loading failed: {e}")
