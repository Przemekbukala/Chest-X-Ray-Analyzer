import torch
import logging
from mashine_learning.models.simple_cnn import SimpleCNN
from mashine_learning.helpers.model_utils import get_device

logger = logging.getLogger(__name__)
MODEL = None

def load_model():
    global MODEL

    if MODEL is not None:
        return MODEL

    try:
        model_path = 'model.pth'

        device = get_device()
        model = SimpleCNN(num_classes=3).to(device)

        if model_path:
            model.load_state_dict(torch.load(model_path, map_location=device))
            logger.info("Model loaded successfully")
        else:
            logger.info("No model path provided")

        MODEL = model
        return MODEL


    except Exception:
        logger.exception("Model initialization failed")
        raise
