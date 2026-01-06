import os
import torch
import logging
from PIL import Image
from sympy.stats.rv import probability
from torchvision import transforms

from mashine_learning.helpers.model_utils import *
from mashine_learning.config import  *
from . import model_loader
# import model_loader
logger = logging.getLogger(__name__)

class ImageAnalizer():
    """
    Class responsible for analyzing chest X-ray images and making Heatmap.
    This class is using a pre-trained  Learning model to classify images into
    three categories: normal, pneumonia, and tuberculosis.
    Used for example in views.py/upload_xray
    """
    def __init__(self, model_filename=None):
        self.device = get_device()
        self.model = model_loader.load_model()
        self.output = None

    def analyze(self, image_path) ->  dict[str, float]:
        self.model.eval()
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])

        try:
            img = Image.open(image_path).convert('RGB')
        except Exception as e:
            logger.error(f"Cannot open image {image_path}: {e}")
            return None
        img_tensor = transform(img).unsqueeze(0).to(self.device)
        with (torch.no_grad()):
            self.output = self.model(img_tensor)
            probabilities = torch.softmax(self.output, dim=1)
            class_names = ["normal", "pneumonia", "tuberculosis"]
            confidence, predicted_idx = torch.max(probabilities, dim=1)
            result = {
                "predicted_class": class_names[predicted_idx.item()],
                "confidence": confidence.item(),
                "probabilities": {name: round(float(prob) *100,2)  for name, prob in zip(class_names, probabilities[0])}
            }

            logger.info(f"Analysis result: {result}")
            return result


if __name__ == "__main__":
    # print(MEDIA_DIR)
    # logging.basicConfig(level=logging.INFO)
    # test_img = f"{MEDIA_DIR}/"+"xrays/2026/01/06/normal-96.jpg"
    # if os.path.exists(test_img):
    #     print(f"Testing on: {test_img}")
    #     try:
    #         analyzer = ImageAnalizer()
    #         print(analyzer.analyze(test_img))
    #     except Exception as e:
    #         print(f"Error: {e}")
    # else:
    #     logging.warning(f"Test image not found at: {test_img}")
    pass