import os
import torch
import logging
from PIL import Image
from sympy.stats.rv import probability
from torch import nn
from torchvision import transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_pil_image, resize

from mashine_learning.helpers.model_utils import *
from mashine_learning.config import  *
# from . import model_loader
import model_loader
logger = logging.getLogger(__name__)

class ImageAnalizer():
    """
    Class responsible for analyzing chest X-ray images and making Heatmap.
    This class is using a pre-trained  Learning model to classify images into
    three categories: normal, pneumonia, and tuberculosis.
    output is a PyTorch tensor  that represents an image in a format understood by a neural network model.
    Used for example in views.py/upload_xray
    """
    def __init__(self, model_filename=None):
        self.device = get_device()
        self.model = model_loader.load_model()
        self.output = None
        self.transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
        self.activations = None
        self.gradients = None
        self.img_tensor = None  # stores the tensor that represents the image
        self.model.eval()

    def analyze(self, image_path) ->  dict[str, float]:
        """
        Method for analyzing the probability of:
        - Healthy lungs
        - Pneumonia infection
        - Tuberculosis infection
        """
        try:
            img = Image.open(image_path).convert('RGB')
        except Exception as e:
            logger.error(f"Cannot open image {image_path}: {e}")
            return None
        self.img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            self.output = self.model(self.img_tensor)
            probabilities = torch.softmax(self.output, dim=1)
            class_names = ["normal", "pneumonia", "tuberculosis"]
            result = {name: round(float(prob) *100,2)  for name, prob in zip(class_names, probabilities[0])}
            logger.info(f"Analysis result: {result}")
            return result

    def _forward_hook(self, module, inp, out):
        print('Forward hook running...')
        self.activations = out
        print(f'Activations size: {self.activations.size()}')

    def _backward_hook(self, module, grad_in, grad_out):
        print('Backward hook running...')
        self.gradients = grad_out[0]
        print(f'Gradients size: {self.gradients.size()}')

    def get_conv2_feature_map(self)->torch.Tensor:
        """getting feature maps from  layer conv2"""
        backward_hook = self.model.conv2.register_full_backward_hook(self._backward_hook, prepend=False)
        forward_hook = self.model.conv2.register_forward_hook(self._forward_hook, prepend=False)
        out = self.model(self.img_tensor)
        score = out[:, out.argmax(dim=1)]
        score.backward()
        forward_hook.remove()
        backward_hook.remove()

    def compute_heatmap(self):
        self.get_conv2_feature_map()
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        print(f"Pooled gradients size: {pooled_gradients.size()}")
        print(f"Pooled gradients size: {pooled_gradients}")

        for i in range(self.activations.size()[1]):
            self.activations[:, i, :, :] *= pooled_gradients[i]

        heatmap = torch.mean(self.activations, dim=1).squeeze()
        heatmap = F.relu(heatmap)
        # normalize the heatmap
        heatmap /= torch.max(heatmap)
        plt.matshow(heatmap.detach())
        fig, ax = plt.subplots()
        ax.axis('off')
        ax.imshow(to_pil_image(self.img_tensor[0], mode='RGB'))
        overlay = to_pil_image(heatmap.detach(), mode='F').resize((256, 256), resample=PIL.Image.BICUBIC)
        cmap = plt.colormaps['jet']
        overlay = (255 * cmap(np.asarray(overlay) ** 2)[:, :, :3]).astype(np.uint8)
        ax.imshow(overlay, alpha=0.4, interpolation='nearest')
        plt.show()


if __name__ == "__main__":
    # print(MEDIA_DIR)
    # logging.basicConfig(level=logging.INFO)
    # test_img = f"{MEDIA_DIR}/"+"xrays/2026/01/06/tuberculosis-900.jpg"
    # analyzer = None
    # if os.path.exists(test_img):
    #     print(f"Testing on: {test_img}")
    #     try:
    #         analyzer = ImageAnalizer()
    #         print(analyzer.analyze(test_img))
    #     except Exception as e:
    #         print(f"Error: {e}")
    # else:
    #     logging.warning(f"Test image not found at: {test_img}")
    # analyzer.compute_heatmap()
    pass