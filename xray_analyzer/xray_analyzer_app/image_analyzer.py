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
import cv2
import numpy as np

from mashine_learning.helpers.model_utils import *
from mashine_learning.config import  *
from . import model_loader
#import model_loader
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
        self.img_tensor = None  
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

    def _get_target_conv_layer(self) -> nn.Module:
        last_conv = None
        for m in self.model.modules():
            if isinstance(m, nn.Conv2d):
                last_conv = m
        if last_conv is None:
            raise RuntimeError("No Conv2d layer found in model for Grad-CAM.")
        return last_conv


    def _get_lung_mask(self, img_bgr: np.ndarray) -> np.ndarray:
        """
        Heuristic lung field mask from CXR using OpenCV.
        Returns uint8 mask: 0/255, same HxW as image.
        """
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        g = clahe.apply(gray)
        g = cv2.GaussianBlur(g, (5, 5), 0)

        def build_mask(thresh_type):
            _, th = cv2.threshold(g, 0, 255, thresh_type | cv2.THRESH_OTSU)

            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
            th = cv2.morphologyEx(th, cv2.MORPH_OPEN, k, iterations=1)
            th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, k, iterations=2)

            h, w = th.shape
            flood = th.copy()
            mask_ff = np.zeros((h + 2, w + 2), np.uint8)
            cv2.floodFill(flood, mask_ff, (0, 0), 255)
            flood_inv = cv2.bitwise_not(flood)
            filled = th | flood_inv

            num, labels, stats, _ = cv2.connectedComponentsWithStats(filled, connectivity=8)
            if num <= 1:
                return np.zeros_like(filled)

            areas = stats[1:, cv2.CC_STAT_AREA]
            idx = np.argsort(areas)[::-1]  
            keep = idx[:2] + 1  

            comp = np.zeros_like(filled)
            for kidx in keep:
                comp[labels == kidx] = 255

            comp = cv2.morphologyEx(comp, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)), iterations=1)
            return comp

        m1 = build_mask(cv2.THRESH_BINARY_INV)
        m2 = build_mask(cv2.THRESH_BINARY)

        h, w = gray.shape
        area1 = cv2.countNonZero(m1) / float(h * w)
        area2 = cv2.countNonZero(m2) / float(h * w)

        def score(area):
            if 0.08 <= area <= 0.60:
                return 1.0 - abs(area - 0.28)  
            return -1.0

        mask = m1 if score(area1) >= score(area2) else m2
        return mask

    
    def get_conv2_feature_map(self, class_idx: int | None = None) -> None:
        if self.img_tensor is None:
            raise ValueError("img_tensor is None. Call analyze(image_path) first.")

        self.activations = None
        self.gradients = None

        target_layer = self._get_target_conv_layer()

        forward_hook = target_layer.register_forward_hook(self._forward_hook, prepend=False)
        backward_hook = target_layer.register_full_backward_hook(self._backward_hook, prepend=False)

        self.model.zero_grad(set_to_none=True)

        out = self.model(self.img_tensor) 

        if class_idx is None:
            class_idx = int(out.argmax(dim=1).item())

        score = out[:, class_idx].sum()
        score.backward()

        forward_hook.remove()
        backward_hook.remove()

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM failed: no activations/gradients captured.")




    def compute_heatmap(self, output_path: str):
        if self.img_tensor is None:
            raise ValueError("img_tensor is None. Call analyze(image_path) first.")

        self.get_conv2_feature_map()  

        activations = self.activations.detach().cpu().numpy()[0]  # [C, h, w]
        gradients = self.gradients.detach().cpu().numpy()[0]      # [C, h, w]

        weights = np.mean(gradients, axis=(1, 2))  # [C]
        cam = np.zeros(activations.shape[1:], dtype=np.float32)   # [h, w]
        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = np.maximum(cam, 0)
        if cam.max() > 1e-8:
            cam = cam / cam.max()

        if cam.max() < 1e-4:
            self.model.zero_grad(set_to_none=True)

            inp = self.img_tensor.clone().detach().requires_grad_(True)
            out2 = self.model(inp)
            class_idx = int(out2.argmax(dim=1).item())
            score2 = out2[:, class_idx].sum()
            score2.backward()

            sal = inp.grad.detach().abs()[0].mean(dim=0).cpu().numpy()
            sal = sal - sal.min()
            if sal.max() > 1e-8:
                sal = sal / sal.max()
            cam = sal.astype(np.float32)

        cam = np.power(cam, 2.0)

        img = self.img_tensor[0].detach().cpu().permute(1, 2, 0).numpy()
        img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        height, width = img_bgr.shape[:2]

        cam_resized = cv2.resize(cam, (width, height), interpolation=cv2.INTER_LINEAR)
        cam_resized = cv2.GaussianBlur(cam_resized, (15, 15), 0)

        lung_mask = self._get_lung_mask(img_bgr)  # 0/255, HxW
        lung_mask_f = lung_mask.astype(np.float32) / 255.0

        cam_lung = cam_resized * lung_mask_f

        if np.any(lung_mask_f > 0):
            vals = cam_lung[lung_mask_f > 0]
            p_low, p_high = np.percentile(vals, [50, 99.5])
            cam_lung = (cam_lung - p_low) / (p_high - p_low + 1e-8)
            cam_lung = np.clip(cam_lung, 0, 1)
        else:
            mx = cam_lung.max()
            if mx > 1e-8:
                cam_lung = cam_lung / mx
            cam_lung = np.clip(cam_lung, 0, 1)
        cam_lung = np.where(cam_lung < 0.2, 0.0, cam_lung)



        heatmap_color = cv2.applyColorMap((cam_lung * 255).astype(np.uint8), cv2.COLORMAP_JET)

        alpha = 0.45
        result = img_bgr.copy().astype(np.float32)

        a = (alpha * cam_lung).astype(np.float32)
        a3 = np.dstack([a, a, a])

        result = result * (1.0 - a3) + heatmap_color.astype(np.float32) * a3

        inv = (1.0 - lung_mask_f)
        inv3 = np.dstack([inv, inv, inv])
        result = result * (1.0 - inv3) + img_bgr.astype(np.float32) * inv3

        result = np.clip(result, 0, 255).astype(np.uint8)
        cv2.imwrite(output_path, result)
        return output_path




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