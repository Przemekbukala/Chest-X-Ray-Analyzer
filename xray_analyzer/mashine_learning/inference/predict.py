import torch
from PIL import Image
from ..models.classifier import create_model
from ..training.transforms import test_transforms

CLASS_NAMES = ["normal", "pneumonia", "tuberculosis"]

def predict(image_path: str) -> str:
    model = create_model()
    model.load_state_dict(torch.load("model.pth", map_location="cpu"))
    model.eval()

    img = Image.open(image_path).convert("RGB")
    img = test_transforms(img).unsqueeze(0)

    with torch.no_grad():
        preds = model(img)
        label = preds.argmax(1).item()

    return CLASS_NAMES[label]
