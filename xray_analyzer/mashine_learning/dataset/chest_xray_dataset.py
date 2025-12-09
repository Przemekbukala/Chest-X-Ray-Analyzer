import os
from PIL import Image
from typing import Tuple
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class ChestXrayDataset(Dataset):
    def __init__(self, root_dir, split="train", transform=None):
        self.transform = transform
        self.class_names = {"normal": 0, "pneumonia": 1, "tuberculosis": 2}
        self.image_paths = []
        self.labels = []

        split_dir = os.path.join(root_dir, split)
        for class_name, label in self.class_names.items():
            class_dir = os.path.join(split_dir, class_name)
            if not os.path.exists(class_dir):
                continue
            for fname in os.listdir(class_dir):
                if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                    self.image_paths.append(os.path.join(class_dir, fname))
                    self.labels.append(label)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label
