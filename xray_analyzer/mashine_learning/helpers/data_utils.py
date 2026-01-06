import logging
from typing import Tuple
from torchvision import transforms
from torch.utils.data import DataLoader
from ..dataset.chest_xray_dataset import ChestXrayDataset
from ..dataset.init_dataset import download_dataset


def get_dataloaders(
        batch_size: int = 16,
        num_workers: int = 0
        ) -> Tuple[DataLoader, DataLoader, DataLoader]:
    try:
        dataset_path = download_dataset()
        
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])
        
        train_ds = ChestXrayDataset(dataset_path, "train", transform)
        val_ds = ChestXrayDataset(dataset_path, "val", transform)
        test_ds = ChestXrayDataset(dataset_path, "test", transform)
        
        logging.info(f"Dataset loaded: {len(train_ds)} train, {len(val_ds)} val, {len(test_ds)} test images")
        
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        
        return train_loader, val_loader, test_loader
        
    except Exception as e:
        logging.error(f"Failed to load dataset: {e}")
        raise RuntimeError(f"Dataset loading failed: {e}")
