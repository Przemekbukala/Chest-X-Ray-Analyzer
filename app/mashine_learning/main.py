import logging
from torchvision import transforms
from dataset import ChestXrayDataset

if __name__ == '__main__':
    logging.basicConfig(level = logging.DEBUG)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    train_dataset = ChestXrayDataset(split='train', transform=transform)
    logging.info(f'Number of training samples: {len(train_dataset)}')

    image, label = train_dataset[0]
    logging.info(f'train_dataset[0] - image.shape: {image.shape}, label: {label}')