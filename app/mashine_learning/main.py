import logging
from dataset import download_dataset

if __name__ == "__main__":
    logging.basicConfig(level = logging.DEBUG)

    download_dataset()