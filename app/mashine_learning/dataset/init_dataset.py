import os
import kagglehub
import logging

def download_dataset() -> str:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    data_dir = os.path.join(project_root, 'data')
    target_dir = os.path.join(data_dir, 'chest_xray')

    if os.path.exists(target_dir):
        logging.info(f'Dataset already exists: {target_dir}')
        return target_dir

    logging.info('Downloading dataset...')
    kaggle_path = kagglehub.dataset_download('muhammadrehan00/chest-xray-dataset')

    os.makedirs(data_dir, exist_ok = True)
    os.rename(kaggle_path, target_dir)

    logging.info(f'Dataset downloaded to: {target_dir}')
    return target_dir
