import os
from pathlib import Path

FILE_PATH = Path(__file__).resolve()

ML_PACKAGE_DIR = FILE_PATH.parent

DJANGO_APP_DIR = ML_PACKAGE_DIR.parent

WORKSPACE_ROOT = DJANGO_APP_DIR.parent


MODELS_DIR = WORKSPACE_ROOT / "runs"

MEDIA_DIR = DJANGO_APP_DIR / "media"

DATA_DIR = WORKSPACE_ROOT / "data" / "chest_xray"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
MEDIA_DIR.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    print(f"Workspace Root: {WORKSPACE_ROOT}")
    print(f"Django:     {DJANGO_APP_DIR}")
    print(f"Models Dir:     {MODELS_DIR}")