#!/usr/bin/env python3

print("""
CHEST X-RAY ANALYZER - ML MODULE HELP
================================================================================

TRAINING (main.py)
--------------------------------------------------------------------------------

  Train new model with default 3 epochs:
    $ python3 main.py

  Train new model with custom number of epochs:
    $ python3 main.py 5
    $ python3 main.py 10

  Load existing model (skip training):
    $ python3 main.py run_[...]
    $ python3 main.py runs/run_[...]
    $ python3 main.py runs/run_[...]/model.pth

  Training results are saved to:
    runs/run_YYYYMMDD_HHMMSS/
      model.pth           - trained model weights
      training_plots.png  - loss and accuracy plots
      metrics.txt         - detailed metrics for each epoch


TESTING (test_model.py)
--------------------------------------------------------------------------------

  Test model on test set:
    $ python3 test_model.py run_[...]
    $ python3 test_model.py runs/run_[...]
    $ python3 test_model.py runs/run_[...]/model.pth

  Outputs:
    - Overall accuracy
    - Per-class accuracy (normal, pneumonia, tuberculosis)
    - Correct predictions count per class


PROJECT STRUCTURE
--------------------------------------------------------------------------------

  dataset/
    chest_xray_dataset.py  - PyTorch Dataset for X-ray images
    init_dataset.py        - Kaggle dataset downloader

  models/
    simple_cnn.py          - Simple CNN model (2 conv layers)

  training/
    trainer.py             - Training and validation logic

  helpers/
    model_utils.py         - Model loading, argument parsing
    data_utils.py          - DataLoader utilities
    training_utils.py      - Save model, plots, metrics

  data/                    - Training/test data (gitignored)
  runs/                    - Training results (gitignored)


DATASET
--------------------------------------------------------------------------------

  Automatically downloaded from Kaggle on first run.
  Structure: data/chest_xray/
    train/     - training set
    val/       - validation set
    test/      - test set

  Each folder contains: normal/, pneumonia/, tuberculosis/


CLASSES
--------------------------------------------------------------------------------

  0: normal       - healthy lungs
  1: pneumonia    - pneumonia infection
  2: tuberculosis - tuberculosis infection


EXAMPLE WORKFLOW
--------------------------------------------------------------------------------

  1. Train model:
     $ python3 main.py 10

  2. Test model:
     $ python3 test_model.py run_[...]

  3. Check results:
     $ cat runs/run_[...]/metrics.txt
     $ open runs/run_[...]/training_plots.png

  4. To be implemented (regarding usage of existing model):
     $ python3 main.py runs/run_[...]/model.pth


REQUIREMENTS
--------------------------------------------------------------------------------

  Python 3.8+
  PyTorch, torchvision, PIL, matplotlib, tqdm, kagglehub

================================================================================
""")
