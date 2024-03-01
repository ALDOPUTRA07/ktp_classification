from pathlib import Path

import torch

# Number of epoch for training model
epochs = int(5)

# Define target device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Define project name
PROJECT_NAME = "KTP Classification"

# Setuo for saving model
MODEL_PATH = Path("ktp_classification/train_model/")
MODEL_NAME = "03_pytorch_computer_vision_model_2.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME
