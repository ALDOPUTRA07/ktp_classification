"""
utils.py
====================================
This script uses for loading data.
"""

from pathlib import Path

import requests
import torch
from torch import nn


def download_helper():
    """Downloading helper function for accuracy_fn function."""
    # Download helper functions from Learn PyTorch repo (if not already downloaded)
    if Path("helper_functions.py").is_file():
        print("helper_functions.py already exists, skipping download")
    else:
        print("Downloading helper_functions.py")
    # Note: you need the "raw" GitHub URL for this to work
    request = requests.get(
        "https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/helper_functions.py"
    )
    with open("helper_functions.py", "wb") as f:
        f.write(request.content)


def loss_accuracy(model, learn_rate):
    """Loading data with dataloader.

    Parameters
    ----------
    model
        pretrained model.

    learn_rate : float
        Learning rate for training model.

    """
    # Setup loss function and optimizer
    loss_fn = (
        nn.CrossEntropyLoss()
    )  # nn.BCELoss() # this is also called "criterion"/"cost function" in some places
    optimizer = torch.optim.SGD(params=model.parameters(), lr=learn_rate)

    return loss_fn, optimizer
