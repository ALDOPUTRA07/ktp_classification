"""
data_manager.py
====================================
This script uses for loading data.
"""

import os
from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets

from ktp_classification.model import auto_transform, load_pretrained_model


def load_data():
    """Loading data based on data at data folder.

    Loading and splitting data based on data at data folder to
    train, valid, and test data.

    Returns
    -------
    train_data : datasets
        data for training model.

    valid_data : datasets
        data for validating model.

    test_data : datasets
        data for splitting model.

    """
    # Setup path to data folder
    data_path = Path("data/KTP.v1i.folder")

    # Load data
    data_train = os.path.join(data_path, 'train')
    data_validation = os.path.join(data_path, 'valid')
    data_test = os.path.join(data_path, 'test')

    # Load and transform data
    weights, model = load_pretrained_model()
    auto_transforms = auto_transform(
        weights
    )  # Getting auto transform by pretrained model

    # Splitting data
    train_data = datasets.ImageFolder(
        root=data_train, transform=auto_transforms, target_transform=None
    )

    valid_data = datasets.ImageFolder(root=data_validation, transform=auto_transforms)

    test_data = datasets.ImageFolder(root=data_test, transform=auto_transforms)

    return train_data, valid_data, test_data


def class_name(data):
    """Getting classes of data.

    Parameters
    ----------
    data : dataframe
        Image dataset.

    Returns
    -------
    class_names : list
        classes name of data.

    """
    # Getting classes of data
    class_names = data.classes

    return class_names


def data_loader(batch, num_worker):
    """Loading data with dataloader.

    Parameters
    ----------
    batch : int
        Batch of data.

    num_worker : int

    Returns
    -------
    train_dataloader : dataloader
        Data loader for training model.

    valid_dataloader : dataloader
        Data loader for validating model.

    """
    # Load data
    train_data, valid_data, _ = load_data()

    # Load data with dataloader
    train_dataloader = DataLoader(
        dataset=train_data, batch_size=batch, num_workers=num_worker, shuffle=True
    )

    valid_dataloader = DataLoader(
        dataset=valid_data, batch_size=batch, num_workers=num_worker, shuffle=False
    )

    return train_dataloader, valid_dataloader
