import random
from pathlib import Path

import pytest
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

from ktp_classification.processing.data_manager import load_data


@pytest.fixture
def load_dataset():
    # load train_data, valid_data, test_data
    train_data, valid_data, test_data = load_data()

    return train_data


@pytest.fixture
def load_pretrained_model():
    # Setup pretrained model
    weights = EfficientNet_B0_Weights.DEFAULT
    model = efficientnet_b0(weights=weights)

    # Freeze the base layers in the model (this will freeze all layers to begin with)
    for param in model.parameters():
        param.requires_grad = False

    return model


@pytest.fixture
def load_weight_model():
    # Setup pretrained model
    weights = EfficientNet_B0_Weights.DEFAULT

    return weights
