"""
predict.py
====================================
This script uses for predicting image.
"""

import io
from timeit import default_timer as timer
from typing import List, Tuple

import torch
from PIL import Image
from torch import nn
from torchvision import transforms
from torchvision.models import efficientnet_b0

from ktp_classification.config.config import device
from ktp_classification.eval_model import print_train_time


def load_model(MODEL_SAVE_PATH, device):
    """Loading model with weights after training.

    Parameters
    ----------
    MODEL_SAVE_PATH : Path
        Path for saving the models.

    device : [type], optional
        Device that compute is running on. Defaults to None.

    Returns
    -------
    loaded_model : torch.nn.Module
        Model with weight after training.

    """
    # Setup pretrained model
    loaded_model = efficientnet_b0()

    # Updating classifier head
    loaded_model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True), nn.Linear(in_features=1280, out_features=2)
    )

    # Load weight of pretrained model
    loaded_model.load_state_dict(torch.load(f=MODEL_SAVE_PATH, map_location=device))

    return loaded_model


def pred_image(
    model: torch.nn.Module,
    image_path: str,
    class_names: List[str],
    image_size: Tuple[int, int] = (224, 224),
    transform: transforms = None,
    device: torch.device = device,
):
    """Making a prediction on a target image.

    Parameters
    ----------
    model : torch.nn.Module
        Model with weight after training.

    image_path : str
        Path of target image.

    class_names : List
        Name class of classification ['KTP', 'Non-KTP'].

    Image_size : Tuple
        Size of image.

    transform
       transforms used to create our pretrained weights.

    device : [type], optional
        Device that compute is running on. Defaults to None.

    Returns
    -------
    predict_label : str
        Predict result of classify image.

    prob_label : float
        Probability of predict result.

    total_train_time : float
        time between start and end in seconds (higher is longer).

    img
        Image data.

    """
    # Starting to calculate time
    train_time_start = timer()

    # Open image
    img = Image.open(io.BytesIO(image_path)).convert("RGB")

    # Create transformation for image
    if transform is not None:
        image_transform = transform
    else:
        image_transform = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    # Make sure the model is on the target device
    model.to(device)

    # Turn on model evaluation mode and inference mode
    model.eval()
    with torch.inference_mode():
        # Transform and add an extra dimension to image (model requires
        # samples in [batch_size, color_channels, height, width])
        transformed_image = image_transform(img).unsqueeze(dim=0)

        # Make a prediction on image with an extra dimension
        # and send it to the target device
        target_image_pred = model(transformed_image.to(device))

    # Convert logits -> prediction probabilities (using torch.softmax()
    # for multi-class classification)
    target_image_pred_probs = torch.softmax(target_image_pred, dim=1)

    # Convert prediction probabilities -> prediction labels
    target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)

    # Get class name of predicition
    predict_label = class_names[target_image_pred_label]
    prob_label = target_image_pred_probs.max()

    # Stop calculating time
    train_time_end = timer()
    total_train_time = print_train_time(
        start=train_time_start, end=train_time_end, device=device
    )

    return predict_label, float(prob_label), total_train_time, img
