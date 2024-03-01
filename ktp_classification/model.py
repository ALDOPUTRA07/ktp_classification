"""
model.py
====================================
This script uses for loading pretrained model and saving model.
"""

import torch
from torch import nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0


def load_pretrained_model():
    """Loading model pretrained.

    Returns
    -------
    weight
        Weights of pretrained model.

    model : torch.nn.Module
        Pretrained model.
    """
    # Setup pretrained model
    weights = EfficientNet_B0_Weights.DEFAULT
    model = efficientnet_b0(weights=weights)

    # Freeze the base layers in the model (this will freeze all layers to begin with)
    for param in model.parameters():
        param.requires_grad = False

    return weights, model


def model_classifier(model, p, infeatures, outfeatures):
    """Update the classifier head.

    Parameters
    ----------
    model : torch.nn.Module
        Pretrained model.

    p :  float
        Probability for dropout paramaters.

    infeatures : int
        Number of in features.

    outfeatures : int
        Number of out features.

    Returns
    -------
    model : torch.nn.Module
        Model after updating classifier head.

    """
    # Update the classifier head
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features=infeatures, out_features=outfeatures),
    )

    return model


def auto_transform(weights):
    """Get the transforms used to create pretrained weights.

    Parameters
    ----------
    weights
        Woeights of pretrained model.

    Returns
    -------
    transform
       transforms used to create our pretrained weights.

    """
    # Get the transforms used to create our pretrained weights
    auto_transforms = weights.transforms()

    return auto_transforms


def save_model(model, MODEL_SAVE_PATH):
    """Saving the model at MODEL_SAVE_PATH.

    Parameters
    ----------
    model : torch.nn.Module
        Pretrained model.

    MODEL_SAVE_PATH : Path
        Path for saving the models.

    Returns
    -------
    Saving the model at MODEL_SAVE_PATH.

    """
    # Save the model state dict
    print(f"Saving model to: {MODEL_SAVE_PATH}")
    torch.save(
        obj=model.state_dict(),  # only saving the state_dict() only
        # saves the learned parameters
        f=MODEL_SAVE_PATH,
    )
