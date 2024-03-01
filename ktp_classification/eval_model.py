"""
eval_model.py
====================================
This script uses for evaluating model.
"""

import torch

from ktp_classification.config.config import device


def eval_model(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    accuracy_fn,
    device: torch.device = device,
):
    """Evaluates a given model on a given dataset.

    Parameters
    ----------
    model : torch.nn.Module
        A PyTorch model capable of making predictions on data_loader.

    data_loader : torch.utils.data.DataLoader
        The target dataset to predict on.

    loss_fn : torch.nn.Module
        The loss function of model.

    accuracy_fn:
        An accuracy function to compare the models predictions to the truth labels.

    device : str, optional
        Target device to compute on. Defaults to device

    Returns
    -------
    dict
        Results of model making predictions on data_loader.

    """
    loss, acc = 0, 0
    model.eval()
    with torch.inference_mode():
        for X, y in data_loader:
            # Send data to the target device
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            loss += loss_fn(y_pred, y)
            acc += accuracy_fn(y_true=y, y_pred=y_pred.argmax(dim=1))

        # Scale loss and acc
        loss /= len(data_loader)
        acc /= len(data_loader)
    return {
        "model_name": model.__class__.__name__,  # only works when model
        # was created with a class
        "model_loss": loss.item(),
        "model_acc": acc,
    }


def print_train_time(start: float, end: float, device: torch.device = None):
    """Prints difference between start and end time.

    Parameters
    ----------
    model : torch.nn.Module
        A PyTorch model capable of making predictions on data_loader.

    start : float
        Start time of computation (preferred in timeit format).

    end : float
        End time of computation.

    device : [type], optional
        Device that compute is running on. Defaults to None.

    Returns
    -------
    total_time : float
        time between start and end in seconds (higher is longer).

    """
    total_time = end - start
    return total_time
