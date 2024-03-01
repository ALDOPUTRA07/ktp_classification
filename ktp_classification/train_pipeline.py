"""
train_pipeline.py
====================================
This script uses for training and testing model.
"""

from timeit import default_timer as timer

import torch
from tqdm.auto import tqdm

from ktp_classification.config.config import MODEL_SAVE_PATH, device, epochs
from ktp_classification.eval_model import print_train_time
from ktp_classification.model import load_pretrained_model, model_classifier, save_model
from ktp_classification.processing.data_manager import data_loader
from ktp_classification.processing.utils import download_helper, loss_accuracy


def train_step(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    accuracy_fn,
    device: torch.device = device,
):
    """Training model.

    Parameters
    ----------
    model : torch.nn.Module
        Model with weight after training.

    data_loader : torch.utils.data.DataLoader
        Data loader for training model.

    loss_fn : torch.nn.Module
        Loss function.

    optimizer : torch.optim.Optimizer
        Optimizer for training model.

    accuracy_fn
       Function for calculating accuracy.

    device : [type], optional
        Device that compute is running on. Defaults to None.

    Returns
    -------
    Result of training model

    """
    train_loss, train_acc = 0, 0
    model.to(device)
    for batch, (X, y) in enumerate(data_loader):
        # Send data to GPU
        X, y = X.to(device), y.to(device)

        # Forward pass
        y_pred = model(X)

        # Calculate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss
        train_acc += accuracy_fn(
            y_true=y, y_pred=y_pred.argmax(dim=1)
        )  # Go from logits -> pred labels

        # Optimizer zero grad
        optimizer.zero_grad()

        # Loss backward
        loss.backward()

        # Optimizer step
        optimizer.step()

    # Calculate loss and accuracy per epoch and print out what's happening
    train_loss /= len(data_loader)
    train_acc /= len(data_loader)
    print(f"Train loss: {train_loss:.5f} | Train accuracy: {train_acc:.2f}%")


def test_step(
    data_loader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    accuracy_fn,
    device: torch.device = device,
):
    """Testing model.

    Parameters
    ----------
    model : torch.nn.Module
        Model with weight after training.

    data_loader : torch.utils.data.DataLoader
        Data loader for training model.

    loss_fn : torch.nn.Module
        Loss function.

    optimizer : torch.optim.Optimizer
        Optimizer for training model.

    accuracy_fn
       Function for calculating accuracy.

    device : [type], optional
        Device that compute is running on. Defaults to None.

    Returns
    -------
    Result of testing model

    """
    test_loss, test_acc = 0, 0
    model.to(device)
    model.eval()  # put model in eval mode
    # Turn on inference context manager
    with torch.inference_mode():
        for X, y in data_loader:
            # Send data to GPU
            X, y = X.to(device), y.to(device)

            # Forward pass
            test_pred = model(X)

            # Calculate loss and accuracy
            test_loss += loss_fn(test_pred, y)
            test_acc += accuracy_fn(
                y_true=y,
                y_pred=test_pred.argmax(dim=1),  # Go from logits -> pred labels
            )

        # Adjust metrics and print out
        test_loss /= len(data_loader)
        test_acc /= len(data_loader)
        print(f"Test loss: {test_loss:.5f} | Test accuracy: {test_acc:.2f}%\n")


def train_test(batch, num_worker, p, infeatures, outfeatures):
    """Training and testing model.

    Parameters
    ----------
    batch : int
        Batch of dataset.

    num_worker : int
        Number of worker.

    p : float
        Probability for dropping out parameter.

    infeatures : int
        Number of in features.

    outfeatures : int
        Number of out features.

    Returns
    -------
    Result of training and testing model

    """
    # Data loader
    train_dataloader, valid_dataloader = data_loader(batch, num_worker)

    # Setup pretrained model
    weights, model = load_pretrained_model()

    # Update head classifier
    model = model_classifier(model, p, infeatures, outfeatures)

    # Setup loss function and optimier
    loss_fn, optimizer = loss_accuracy(model, learn_rate=0.1)

    # Download helper functions to load accuracy_fn
    download_helper()
    from helper_functions import accuracy_fn

    # Calculate time
    train_time_start_model = timer()

    # Start training and testing model
    for epoch in tqdm(range(epochs)):
        print(f"Epoch: {epoch}\n---------")
        train_step(
            data_loader=train_dataloader,
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            accuracy_fn=accuracy_fn,
            device=device,
        )
        test_step(
            data_loader=valid_dataloader,
            model=model,
            loss_fn=loss_fn,
            accuracy_fn=accuracy_fn,
            device=device,
        )

    # Stop calculating time
    train_time_end_model = timer()
    total_train_time = print_train_time(
        start=train_time_start_model, end=train_time_end_model, device=device
    )

    save_model(model, MODEL_SAVE_PATH)
