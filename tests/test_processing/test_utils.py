from torch import nn

from ktp_classification.model import load_pretrained_model
from ktp_classification.processing.utils import download_helper, loss_accuracy


def test_download_helper():
    # act
    helper = download_helper()

    # assert
    assert helper is None


def test_loss_accuracy():
    # arrange
    weight, model = load_pretrained_model()

    # act
    loss_fn, optimizer = loss_accuracy(model, 0.1)

    # assert
    assert type(loss_fn) == nn.modules.loss.CrossEntropyLoss
    assert optimizer is not None
