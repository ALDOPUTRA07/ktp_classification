from torchvision.models import EfficientNet_B0_Weights

from ktp_classification.model import (
    auto_transform,
    load_pretrained_model,
    model_classifier,
)


def test_pretrained_model():
    # act
    weights, model = load_pretrained_model()

    # assert
    assert weights == EfficientNet_B0_Weights.IMAGENET1K_V1
    assert model is not None


def test_model_classifier(load_pretrained_model):
    # arrange
    model = load_pretrained_model

    # act
    classifier = model_classifier(model, 0.3, 1280, 2)

    # assert
    assert classifier is not None


def test_auto_transform(load_weight_model):
    # arrange
    weight = load_weight_model

    # act
    trasnform = auto_transform(weight)

    # assert
    assert trasnform is not None
