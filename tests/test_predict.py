from ktp_classification.config.config import MODEL_SAVE_PATH, device
from ktp_classification.predict import load_model


def test_load_model():
    # act
    loaded_model = load_model(MODEL_SAVE_PATH, device)

    # asser
    assert loaded_model is not None
