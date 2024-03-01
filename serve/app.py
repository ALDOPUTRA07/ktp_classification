import logging
from typing import Any

import uvicorn
from fastapi import FastAPI, File, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from ktp_classification import __version__ as _version
from ktp_classification.config.config import MODEL_SAVE_PATH, PROJECT_NAME, device
from ktp_classification.predict import load_model, pred_image
from serve.config import description, origins, title
from serve.schema.schema import PredictionResults

# Logger
logger = logging.getLogger(__name__)

# Declaring our FastAPI instance
app = FastAPI(title=title, description=description, version=_version)

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Defining path operation for root endpoint
@app.get('/')
def model_info():
    return {'name': PROJECT_NAME, 'description': description, 'version': _version}


# predict
@app.post('/predict', status_code=200, response_model=PredictionResults)
def predict(file: bytes = File(...)) -> Any:
    loaded_model = load_model(MODEL_SAVE_PATH, device)

    try:
        predict_label, prob_label, total_train_time, _ = pred_image(
            model=loaded_model,
            image_path=file,
            class_names=['ktp', 'nonktp'],
            image_size=(224, 224),
        )
    except Exception as e:
        logger.exception(e)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Unable to process file",
        )

    results = {}
    results['predict_label'] = predict_label
    results['probability'] = prob_label
    results['execution_time'] = total_train_time

    return JSONResponse(results)


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
