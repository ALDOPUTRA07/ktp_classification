from pydantic import BaseModel


# Define response model
class PredictionResults(BaseModel):
    predict_label: str
    probability: float
    execution_time: float
