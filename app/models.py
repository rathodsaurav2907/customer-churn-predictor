from pydantic import BaseModel
from typing import List, Optional

class ChurnPredictionRequest(BaseModel):
    age: int
    tenure: int
    monthly_charges: float
    total_charges: float
    contracts: str
    internet_type: str

class ChurnPredictionResponse(BaseModel):
    churn_probability: float
    churn_prediction: bool
    risk_level: str
    confidence: float

class ModelInfoResponse(BaseModel):
    model_name: str
    model_type: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    feature_importance: dict
    training_samples: int
    features: List[str]

class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    timestamp: str
