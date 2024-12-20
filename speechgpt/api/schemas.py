from pydantic import BaseModel
from typing import List, Dict, Optional, Any, Union


class ModelConfig(BaseModel):
    """Configuration for the model."""
    id: str
    hyperparameters: Dict[str, Any]


class FitRequest(BaseModel):
    """Request schema for fitting a model."""
    X: List[List[float]]
    y: List[float]
    config: ModelConfig


class FitResponse(BaseModel):
    """Response schema for the fitting process."""
    message: str


class PredictResponse(BaseModel):
    """Response schema for prediction."""
    text: str


class StatusResponse(BaseModel):
    """Response schema for health check."""
    status: str
