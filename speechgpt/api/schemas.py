from typing import List, Dict, Any, Union, Optional
from pydantic import BaseModel
from fastapi import File, UploadFile

class ModelConfig(BaseModel):
    """Configuration for the model."""
    id: str
    hyperparameters: Dict[str, Any]


class FitRequest(BaseModel):
    """Request schema for fitting a model."""
    config: ModelConfig


class ModelListResponse(BaseModel):
    """Response schema for model list."""
    models: List[Dict[str, Any]]


class FitResponse(BaseModel):
    """Response schema for the fitting process."""
    message: str


class PredictResponse(BaseModel):
    """Response schema for prediction."""
    text: str


class StatusResponse(BaseModel):
    """Response schema for health check."""
    status: str


class ModelListResponse(BaseModel):
    """List of models response."""
    models: List[Dict[str, Any]]


class SetModelRequest(BaseModel):
    """Request for setting model."""
    id: str


class SetModelResponse(BaseModel):
    """Responce for setting model."""
    message: str
