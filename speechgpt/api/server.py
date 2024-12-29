from typing import Annotated, Optional

import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.responses import JSONResponse
from requests import Request

from speechgpt.api.model_manager import ModelManager
from speechgpt.api.schemas import FitRequest, FitResponse, PredictResponse, StatusResponse, ModelListResponse, \
    SetModelResponse, SetModelRequest
from speechgpt.logger import get_logger
from speechgpt.api.config import settings


logger = get_logger()

app = FastAPI()

model_manager = ModelManager(
    settings.num_cores,
    settings.temp_dir,
    settings.parallel
)

@app.get("/", response_model=StatusResponse)
async def check_health() -> StatusResponse:
    """
    Health check endpoint to verify if the app is running.

    Returns:
        StatusResponse: The health status of the application.
    """
    logger.info("Health check request received.")
    return StatusResponse(status="App healthy")


@app.post("/predict/", response_model=PredictResponse)
async def predict(
    file: Annotated[UploadFile, File(...)],
) -> PredictResponse:
    """
    Endpoint for audio prediction.

    Args:
        file (UploadFile): Audio file for prediction.

    Returns:
        PredictResponse: The prediction result from the model.
    """
    try:
        file_bytes = await file.read()

        if int(len(file_bytes)) > settings.max_file_size:
            logger.warning("File size exceeds the limit: %s", len(file_bytes))
            raise HTTPException(status_code=413, detail="File size exceeds 10 MB")

        file_extension = file.filename.split(".")[-1]
        if file_extension not in ["wav", "mp3", "flac"]:
            logger.error("Unsupported file format: %s", file_extension)
            raise HTTPException(status_code=400, detail="Unsupported file format")

        result = await model_manager.process_request(
            file_bytes=file_bytes,
            file_extension=file_extension
        )

        logger.info("Prediction successful for file: %s", file.filename)
        response = PredictResponse(text=result)
        return JSONResponse(content=response.model_dump(), status_code=200)

    except HTTPException as e:
        logger.error("HTTP error occurred: %s", str(e))
        raise e
    except Exception as e:
        logger.exception("Internal server error occurred.")
        raise HTTPException(status_code=500, detail="Internal server error") from e


@app.post("/fit", response_model=FitResponse)
async def fit(request: FitRequest) -> FitResponse:
    """
    Endpoint to fit and train a model.

    Args:
        request (FitRequest): The training request with features, labels, and model config.

    Returns:
        FitResponse: The result message after fitting the model.
    """
    logger.info("Fit request received.")
    _ = model_manager.fit(
        config=request.config
    )
    message = "Model trained and loaded to inference"
    logger.info(message)
    response = FitResponse(message=message)
    return JSONResponse(content=response.model_dump(), status_code=200)


@app.get("/models", response_model=ModelListResponse)
async def list_models() -> ModelListResponse:
    """
    Endpoint to retrieve the list of models with detailed information.

    Returns:
        ModelListResponse: A list of models with detailed information.
    """
    logger.info("Models list request received.")
    models = model_manager.get_models_list()
    logger.info("Retrieved models: %s", models)

    response = ModelListResponse(models=models)
    return JSONResponse(content=response.model_dump(), status_code=200)


@app.post("/set", response_model=SetModelResponse)
async def set_model(request: SetModelRequest) -> SetModelResponse:
    """
    Endpoint to set the active model by its ID.

    Args:
        request (SetModelRequest): The request containing the model ID.

    Returns:
        SetModelResponse: A response confirming the active model is set.
    """

    logger.info("Set model request received for ID: %s", request.id)

    _ = model_manager.set_active_model(request.id)

    message = f"Model with ID {request.id} set as active."
    logger.info(message)
    response = SetModelResponse(message=message)
    return JSONResponse(content=response.model_dump(), status_code=200)


async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={
            "detail": str(exc)
        }
    )

app.add_exception_handler(Exception, global_exception_handler)

if __name__ == "__main__":
    # Starts the FastAPI app with Uvicorn.
    logger.info("Starting the FastAPI app.")
    uvicorn.run(app, host="0.0.0.0", port=8001)
    logger.info("Uvicorn started.")
