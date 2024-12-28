from typing import Annotated

import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from speechgpt.api.model_manager import ModelManager
from speechgpt.api.schemas import FitRequest, FitResponse, PredictResponse, StatusResponse
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
        X=request.X,
        y=request.y
    )
    message = "Model trained and loaded to inference"
    logger.info(message)
    response = FitResponse(message=message)
    return JSONResponse(content=response.model_dump(), status_code=200)


if __name__ == "__main__":
    # Starts the FastAPI app with Uvicorn.
    logger.info("Starting the FastAPI app.")
    uvicorn.run(app, host="0.0.0.0", port=8000)
    logger.info("Uvicorn started.")
