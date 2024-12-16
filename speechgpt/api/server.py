import os
import uvicorn
from api.model_manager import ModelManager
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
# from speechgpt.api.config import settings

MAX_FILE_SIZE_MB = 10 * 1024 * 1024

app = FastAPI()
model_manager = ModelManager(4, "./temp")

@app.get("/")
async def root():
     return {"greeting":"Hello world"}

@app.post("/predict/")
async def predict_audio(file: UploadFile = File(...)):
    try:
        file_bytes = await file.read()

        if int(len(file_bytes)) > MAX_FILE_SIZE_MB:
            raise HTTPException(status_code=413, detail="File size exceeds 10 MB")

        file_extension = file.filename.split(".")[-1]
        if file_extension not in ["wav", "mp3", "flac"]:
            raise HTTPException(status_code=400, detail="Unsupported file format")

        result = model_manager.process_request(
            file_bytes=file_bytes,
            file_extension=file_extension
        )

        return JSONResponse(content={"text":result}, status_code=200)

    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error")

if __name__ == "__main__":
    print("running uvicorn")
    uvicorn.run(app, host="0.0.0.0", port=8000)
    print("runned uvicorn")