import asyncio
import os
import time
from concurrent.futures import ProcessPoolExecutor
from uuid import uuid4
import soundfile as sf
from logger import get_logger
import multiprocessing as mp
from multiprocessing import Queue, Process
from models import get_cascade_model

import torch.multiprocessing
mp.set_start_method("spawn", force=True)
torch.multiprocessing.set_sharing_strategy('file_descriptor')  # Альтернативная стратегия


logger = get_logger()

class ModelManager:
    """
    Class to manage the models and temporary files.
    """

    def __init__(self, num_processes: int, temp_dir: str, parallel: bool):
        """
        Initialize the ModelManager.

        Parameters:
            num_processes (int): The number of processes to use in parallel.
            temp_dir (str): The directory to store the temporary files.
        """
        self.num_processes = num_processes
        self.temp_dir = temp_dir
        self.model = get_cascade_model()
        self.parallel = parallel
        self.executor = ProcessPoolExecutor(max_workers=self.num_processes)

        self.check_dir()

    def check_dir(self):
        """
        Create the temporary directory if it doesn't exist.
        """
        os.makedirs(self.temp_dir, exist_ok=True)
        logger.info("Temporary directory created: %s", self.temp_dir)

    def save_file(self, file_bytes: bytes, file_extension: str) -> str:
        """
        Save the file to the temporary directory.

        Parameters:
            file_bytes (bytes): The file contents.
            file_extension (str): The file extension.

        Returns:
            str: The path to the temporary file.
        """
        filename = f"{uuid4()}.{file_extension}"
        file_path = os.path.join(self.temp_dir, filename)
        try:
            with open(file_path, 'wb') as f:
                f.write(file_bytes)
            logger.info("File saved: %s", file_path)

            audio, samplerate = sf.read(file_path)
            if samplerate != 16000:
                sf.write(file_path, audio, 16000)
                logger.info("Resampled file to 16000 Hz: %s", file_path)

            return file_path
        except Exception as e:
            logger.exception("Error saving file: %s", file_path)
            raise e

    def delete_file(self, file_path: str):
        """
        Delete the file from the temporary directory.

        Parameters:
            file_path (str): The path to the file.
        """
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info("File deleted: %s", file_path)
        except Exception as e:
            logger.exception("Error deleting file: %s", file_path)

    def fit(self, X, y, *args, **kwargs):
        """
        Fit the model.

        Parameters:
            X (list): The features.
            y (list): The labels.
        """
        def training():
            time.sleep(5)

        process = Process(target=training)
        process.start()
        process.join()

    async def predict(self, file_path: str, **kwargs) -> str:
        """
        Make a prediction using the model.

        Parameters:
            file_path (str): The path to the file to predict.

        Returns:
            tuple(str/future, bool): prediction and type
        """
        try:
            if self.parallel:
                result = await self._run_parallel_prediction(file_path, **kwargs)
            else:
                result = self._run_prediction(logger, self.model, file_path, **kwargs)
            return result
        except Exception as e:
            logger.exception("Error making prediction: %s", e)

    async def _run_parallel_prediction(self, file_path: str, **kwargs) -> str:
        """
        Run prediction in parallel using ProcessPoolExecutor.

        Parameters:
            file_path (str): The path to the file for prediction.

        Returns:
            str: The prediction result.
        """

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            self.executor,
            ModelManager._run_prediction,
            logger,
            self.model,
            file_path,
            **kwargs
        )

        logger.info(f"Prediction task started for {file_path}")

        return result

    @staticmethod
    def _run_prediction(log, model, file_path: str, **kwargs) -> str:
        """
        Process the prediction in a separate process.

        Parameters:
            file_path (str): The path to the file to predict.
        """
        try:
            results = model.generate(file=file_path,
                                     logger=log,
                                         max_new_tokens=150,
                                         do_sample=True,
                                         top_k=50,
                                         top_p=0.95)
            response = results[0]
            result = ". ".join(response.split(". ")[:-1]) + "."

            return result
        except Exception as e:
            logger.exception("Error during prediction for file: %s", file_path)
            raise e

    async def process_request(self, file_bytes: bytes, file_extension: str, **kwargs) -> str:
        """
        Process a request by saving the file, making a prediction, and deleting the file.

        Parameters:
            file_bytes (bytes): The file contents.
            file_extension (str): The file extension.

        Returns:
             tuple(str/future, bool): prediction and type.
        """
        file_path = self.save_file(file_bytes, file_extension)
        try:
            result = await self.predict(file_path, **kwargs)
        finally:
            self.delete_file(file_path)
        return result
