import multiprocessing
import os
from multiprocessing import Queue, Semaphore, Process
from uuid import uuid4
import soundfile as sf
from fastapi import UploadFile
from models import get_cascade_model

class ModelManager:
    def __init__(self, num_processes: int, temp_dir: str):
        self.num_processes = num_processes
        self.temp_dir = temp_dir
        self.semaphore = multiprocessing.Semaphore(num_processes)
        self.queue = Queue()
        self.model = get_cascade_model()
        self.parallel = False

        self.check_dir()

    def check_dir(self):
        os.makedirs(self.temp_dir, exist_ok=True)

    def save_file(self, file_bytes: bytes, file_extension: str) -> str:
        filename = f"{uuid4()}.{file_extension}"
        file_path = os.path.join(self.temp_dir, filename)
        with open(file_path, 'wb') as f:
            f.write(file_bytes)

        audio, samplerate = sf.read(file_path)
        if samplerate != 16000:
            sf.write(file_path, audio, 16000)

        return file_path


    def delete_file(self, file_path: str):
        if os.path.exists(file_path):
            os.remove(file_path)

    def predict(self, file_path: str, **kwargs):
        if self.parallel:
            try:
                self.semaphore.acquire()
                result = self._run_prediction(file_path, **kwargs)
                return result
            finally:
                self.semaphore.release()
        else:
            try:
                result = self._run_prediction(file_path, **kwargs)
                return result
            finally:
                pass
    def _run_prediction(self, file_path: str, **kwargs):
        if self.parallel:
            with multiprocessing.Manager() as manager:
                result_queue = manager.Queue()
                process = multiprocessing.Process(
                    target= self._process_prediction,
                    args=(file_path, result_queue, kwargs)
                )
                process.start()
                process.join()

                if not result_queue.empty():
                    return result_queue.get()
                else:
                    raise Exception("Empty queue")
        else:
            return self._process_prediction(file_path, None, kwargs)

    def _process_prediction(self, file_path: str, result_queue: Queue, kwargs):
        try:
            # TODO: раскомментировать после фикса
            # result = self.model.generate(file=file_path, max_new_tokens=150, do_sample=True, top_k=50, top_p=0.95)
            result = self.model.get_text()
            if self.parallel:
                result_queue.put(result)
            else:
                return result
        except Exception as e:
            raise e

    def process_request(self, file_bytes: bytes, file_extension: str, **kwargs):
        file_path = self.save_file(file_bytes, file_extension)
        try:
            result = self.predict(file_path, **kwargs)
        finally:
            self.delete_file(file_path)
        return result