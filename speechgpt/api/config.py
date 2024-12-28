import os
from pydantic import ValidationError
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

dotenv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '.env')
load_dotenv(dotenv_path=dotenv_path, encoding='utf-8')


class Settings(BaseSettings):
    """
    Settings class that loads configuration parameters from environment variables or the .env file.

    Attributes:
        temp_dir (str): Directory for temporary files. Default is './temp'.
        num_cores (int): Number of cores to use. Default is 4.
    """
    temp_dir: str = os.getenv("TEMP_DIR", "./temp")
    num_cores: int = int(os.getenv("NUM_CORES", "4"))
    max_file_size: int = int(os.getenv("MAX_FILE_SIZE", f"{10 * 1024 * 1024}"))
    parallel: bool = os.getenv('PARALLEL', 'False') == 'True'

try:
    settings = Settings()

    if not os.path.exists(settings.temp_dir):
        os.makedirs(settings.temp_dir)
        print(f"Created directory: {settings.temp_dir}")

except ValidationError as e:
    raise RuntimeError("Failed to load configuration from .env") from e
