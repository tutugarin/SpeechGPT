# import os
# from pydantic import ValidationError
# from pydantic_settings import BaseSettings
# from dotenv import load_dotenv
#
# dotenv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '.env')
# load_dotenv(dotenv_path=dotenv_path, encoding='utf-8')
#
#
# class Settings(BaseSettings):
#     # models_path: str = os.getenv("MODELS_PATH", "./store")
#     # num_cores: int = int(os.getenv("NUM_CORES", "4"))
#     # max_inference_models: int = int(os.getenv("MAX_INFERENCE_MODELS", "3"))
#     num_cores: int = 4
#     temp_dir: str = "./temp"
#
#
# try:
#     settings = Settings()
#
#     if not os.path.exists(settings.models_path):
#         os.makedirs(settings.models_path)
#         print(f"Created directory: {settings.models_path}")
#
#     if settings.max_inference_models < 2:
#         raise ValueError("There are must be at least 2 inference models")
#
# except ValidationError as e:
#     raise RuntimeError("Failed to load configuration from .env") from e