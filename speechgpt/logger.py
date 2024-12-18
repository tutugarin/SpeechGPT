import logging
from logging.handlers import RotatingFileHandler
import os

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")

if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

logger = logging.getLogger("speechgpt_logger")
logger.setLevel(logging.DEBUG)

log_handler = RotatingFileHandler(os.path.join(LOG_DIR, "app.log"), maxBytes=10*1024*1024, backupCount=5)
log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
log_handler.setFormatter(log_formatter)

logger.addHandler(log_handler)

def get_logger():
    return logger