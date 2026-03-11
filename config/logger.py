import logging
import sys
from pathlib import Path
from datetime import datetime

try:
    from config.config import LOGS_PATH, LOG_LEVEL, DEBUG_MODE
except ImportError:
    LOGS_PATH = Path("D:/ecg_project/logs")
    LOG_LEVEL = "INFO"
    DEBUG_MODE = True

def get_logger(module_name: str) -> logging.Logger:
    logger = logging.getLogger(module_name)
    if logger.handlers:
        return logger
    level = getattr(logging, LOG_LEVEL.upper(), logging.INFO)
    logger.setLevel(level)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    try:
        LOGS_PATH.mkdir(parents=True, exist_ok=True)
        log_filename = LOGS_PATH / f"ecg_{datetime.now().strftime('%Y-%m-%d')}.log"
        file_handler = logging.FileHandler(log_filename, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        logger.warning(f"Could not create log file on D: drive: {e}")
        logger.warning("Logging to terminal only")
    return logger

if __name__ == "__main__":
    logger = get_logger("logger_test")
    logger.info("Logger initialized successfully")
    logger.info(f"Log level     : {LOG_LEVEL}")
    logger.info(f"Log folder    : {LOGS_PATH}")
    logger.info(f"Debug mode    : {DEBUG_MODE}")
    logger.warning("This is a sample WARNING message")
    logger.error("This is a sample ERROR message — not a real error!")
    logger.info("Logger test complete — check D:/ecg_project/logs for the log file")