"""
Structured logging configuration using python-json-logger.
Supports both console and file outputs with different levels.
"""

import logging
import sys
from pathlib import Path
from pythonjsonlogger import jsonlogger
from src.config import settings


def setup_logger(name: str = "visa_rag") -> logging.Logger:
    """Setup structured logging with JSON formatter."""
    
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, settings.log_level))
    
    # Create logs directory
    logs_dir = settings.logs_dir
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    # JSON formatter for structured logging
    json_formatter = jsonlogger.JsonFormatter(
        "%(timestamp)s %(level)s %(name)s %(message)s",
        timestamp=True,
    )
    
    # Console handler (always INFO or above in dev)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO if settings.debug else logging.WARNING)
    console_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    
    # File handler (all levels)
    file_handler = logging.FileHandler(logs_dir / f"{name}.log")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(json_formatter)
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger


# Global logger instance
logger = setup_logger()
