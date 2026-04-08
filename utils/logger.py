import logging
import logging.handlers
from pathlib import Path

from config.settings import get_settings


def setup_logger(name: str) -> logging.Logger:
    """
    Sets up a robust logger with both console and rotating file handlers.
    
    Args:
        name (str): The name of the logger, typically __name__.
        
    Returns:
        logging.Logger: The configured logger instance.
    """
    # Get application settings to retrieve the log directory
    settings = get_settings()
    log_dir = settings.paths.log_dir
    
    # Ensure log directory exists (handled by get_settings, but safe to verify)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger(name)
    
    # Only configure if it doesn't already have handlers to prevent duplicates
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        
        # Define professional formatting
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        
        # 1. Console Handler (Standard Out)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        
        # 2. Rotating File Handler
        log_file = log_dir / f"{name.replace('.', '_')}.log"
        file_handler = logging.handlers.RotatingFileHandler(
            filename=log_file,
            maxBytes=5 * 1024 * 1024,  # 5 MB
            backupCount=3,
            encoding="utf-8"
        )
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        
        # Add handlers to the logger
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
        
        # Prevent log messages from being propagated to the root logger
        logger.propagate = False
        
    return logger
