"""Logger utility for MulXAI-CropNet project."""

import logging
from pathlib import Path
from datetime import datetime


def setup_logger(name: str, log_dir: str = './logs') -> logging.Logger:
    """
    Setup logger with file and console handlers.
    
    Args:
        name: Logger name
        log_dir: Directory to save logs
        
    Returns:
        Configured logger
    """
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True, parents=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_path / f'{name}_{timestamp}.log'
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """Get logger by name."""
    return logging.getLogger(name)
