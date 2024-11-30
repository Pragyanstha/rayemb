import logging
import sys
from loguru import logger


def setup_logger(name):
    # Remove default Loguru logger
    logger.remove()

    # Add a new logger with a specific format
    logger.add(sys.stderr, format="{time} {level} {message}", level="INFO")
    return logger.bind(name=name)
