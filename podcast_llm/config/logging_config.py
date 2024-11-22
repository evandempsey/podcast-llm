"""
Logging configuration module for the podcast generation system.

This module provides standardized logging setup across the application.
It configures a consistent log format and allows for dynamic log level setting.

Example:
    >>> from podcast_llm.config.logging_config import setup_logging
    >>> setup_logging(logging.DEBUG)  # Set debug level logging
    >>> logger = logging.getLogger(__name__)
    >>> logger.debug('Debug message')
    2024-01-01 12:00:00 - DEBUG - Debug message

The logging format includes:
- Timestamp in YYYY-MM-DD HH:MM:SS format
- Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- Message content
"""

import logging
import sys
from typing import Optional


def setup_logging(log_level: Optional[int] = None, output_file: Optional[str] = None) -> None:
    """
    Set up standardized logging configuration for the podcast generation system.

    Args:
        log_level: Optional logging level to set. If None, defaults to INFO.
            Use logging.DEBUG for debug output.
        output_file: Optional file path to write logs to. If None, logs to stdout.

    The format for log messages is:
    YYYY-MM-DD HH:MM:SS - LEVEL - MESSAGE
    """
    # Remove any existing handlers to avoid duplicate logs
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Configure logging handler based on output destination
    if output_file:
        handler = logging.FileHandler(output_file)
    else:
        handler = logging.StreamHandler(sys.stdout)

    handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))
    
    # Set up root logger
    root_logger.addHandler(handler)
    root_logger.setLevel(log_level or logging.INFO)
