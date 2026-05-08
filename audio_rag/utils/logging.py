"""Logging utilities for Audio RAG with Triton support."""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logging(
    level: str = "INFO",
    log_format: Optional[str] = None,
    log_file: Optional[str] = None,
) -> logging.Logger:
    """Setup logging configuration for Audio RAG.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format: Custom log format string
        log_file: Optional file path for logging

    Returns:
        Configured logger instance
    """
    # Default format for Triton containers
    if log_format is None:
        log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"

    # Convert string level to logging constant
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    # Configure root logger
    logging.basicConfig(
        level=numeric_level,
        format=log_format,
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[],
    )

    # Get root logger
    root_logger = logging.getLogger()
    root_logger.handlers = []  # Clear existing handlers

    # Console handler (stdout for Triton)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(logging.Formatter(log_format, datefmt="%Y-%m-%d %H:%M:%S"))
    root_logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        log_path = Path(log_file).expanduser().resolve()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(logging.Formatter(log_format, datefmt="%Y-%m-%d %H:%M:%S"))
        root_logger.addHandler(file_handler)

    # Reduce noise from third-party libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("tritonclient").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

    return root_logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance for a specific module.

    Args:
        name: Logger name (usually __name__)

    Returns:
        Logger instance
    """
    return logging.getLogger(name)


def log_model_loading(logger: logging.Logger, model_name: str, cache_dir: Optional[str] = None) -> None:
    """Log model loading information.

    Args:
        logger: Logger instance
        model_name: Name of the model being loaded
        cache_dir: Cache directory path
    """
    if cache_dir:
        logger.info(f"Loading model '{model_name}' from cache: {cache_dir}")
    else:
        logger.info(f"Loading model '{model_name}' (will download if not cached)")


def log_model_loaded(logger: logging.Logger, model_name: str, load_time: float) -> None:
    """Log successful model loading.

    Args:
        logger: Logger instance
        model_name: Name of the model
        load_time: Time taken to load the model in seconds
    """
    logger.info(f"Model '{model_name}' loaded successfully in {load_time:.2f}s")


def log_request(logger: logging.Logger, model_name: str, request_id: str, inputs: dict) -> None:
    """Log incoming request.

    Args:
        logger: Logger instance
        model_name: Name of the model receiving the request
        request_id: Unique request identifier
        inputs: Request inputs (simplified for logging)
    """
    # Simplify inputs for logging (don't log large arrays)
    simplified_inputs = {}
    for key, value in inputs.items():
        if isinstance(value, (list, tuple)):
            if len(value) > 5:
                simplified_inputs[key] = f"[{type(value[0]).__name__} x {len(value)}]"
            else:
                simplified_inputs[key] = value
        elif isinstance(value, bytes):
            simplified_inputs[key] = f"<bytes len={len(value)}>"
        else:
            simplified_inputs[key] = value

    logger.info(f"[{request_id}] Request to '{model_name}': {simplified_inputs}")


def log_response(logger: logging.Logger, model_name: str, request_id: str, response_time: float, success: bool = True) -> None:
    """Log response information.

    Args:
        logger: Logger instance
        model_name: Name of the model
        request_id: Unique request identifier
        response_time: Time taken to process the request in seconds
        success: Whether the request was successful
    """
    status = "SUCCESS" if success else "FAILED"
    logger.info(f"[{request_id}] Response from '{model_name}': {status} in {response_time:.3f}s")


def log_download_progress(logger: logging.Logger, model_name: str, progress: str) -> None:
    """Log model download progress.

    Args:
        logger: Logger instance
        model_name: Name of the model being downloaded
        progress: Progress information
    """
    logger.info(f"Downloading model '{model_name}': {progress}")
