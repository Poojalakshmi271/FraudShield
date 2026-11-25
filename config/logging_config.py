"""
FraudShield – Logging Configuration
Centralised logging setup for all modules.
"""

import logging
import os
import sys
from pathlib import Path


def setup_logging(
    level: str = "INFO",
    log_file: str | None = None,
    logger_name: str = "fraudshield",
) -> logging.Logger:
    """
    Configure and return a named logger.

    Parameters
    ----------
    level : str
        Logging level string (DEBUG, INFO, WARNING, ERROR, CRITICAL).
    log_file : str | None
        Optional path to a log file. If None, logs go to stdout only.
    logger_name : str
        Name of the logger instance.

    Returns
    -------
    logging.Logger
        Configured logger.
    """
    logger = logging.getLogger(logger_name)

    # Prevent duplicate handlers when called multiple times
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(name)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler (optional)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


# Module-level default logger
logger = setup_logging()
