# utils/logger.py

import logging
import os
import colorlog


def get_logger(name: str = "HPO"):
    logger = logging.getLogger(name)

    if not logger.handlers:
        # Set log level based on environment or default to INFO
        level = os.getenv("LOG_LEVEL", "INFO").upper()
        logger.setLevel(getattr(logging, level, logging.INFO))

        color_formatter = colorlog.ColoredFormatter(
            fmt="[%(asctime)s] [%(log_color)s%(levelname)s%(reset)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            log_colors={
                "DEBUG": "cyan",
                "INFO": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "bold_red",
            },
        )

        # Plain formatter for file output
        file_formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S"
        )

        # Console handler with color
        ch = colorlog.StreamHandler()
        ch.setFormatter(color_formatter)
        logger.addHandler(ch)

        # File handler
        fh = logging.FileHandler("hpo_log.txt")
        fh.setFormatter(file_formatter)
        logger.addHandler(fh)

    return logger
