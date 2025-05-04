# utils/logger.py

import logging
import os


def get_logger(name: str = "HPO"):
    logger = logging.getLogger(name)

    if not logger.handlers:
        # Set log level based on environment or default to INFO
        level = os.getenv("LOG_LEVEL", "INFO").upper()
        logger.setLevel(getattr(logging, level, logging.INFO))

        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S"
        )

        # Console handler
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        # File handler (optional)
        fh = logging.FileHandler("hpo_log.txt")
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger
