"""Logging padronizado do QSim: saida simultanea no terminal e em arquivo."""
from __future__ import annotations

import logging
import logging.handlers
from pathlib import Path

FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


def setup_logging(service_name: str, log_dir: str = "logs",
                  level: str = "INFO") -> logging.Logger:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(service_name)
    logger.setLevel(level.upper())
    logger.propagate = False

    if logger.handlers:
        return logger

    formatter = logging.Formatter(FORMAT)

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)

    file_handler = logging.handlers.RotatingFileHandler(
        Path(log_dir) / f"{service_name}.log",
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
