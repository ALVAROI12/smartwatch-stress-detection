"""Utility helpers for configuring consistent logging across the project."""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, Optional, Union


def _resolve_level(level: Union[str, int], fallback: int) -> int:
    """Translate a level name or integer into a valid logging level."""
    if isinstance(level, int):
        return level
    if isinstance(level, str):
        numeric_level = getattr(logging, level.upper(), None)
        if isinstance(numeric_level, int):
            return numeric_level
    return fallback


def setup_logging(
    config: Optional[Dict[str, Any]] = None,
    override_level: Optional[Union[str, int]] = None,
) -> None:
    """Configure root logging handlers using dictionary-style settings."""
    config = config or {}

    base_level = _resolve_level(config.get('level', logging.INFO), logging.INFO)
    if override_level is not None:
        base_level = _resolve_level(override_level, base_level)

    log_format = config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    datefmt = config.get('datefmt', '%Y-%m-%d %H:%M:%S')

    handlers = []

    console_config = config.get('console', {})
    if console_config.get('enabled', True):
        console_handler = logging.StreamHandler()
        console_level = _resolve_level(console_config.get('level', base_level), base_level)
        console_handler.setLevel(console_level)
        console_handler.setFormatter(logging.Formatter(log_format, datefmt=datefmt))
        handlers.append(console_handler)

    file_config = config.get('file', {})
    if file_config.get('enabled'):
        log_path = Path(file_config.get('path', 'logs/app.log'))
        log_path.parent.mkdir(parents=True, exist_ok=True)
        max_bytes = int(file_config.get('max_bytes', 1_048_576))
        backup_count = int(file_config.get('backup_count', 5))

        rotating_handler = RotatingFileHandler(
            log_path,
            maxBytes=max_bytes,
            backupCount=backup_count,
        )
        file_level = _resolve_level(file_config.get('level', base_level), base_level)
        rotating_handler.setLevel(file_level)
        rotating_handler.setFormatter(logging.Formatter(log_format, datefmt=datefmt))
        handlers.append(rotating_handler)

    logging.basicConfig(
        level=base_level,
        format=log_format,
        datefmt=datefmt,
        handlers=handlers or None,
        force=True,
    )

    logging.captureWarnings(True)


def initialize_logging(
    logger_name: str,
    override_level: Optional[Union[str, int]] = None,
) -> logging.Logger:
    """Set up logging using project configuration and return named logger."""

    config = None
    try:
        from utils.config_manager import get_config  # local import avoids cycles

        config = get_config().get_logging_config()
    except Exception as exc:  # pragma: no cover - defensive fallback
        logging.basicConfig(level=logging.INFO, force=True)
        fallback_logger = logging.getLogger(logger_name)
        fallback_logger.warning(
            "Failed to load logging configuration; using INFO fallback (%s)",
            exc,
        )
        return fallback_logger

    setup_logging(config, override_level=override_level)
    return logging.getLogger(logger_name)
