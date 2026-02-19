"""
Structured logging framework for the LLM Safety Evaluation pipeline.

Provides a consistent logging configuration with JSON-formatted output
suitable for log aggregation systems (e.g. CloudWatch, Stackdriver,
Datadog) as well as human-readable console output.

Usage
-----
    from src.utils.logger import get_logger, configure_logging

    configure_logging(level="INFO", json_format=False)
    logger = get_logger(__name__)
    logger.info("Evaluation started", extra={"run_id": "abc123"})
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any


# ---------------------------------------------------------------------------
# JSON log formatter
# ---------------------------------------------------------------------------

class JsonFormatter(logging.Formatter):
    """Formats log records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        log_dict: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            log_dict["exception"] = self.formatException(record.exc_info)
        # Include any extra fields passed via the `extra` kwarg
        _std_attrs = logging.LogRecord.__dict__.keys() | {
            "message", "asctime", "args", "msg",
        }
        for key, val in record.__dict__.items():
            if key not in _std_attrs and not key.startswith("_"):
                log_dict[key] = val
        return json.dumps(log_dict, default=str)


# ---------------------------------------------------------------------------
# Human-readable formatter
# ---------------------------------------------------------------------------

_HUMAN_FORMAT = (
    "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def configure_logging(
    level: str | int = "INFO",
    json_format: bool = False,
    stream=None,
) -> None:
    """
    Configure the root logger for the framework.

    Parameters
    ----------
    level:
        Log level name (``"DEBUG"``, ``"INFO"``, etc.) or integer constant.
    json_format:
        When ``True``, use JSON log formatting; otherwise use a human-readable
        format.
    stream:
        Output stream.  Defaults to ``sys.stderr``.
    """
    root = logging.getLogger()
    root.setLevel(level)

    # Remove any existing handlers to avoid duplicate logs
    for handler in root.handlers[:]:
        root.removeHandler(handler)

    handler = logging.StreamHandler(stream or sys.stderr)
    handler.setLevel(level)

    if json_format:
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter(_HUMAN_FORMAT, datefmt="%Y-%m-%d %H:%M:%S"))

    root.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    """
    Return a logger with the given *name*.

    This is a thin wrapper around :func:`logging.getLogger` that ensures
    the framework-wide configuration is respected.

    Parameters
    ----------
    name:
        Logger name (conventionally ``__name__``).
    """
    return logging.getLogger(name)


# ---------------------------------------------------------------------------
# Evaluation run context logger
# ---------------------------------------------------------------------------

class EvalRunLogger:
    """
    Thin context-aware logger that attaches a ``run_id`` to every log record.

    Parameters
    ----------
    run_id:
        Unique identifier for the evaluation run.
    base_logger:
        Optional base logger.  Defaults to the framework root logger.
    """

    def __init__(
        self,
        run_id: str,
        base_logger: logging.Logger | None = None,
    ) -> None:
        self.run_id = run_id
        self._log = base_logger or logging.getLogger("llm_safety")

    def _extra(self, **kwargs: Any) -> dict:
        return {"run_id": self.run_id, **kwargs}

    def info(self, msg: str, **kwargs: Any) -> None:
        self._log.info(msg, extra=self._extra(**kwargs))

    def warning(self, msg: str, **kwargs: Any) -> None:
        self._log.warning(msg, extra=self._extra(**kwargs))

    def error(self, msg: str, **kwargs: Any) -> None:
        self._log.error(msg, extra=self._extra(**kwargs))

    def debug(self, msg: str, **kwargs: Any) -> None:
        self._log.debug(msg, extra=self._extra(**kwargs))
