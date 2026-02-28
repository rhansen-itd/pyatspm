"""Centralized JSON formatter for structured logging."""

import json
import logging
from datetime import datetime
from typing import Any, Dict


class JsonFormatter(logging.Formatter):
    """Emit log records as single-line JSON objects.
    
    Merges any `extra=` kwargs directly into the payload.
    Supports structured queries in log aggregation systems.
    """

    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created).isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }

        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)

        # Merge any extra fields passed via logger.info(..., extra={...})
        for key, value in getattr(record, "__dict__", {}).items():
            if key not in {
                "msg", "args", "levelname", "levelno", "pathname", "filename",
                "module", "exc_info", "exc_text", "stack_info", "lineno",
                "funcName", "created", "msecs", "relativeCreated", "thread",
                "threadName", "processName", "process", "name", "message",
            }:
                payload[key] = value

        # safe fallback for non-serializable objects
        return json.dumps(payload, default=str)