"""
Structured JSON logging for production + coloured console output for dev.
"""

import logging
import sys


class ColorFormatter(logging.Formatter):
    COLORS = {
        "DEBUG":    "\033[36m",   # cyan
        "INFO":     "\033[32m",   # green
        "WARNING":  "\033[33m",   # yellow
        "ERROR":    "\033[31m",   # red
        "CRITICAL": "\033[35m",   # magenta
    }
    RESET = "\033[0m"
    BOLD  = "\033[1m"

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, self.RESET)
        level = f"{color}{self.BOLD}{record.levelname:<8}{self.RESET}"
        name  = f"\033[90m{record.name}\033[0m"

        extras = {
            k: v for k, v in record.__dict__.items()
            if k not in logging.LogRecord.__dict__ and not k.startswith("_")
               and k not in ("message", "asctime", "exc_info", "exc_text",
                             "stack_info", "levelname", "name", "msg",
                             "args", "pathname", "filename", "module",
                             "funcName", "lineno", "created", "msecs",
                             "relativeCreated", "thread", "threadName",
                             "processName", "process")
        }
        extra_str = "  " + "  ".join(f"{k}={v}" for k, v in extras.items()) if extras else ""

        return f"{level} {name}: {record.getMessage()}{extra_str}"


def setup_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(ColorFormatter())

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()
    root.addHandler(handler)

    # Suppress noisy third-party loggers
    for noisy in ("uvicorn.access", "httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)