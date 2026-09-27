"""Loguru-based logger with console + rotating file sink.

Also bridges stdlib `logging` into loguru — discord.py-self /
discord.ext.native_voice emit critical diagnostics (packet drops, DAVE
handshake state, listener teardown) through `logging` which would otherwise
be invisible in our files.
"""
from __future__ import annotations

import logging
import sys
from loguru import logger as _logger

from config.settings import settings


class _InterceptHandler(logging.Handler):
    """Route stdlib logging records through loguru."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            level = _logger.level(record.levelname).name
        except ValueError:
            level = record.levelno
        frame, depth = logging.currentframe(), 2
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1
        _logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


_logger.remove()
_logger.add(
    sys.stderr,
    level=settings.log_level,
    format=(
        "<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    ),
    backtrace=False,
    diagnose=False,
)
_logger.add(
    settings.logs_dir / "engager_{time:YYYY-MM-DD}.log",
    level="DEBUG",
    rotation="10 MB",
    retention="10 days",
    encoding="utf-8",
    backtrace=True,
    diagnose=False,
)

# Bridge stdlib logging -> loguru (native_voice packet drops, DAVE state,
# listener teardown all live there). DEBUG level for the voice internals so
# silent-drop paths become visible in the file sink.
logging.basicConfig(handlers=[_InterceptHandler()], level=logging.WARNING,
                    force=True)
for _name in (
    "discord.ext.native_voice",
    "discord.voice_state",
    "discord.gateway",
    "discord.opus",
):
    _l = logging.getLogger(_name)
    _l.setLevel(logging.DEBUG)
    _l.propagate = True
    _l.handlers = [_InterceptHandler()]

logger = _logger
