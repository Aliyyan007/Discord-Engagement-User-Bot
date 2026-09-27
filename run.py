"""Engager Bot — entry point.

Run with:  python run.py
"""
from __future__ import annotations

import asyncio
import signal
import sys

import os
import threading

from config.settings import settings
from core.bot import EngagerBot
from utils.logger import logger


def _health_server() -> None:
    """Tiny HTTP responder on $PORT — Render's web service requires a port
    bind or the deploy fails. Returns 200 on any GET; the user pings this
    externally to prevent free-tier sleep."""
    port = int(os.environ.get("PORT", "0") or 0)
    if not port:
        return
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    class _H(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"ok")

        def log_message(self, *a):
            pass

    ThreadingHTTPServer(("0.0.0.0", port), _H).serve_forever()


async def main() -> None:
    if not settings.discord_token:
        logger.error("DISCORD_TOKEN is missing from .env — aborting.")
        sys.exit(1)
    if not settings.groq_keys:
        logger.error("No GROQ_API_KEY* found in .env — aborting.")
        sys.exit(1)

    logger.info(f"Starting Engager Bot (model={settings.groq_model_text}).")
    threading.Thread(target=_health_server, daemon=True).start()
    bot = EngagerBot()

    # Handle SIGTERM for clean shutdown in container/production environments.
    loop = asyncio.get_event_loop()
    def _sigterm():
        logger.info("SIGTERM received — shutting down.")
        asyncio.ensure_future(bot.close())
    try:
        loop.add_signal_handler(signal.SIGTERM, _sigterm)
    except (NotImplementedError, RuntimeError):
        pass  # Windows doesn't support add_signal_handler

    try:
        await bot.start(settings.discord_token)
    except KeyboardInterrupt:
        logger.info("Shutdown requested.")
    finally:
        await bot.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
