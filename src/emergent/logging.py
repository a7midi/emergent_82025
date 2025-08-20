# src/emergent/logging.py
"""
Minimal logging setup used across the package.

We intentionally keep this tiny in M00 to avoid side effects. Downstream modules
may use `get_logger(__name__)` to get a namespaced logger.
"""
from __future__ import annotations

import logging
from typing import Optional


def get_logger(name: Optional[str] = None) -> logging.Logger:
    logger = logging.getLogger("emergent" if name is None else f"emergent.{name}")
    if not logger.handlers:
        # Default to WARNING to keep CI noise low; users can tune as needed.
        handler = logging.StreamHandler()
        fmt = logging.Formatter("%(levelname)s | %(name)s | %(message)s")
        handler.setFormatter(fmt)
        logger.addHandler(handler)
        logger.setLevel(logging.WARNING)
        logger.propagate = False
    return logger