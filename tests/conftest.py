"""Shared pytest configuration."""

from __future__ import annotations

import os


def pytest_configure() -> None:
    """Keep tests deterministic even when a developer has local LLM credentials."""
    os.environ.setdefault("YOLO_AGENT_DISABLE_LOCAL_LLM", "1")
