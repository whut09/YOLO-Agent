"""Shared pytest options for explicitly gated hardware tests."""

from __future__ import annotations

import os

import pytest


def pytest_configure() -> None:
    """Keep tests deterministic even when a developer has local LLM credentials."""
    os.environ.setdefault("YOLO_AGENT_DISABLE_LOCAL_LLM", "1")


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-real-gpu",
        action="store_true",
        default=False,
        help="run tests marked real_gpu (may train models on CUDA)",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    enabled = config.getoption("--run-real-gpu") or os.getenv("YOLO_AGENT_RUN_REAL_GPU") == "1"
    if enabled:
        return
    marker = pytest.mark.skip(reason="real GPU acceptance is opt-in; pass --run-real-gpu")
    for item in items:
        if "real_gpu" in item.keywords:
            item.add_marker(marker)
