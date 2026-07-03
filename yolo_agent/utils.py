"""Shared utility helpers for the YOLO Agent harness."""

from __future__ import annotations

from typing import TypeVar

T = TypeVar("T")


def dedupe_list(values: list[T]) -> list[T]:
    """Return a list with duplicate items removed while preserving order."""
    return list(dict.fromkeys(values))
