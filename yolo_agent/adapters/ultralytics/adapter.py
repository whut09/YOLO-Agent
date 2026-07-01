"""Adapter boundary for future Ultralytics integration."""

from __future__ import annotations


class UltralyticsAdapter:
    """Placeholder adapter for Ultralytics YOLO workflows."""

    name: str = "ultralytics"

    def is_available(self) -> bool:
        """Return whether the external integration is available."""
        return False

