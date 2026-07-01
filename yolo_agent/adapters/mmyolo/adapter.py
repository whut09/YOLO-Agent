"""Adapter boundary for future MMYOLO integration."""

from __future__ import annotations


class MMYOLOAdapter:
    """Placeholder adapter for MMYOLO workflows."""

    name: str = "mmyolo"

    def is_available(self) -> bool:
        """Return whether the external integration is available."""
        return False

