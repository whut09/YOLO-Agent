"""Component registry scaffold."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ComponentDescriptor:
    """Describe a selectable model, loss, block, or training strategy."""

    name: str
    category: str
    description: str = ""

