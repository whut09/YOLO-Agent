"""User-facing automatic optimization budget policy."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


BudgetMode = Literal["auto", "fixed_rounds"]


class AutoOptimizationBudget(BaseModel):
    """Bounded pilot-search budget; rounds are only a final safety limit."""

    schema_version: str = "1.0"
    mode: BudgetMode = "auto"
    max_gpu_hours: float = Field(default=24.0, gt=0.0)
    max_pilots: int = Field(default=12, ge=1)
    no_improvement_patience: int = Field(default=4, ge=1)
    max_concurrent_pilots: int = Field(default=1, ge=1, le=1)
    max_rounds_safety: int = Field(default=60, ge=1)
    full_requires_confirmation: bool = True
    explicit_rounds: int | None = Field(default=None, ge=0)

    @property
    def effective_round_limit(self) -> int:
        """Return the driver cap without treating it as the primary budget."""
        return self.explicit_rounds if self.explicit_rounds is not None else self.max_rounds_safety

    @property
    def expected_pilot_range(self) -> str:
        """Return a deliberately non-promissory pilot-count range."""
        return f"1-{self.max_pilots}"

    @classmethod
    def from_training_config(
        cls,
        path: Path | str,
        *,
        explicit_rounds: int | None = None,
    ) -> "AutoOptimizationBudget":
        """Load budget guardrails from the training config goal section."""
        config_path = Path(path)
        goal: dict[str, object] = {}
        if config_path.is_file():
            raw = yaml.safe_load(config_path.read_text(encoding="utf-8-sig")) or {}
            if isinstance(raw, dict) and isinstance(raw.get("goal"), dict):
                goal = raw["goal"]
        return cls(
            mode="fixed_rounds" if explicit_rounds is not None else "auto",
            max_gpu_hours=goal.get("max_gpu_hours", 24.0),
            max_pilots=goal.get("max_pilot_rounds", 12),
            no_improvement_patience=goal.get("no_improvement_patience", 4),
            max_concurrent_pilots=goal.get("max_concurrent_pilots", 1),
            max_rounds_safety=goal.get("max_auto_rounds_safety", 60),
            full_requires_confirmation=goal.get("full_requires_confirmation", True),
            explicit_rounds=explicit_rounds,
        )

