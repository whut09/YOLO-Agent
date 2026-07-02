"""Stage contracts for the loop harness state machine."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

from yolo_agent.core.loop_state import LoopStage


RetryBackoff = Literal["none", "linear", "exponential"]


class RetryPolicy(BaseModel):
    """Retry behavior for a loop stage."""

    max_attempts: int = Field(default=1, ge=1)
    backoff: RetryBackoff = "none"


class StageContractCheck(BaseModel):
    """Validation result for one stage contract."""

    ok: bool
    missing_required: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class StageContract(BaseModel):
    """Executable contract for one loop stage."""

    id: LoopStage
    description: str = ""
    requires: list[str] = Field(default_factory=list)
    provides: list[str] = Field(default_factory=list)
    evidence_required: list[str] = Field(default_factory=list)
    block_on_missing: bool = True
    retry_policy: RetryPolicy = Field(default_factory=RetryPolicy)
    producer_artifacts: dict[str, str] = Field(default_factory=dict)

    def check(self, available: set[str]) -> StageContractCheck:
        """Check whether required inputs are available."""
        missing = [requirement for requirement in self.requires if requirement not in available]
        if not missing:
            return StageContractCheck(ok=True)
        warnings = [f"Missing required input for {self.id}: {item}" for item in missing]
        return StageContractCheck(
            ok=not self.block_on_missing,
            missing_required=missing,
            warnings=warnings,
        )


class LoopStageContracts(BaseModel):
    """Stage contracts loaded from loop policy YAML."""

    stages: list[StageContract]

    @classmethod
    def from_yaml(cls, path: str | Path) -> "LoopStageContracts":
        """Load stage contracts from loop policy YAML."""
        policy_path = Path(path)
        with policy_path.open("r", encoding="utf-8-sig") as file:
            data = yaml.safe_load(file) or {}
        if not isinstance(data, dict):
            raise ValueError(f"Loop policy YAML must contain a mapping: {policy_path}")
        return cls.model_validate(data)

    @property
    def stage_order(self) -> list[LoopStage]:
        """Return configured stage order."""
        return [stage.id for stage in self.stages]

    def get(self, stage: LoopStage) -> StageContract:
        """Return the contract for a stage."""
        for contract in self.stages:
            if contract.id == stage:
                return contract
        raise KeyError(f"No stage contract configured for {stage}")
