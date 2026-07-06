"""Configuration schema for optional decision-analysis LLMs."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator

from yolo_agent.resources import ResourcePaths


DecisionRole = Literal["proposal_generator_only"]


class PromptContract(BaseModel):
    """Prompt IO contract for decision-analysis model calls."""

    system_summary: str = ""
    required_inputs: list[str] = Field(default_factory=list)
    required_output_schema: dict[str, object] = Field(default_factory=dict)


class LLMDecisionConfig(BaseModel):
    """Optional LLM config for diagnosis/proposal generation.

    This config does not authorize executable decisions. It only describes the
    model that may draft analysis, evidence requests, and policy proposals.
    """

    enabled: bool = False
    provider: str = "XX"
    model: str = "XX"
    model_alias: str | None = None
    api_key_env: str = "XX"
    base_url: str | None = None
    base_url_env: str | None = None
    timeout_seconds: int = Field(default=60, ge=1)
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    max_output_tokens: int = Field(default=4096, ge=1)
    decision_role: DecisionRole = "proposal_generator_only"
    allowed_outputs: list[str] = Field(default_factory=list)
    blocked_outputs: list[str] = Field(default_factory=list)
    safety_contract: list[str] = Field(default_factory=list)
    prompt_contract: PromptContract = Field(default_factory=PromptContract)

    @field_validator("provider", "model", "api_key_env")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("LLM config fields must not be empty.")
        return value

    @property
    def can_generate_proposals(self) -> bool:
        """Return whether this config can be used as a proposal source."""
        return self.enabled and self.decision_role == "proposal_generator_only"

    @property
    def executable_decisions_allowed(self) -> bool:
        """LLM executable decisions are intentionally disallowed."""
        return False

    @classmethod
    def from_yaml(cls, path: Path | str | None = None) -> "LLMDecisionConfig":
        """Load an LLM decision config from YAML."""
        config_path = Path(path) if path is not None else ResourcePaths.LLM_DECISION_LOCAL
        with config_path.open("r", encoding="utf-8-sig") as file:
            data = yaml.safe_load(file) or {}
        if not isinstance(data, dict):
            raise ValueError(f"LLM decision config must be a mapping: {config_path}")
        return cls.model_validate(data)


def load_llm_decision_config(path: Path | str | None = None) -> LLMDecisionConfig:
    """Load local config when available, otherwise load the redacted example."""
    if path is not None:
        return LLMDecisionConfig.from_yaml(path)
    local_path = ResourcePaths.LLM_DECISION_LOCAL
    if local_path.is_file():
        return LLMDecisionConfig.from_yaml(local_path)
    return LLMDecisionConfig.from_yaml(ResourcePaths.LLM_DECISION_EXAMPLE)


__all__ = [
    "LLMDecisionConfig",
    "PromptContract",
    "load_llm_decision_config",
]
