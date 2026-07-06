"""Configuration schema for optional decision-analysis LLMs."""

from __future__ import annotations

import os
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
    api_key: str | None = None
    base_url: str | None = None
    base_url_env: str | None = None
    timeout_seconds: int = Field(default=60, ge=1)
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    max_output_tokens: int = Field(default=4096, ge=1)
    decision_role: DecisionRole = "proposal_generator_only"
    use_by_default: bool = True
    require_api_key: bool = True
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
        return self.enabled and self.use_by_default and self.decision_role == "proposal_generator_only"

    @property
    def executable_decisions_allowed(self) -> bool:
        """LLM executable decisions are intentionally disallowed."""
        return False

    def resolved_api_key(self) -> str:
        """Resolve the API key from explicit local config or an environment variable."""
        if self.api_key and not _redacted_value(self.api_key):
            return self.api_key
        if _looks_like_api_key(self.api_key_env):
            return self.api_key_env
        return os.getenv(self.api_key_env, "") or _dotenv_value(Path(".env.local"), self.api_key_env)

    def resolved_base_url(self) -> str | None:
        """Resolve the base URL from explicit local config, direct URL, or env var."""
        if self.base_url:
            return self.base_url
        if self.base_url_env and _looks_like_url(self.base_url_env):
            return self.base_url_env
        if not self.base_url_env:
            return None
        return os.getenv(self.base_url_env, "") or _dotenv_value(Path(".env.local"), self.base_url_env) or None

    def api_key_source(self) -> str:
        """Return a safe user-facing description of where the key comes from."""
        if self.api_key and not _redacted_value(self.api_key):
            return "local_config:api_key"
        if _looks_like_api_key(self.api_key_env):
            return "local_config:api_key_env_direct_value"
        return f"env:{self.api_key_env}"

    def base_url_source(self) -> str:
        """Return a safe user-facing description of where the base URL comes from."""
        if self.base_url:
            return "local_config:base_url"
        if self.base_url_env and _looks_like_url(self.base_url_env):
            return "local_config:base_url_env_direct_value"
        return f"env:{self.base_url_env}" if self.base_url_env else "default"

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
    if os.getenv("YOLO_AGENT_DISABLE_LOCAL_LLM", "").strip().lower() in {"1", "true", "yes"}:
        return LLMDecisionConfig.from_yaml(ResourcePaths.LLM_DECISION_EXAMPLE)
    local_path = ResourcePaths.LLM_DECISION_LOCAL
    if local_path.is_file():
        return LLMDecisionConfig.from_yaml(local_path)
    return LLMDecisionConfig.from_yaml(ResourcePaths.LLM_DECISION_EXAMPLE)


def _looks_like_api_key(value: str | None) -> bool:
    if not value:
        return False
    text = value.strip()
    return text.startswith(("sk-", "sk_", "sess-", "key-")) or (len(text) > 24 and "_" not in text and text != "OPENAI_API_KEY")


def _looks_like_url(value: str | None) -> bool:
    if not value:
        return False
    return value.strip().startswith(("http://", "https://"))


def _redacted_value(value: str | None) -> bool:
    return value is None or value.strip() in {"", "XX", "PUT_YOUR_OPENAI_API_KEY_HERE"}


def _dotenv_value(path: Path, name: str) -> str:
    """Read a simple KEY=value from a local .env file without adding a dependency."""
    if not name or not path.is_file():
        return ""
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except OSError:
        return ""
    prefix = f"{name}="
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or not stripped.startswith(prefix):
            continue
        value = stripped[len(prefix) :].strip().strip('"').strip("'")
        return "" if _redacted_value(value) else value
    return ""


__all__ = [
    "LLMDecisionConfig",
    "PromptContract",
    "load_llm_decision_config",
]
