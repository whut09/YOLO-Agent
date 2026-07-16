"""Canonical doctor-style decision context and one-LLM bundle per round."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field

from yolo_agent.agents.strategy_policy import CandidatePolicy
from yolo_agent.core.yaml_io import YAMLModelMixin


class DecisionContext(BaseModel):
    """All evidence and bounded action space supplied to the round LLM."""

    schema_version: str = "decision_context.v1"
    run_id: str
    research_snapshot_hash: str | None = None
    research_snapshot_path: str | None = None
    research_snapshot_verified: bool = False
    baseline_evidence: list[dict[str, Any]] = Field(default_factory=list)
    current_evidence: list[dict[str, Any]] = Field(default_factory=list)
    error_delta: dict[str, Any] = Field(default_factory=dict)
    diagnosis: dict[str, Any] = Field(default_factory=dict)
    paper_candidates: list[dict[str, Any]] = Field(default_factory=list)
    deterministic_recipe_candidates: list[dict[str, Any]] = Field(default_factory=list)
    executable_adapters: list[str] = Field(default_factory=list)
    component_maturity: dict[str, str] = Field(default_factory=dict)
    compatibility: dict[str, Any] = Field(default_factory=dict)
    policy_memory: dict[str, Any] = Field(default_factory=dict)
    tried_actions: list[str] = Field(default_factory=list)
    rejected_actions: list[str] = Field(default_factory=list)
    objective: dict[str, Any] = Field(default_factory=dict)
    budget: dict[str, Any] = Field(default_factory=dict)
    guardrails: list[str] = Field(default_factory=list)
    missing_evidence: list[str] = Field(default_factory=list)
    fallback_policies: list[CandidatePolicy] = Field(default_factory=list)

    @property
    def context_hash(self) -> str:
        payload = self.model_dump(mode="json")
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


class LLMDecisionBundle(BaseModel, YAMLModelMixin):
    """One canonical LLM decision and its deterministic downstream outcome."""

    schema_version: str = "llm_decision_bundle.v1"
    run_id: str
    context: DecisionContext
    llm_status: Literal["used", "skipped", "failed"]
    provider: str
    model: str
    prompt_sha256: str | None = None
    doctor_report_draft: dict[str, Any] | None = None
    proposed_policies: list[CandidatePolicy] = Field(default_factory=list)
    evidence_requests: list[dict[str, Any]] = Field(default_factory=list)
    rejected_actions: list[dict[str, Any]] = Field(default_factory=list)
    critic_result: dict[str, Any] = Field(default_factory=dict)
    critic_accepted_policy_ids: list[str] = Field(default_factory=list)
    selected_for_evaluation_policy_ids: list[str] = Field(default_factory=list)
    decision_mode: Literal["llm", "deterministic_fallback"]
    warnings: list[str] = Field(default_factory=list)
    deterministic_outcome: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def decision_hash(self) -> str:
        """Hash the LLM decision boundary, excluding later evaluator outcomes."""
        payload = self.model_dump(
            mode="json",
            exclude={"created_at", "deterministic_outcome"},
        )
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()
