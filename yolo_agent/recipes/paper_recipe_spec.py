"""Paper-specific recipe specifications.

PaperRecipeSpec is not RecipeSpec. It binds one or more papers to a real
implementation identity. A generic component ID is never enough to cover a
family of papers.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from yolo_agent.core.yaml_io import YAMLModelMixin
from yolo_agent.research.paper_mechanism_resolver import GENERIC_MECHANISM_IDS


PaperRecipeDisposition = Literal[
    "queued",
    "evidence_recovery",
    "implementation_request",
    "incompatible",
    "blocked_runtime",
]
SCHEMA_VERSION = "paper_recipe_spec.v1"


class PaperRecipeError(ValueError):
    """Raised when a paper recipe binding is invalid."""


class PaperRecipeSpec(BaseModel, YAMLModelMixin):
    """One explicit paper-to-recipe binding with an execution fingerprint."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = SCHEMA_VERSION
    recipe_id: str
    paper_ids: list[str] = Field(min_length=1)
    method_profile_ids: list[str] = Field(default_factory=list)
    paper_specific_mechanism_id: str
    canonical_component_ids: list[str] = Field(min_length=1)
    changed_variables: dict[str, Any] = Field(min_length=1)
    runtime_plugin: str
    protocol_hash: str
    required_evidence: list[str] = Field(min_length=1)
    expected_metrics: list[str] = Field(min_length=1)
    stop_conditions: list[str] = Field(min_length=1)
    compatibility_requirements: list[str] = Field(min_length=1)
    execution_fingerprint: str = ""
    target_error_facts: list[dict[str, Any]] = Field(default_factory=list)
    inference_only: bool = False
    model: str = "yolo26n.pt"
    dataset_identity: str = "unspecified"
    teacher_identity: str = "none"
    graph_identity: str = "none"
    runtime_payload_hash: str = "none"
    seed: int = 42
    disposition: PaperRecipeDisposition = "implementation_request"

    @field_validator("recipe_id", "paper_specific_mechanism_id", "runtime_plugin", "protocol_hash")
    @classmethod
    def _required_text(cls, value: str) -> str:
        if not str(value).strip():
            raise ValueError("paper recipe identity fields must not be empty")
        return str(value).strip()

    @field_validator("paper_ids")
    @classmethod
    def _unique_paper_ids(cls, value: list[str]) -> list[str]:
        cleaned = [item.strip() for item in value if item and item.strip()]
        if not cleaned:
            raise ValueError("paper recipe requires paper_ids")
        return list(dict.fromkeys(cleaned))

    @model_validator(mode="after")
    def bind_fingerprint_and_guards(self) -> "PaperRecipeSpec":
        if self.paper_specific_mechanism_id in GENERIC_MECHANISM_IDS:
            raise PaperRecipeError(
                f"generic mechanism cannot be a paper recipe: {self.paper_specific_mechanism_id}"
            )
        if self.paper_specific_mechanism_id.endswith(".general"):
            raise PaperRecipeError(
                f"generic .general mechanism cannot be a paper recipe: {self.paper_specific_mechanism_id}"
            )
        digest = compute_paper_recipe_fingerprint(self)
        if self.execution_fingerprint and self.execution_fingerprint != digest:
            raise PaperRecipeError(
                f"execution_fingerprint mismatch for {self.recipe_id}"
            )
        self.execution_fingerprint = digest
        if not self.target_error_facts and self.disposition == "queued":
            raise PaperRecipeError("empty target_error_facts cannot be queued")
        if self.inference_only and self.disposition == "queued":
            raise PaperRecipeError("inference-only recipe cannot be a train candidate")
        return self

    @property
    def allows_train_queue(self) -> bool:
        return self.disposition == "queued" and not self.inference_only and bool(self.target_error_facts)


def compute_paper_recipe_fingerprint(spec: PaperRecipeSpec) -> str:
    """Hash the implementation identity, not the paper title list."""
    payload = {
        "model": spec.model,
        "paper_specific_mechanism_id": spec.paper_specific_mechanism_id,
        "changed_variables": spec.changed_variables,
        "protocol": spec.protocol_hash,
        "dataset": spec.dataset_identity,
        "teacher": spec.teacher_identity,
        "graph": spec.graph_identity,
        "payload": spec.runtime_payload_hash,
        "seed": spec.seed,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def queue_disposition(
    *,
    target_error_facts: list[dict[str, Any]],
    inference_only: bool,
    has_runtime_adapter: bool,
    incompatible: bool = False,
) -> PaperRecipeDisposition:
    """Deterministic disposition for one paper recipe binding."""
    if incompatible:
        return "incompatible"
    if inference_only:
        return "blocked_runtime"
    if not target_error_facts:
        return "evidence_recovery"
    if not has_runtime_adapter:
        return "implementation_request"
    return "queued"
