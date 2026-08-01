"""Artifact-backed pilot reproduction promotion for paper acceptance."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

from yolo_agent.certification.paper_auto_optimization_research import (
    PaperAcceptanceResearchContext,
    load_sampling_contract,
)
from yolo_agent.certification.paper_auto_optimization_schemas import PaperPairedDelta
from yolo_agent.components.maturity import (
    maturity_artifact,
    transition_maturity,
)
from yolo_agent.components.maturity_registry import ComponentMaturityRegistry
from yolo_agent.core.yaml_io import YAMLModelMixin


class PaperPilotReproductionEvidence(BaseModel, YAMLModelMixin):
    """Immutable evidence used for the gpu_certified -> pilot_reproduced step."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "paper_pilot_reproduction_evidence.v1"
    component_id: str = "sampling.small_object"
    recipe_id: str = "sampling.small_object"
    paper_ids: list[str] = Field(min_length=1)
    adapter_hash: str
    snapshot_hash: str
    acceptance_protocol_hash: str
    maturity_protocol_hash: str
    pilot_3: PaperPairedDelta
    pilot_10: PaperPairedDelta
    status: str = "passed"
    evidence_hash: str = ""

    @model_validator(mode="after")
    def validate_evidence(self) -> "PaperPilotReproductionEvidence":
        for result in (self.pilot_3, self.pilot_10):
            if (
                not result.verified
                or not result.protocol_match
                or result.rejection_reasons
                or not result.result_hash
            ):
                raise ValueError(
                    "pilot_reproduced requires verified, promoted paired evidence"
                )
        expected = self.calculate_hash()
        if self.evidence_hash and self.evidence_hash != expected:
            raise ValueError("paper pilot reproduction evidence hash mismatch")
        self.evidence_hash = expected
        return self

    def calculate_hash(self) -> str:
        payload = self.model_dump(mode="json", exclude={"evidence_hash"})
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()


def promote_sampling_pilot_reproduced(
    *,
    registry_path: Path | str,
    research: PaperAcceptanceResearchContext,
    acceptance_protocol_hash: str,
    pilot_3: PaperPairedDelta,
    pilot_10: PaperPairedDelta,
    output_path: Path | str,
) -> PaperPilotReproductionEvidence:
    """Promote only the exact certified adapter identity frozen in the snapshot."""
    evidence = PaperPilotReproductionEvidence(
        paper_ids=research.paper_ids,
        adapter_hash=research.adapter_hash,
        snapshot_hash=research.snapshot_hash,
        acceptance_protocol_hash=acceptance_protocol_hash,
        maturity_protocol_hash=research.maturity_protocol_hash,
        pilot_3=pilot_3,
        pilot_10=pilot_10,
    )
    path = Path(output_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    evidence.to_yaml(path, exclude_none=True, sort_keys=False)

    registry = ComponentMaturityRegistry(registry_path)
    contract = load_sampling_contract()
    effective, resolution, overlay = registry.resolve(
        contract,
        adapter_hash=research.adapter_hash,
        ultralytics_version=research.ultralytics_version,
        protocol_hash=research.maturity_protocol_hash,
    )
    if overlay is None or resolution.status != "applied":
        raise RuntimeError("certified sampling maturity overlay is no longer valid")
    if effective.maturity == "pilot_reproduced":
        return evidence
    if effective.maturity != "gpu_certified":
        raise RuntimeError(
            "sampling pilot promotion requires gpu_certified; got "
            + effective.maturity
        )
    artifact = maturity_artifact(
        component_id=research.component_id,
        target_maturity="pilot_reproduced",
        artifact_path=path,
        status="passed",
        producer="PaperAutoOptimizationAcceptanceSuite",
        protocol_hash=research.maturity_protocol_hash,
        metadata={
            "acceptance_protocol_hash": acceptance_protocol_hash,
            "snapshot_hash": research.snapshot_hash,
            "pilot_3_result_hash": pilot_3.result_hash,
            "pilot_10_result_hash": pilot_10.result_hash,
        },
    )
    promoted = transition_maturity(
        effective,
        "pilot_reproduced",
        reason="verified matched sampling pilot_3 and pilot_10",
        artifact=artifact,
    )
    registry.record_contract(
        promoted,
        adapter_hash=research.adapter_hash,
        code_commit=overlay.code_commit,
        ultralytics_version=research.ultralytics_version,
        protocol_hash=research.maturity_protocol_hash,
    )
    return evidence


__all__ = [
    "PaperPilotReproductionEvidence",
    "promote_sampling_pilot_reproduced",
]
