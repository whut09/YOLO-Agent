"""Artifact-backed pilot reproduction promotion for paper acceptance."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

from yolo_agent.certification.paper_auto_optimization_research import (
    PaperAcceptanceResearchContext,
    PaperAcceptanceTrackContext,
    load_component_contract,
)
from yolo_agent.certification.paper_auto_optimization_schemas import PaperPairedDelta
from yolo_agent.components.maturity import (
    maturity_artifact,
    transition_maturity,
)
from yolo_agent.components.maturity_registry import ComponentMaturityRegistry
from yolo_agent.core.yaml_io import YAMLModelMixin
from yolo_agent.certification.paper_auto_optimization_tracks import (
    PaperAcceptanceRecipe,
    acceptance_recipe,
)


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
        if self.pilot_3.component_id != self.component_id:
            raise ValueError("pilot_3 component does not match maturity evidence")
        if self.pilot_10.component_id != self.component_id:
            raise ValueError("pilot_10 component does not match maturity evidence")
        if self.pilot_3.recipe_id != self.recipe_id:
            raise ValueError("pilot_3 recipe does not match maturity evidence")
        if self.pilot_10.recipe_id != self.recipe_id:
            raise ValueError("pilot_10 recipe does not match maturity evidence")
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
    """Compatibility wrapper for the sampling-only acceptance suite."""
    track = next(
        (
            item
            for item in research.effective_tracks()
            if item.component_id == "sampling.small_object"
        ),
        None,
    )
    if track is None:
        raise RuntimeError("sampling track is absent from research context")
    return promote_component_pilot_reproduced(
        registry_path=registry_path,
        research=research,
        track=track,
        recipe=acceptance_recipe("sampling.small_object"),
        acceptance_protocol_hash=acceptance_protocol_hash,
        pilot_3=pilot_3,
        pilot_10=pilot_10,
        output_path=output_path,
    )


def promote_component_pilot_reproduced(
    *,
    registry_path: Path | str,
    research: PaperAcceptanceResearchContext,
    track: PaperAcceptanceTrackContext,
    recipe: PaperAcceptanceRecipe,
    acceptance_protocol_hash: str,
    pilot_3: PaperPairedDelta,
    pilot_10: PaperPairedDelta,
    output_path: Path | str,
) -> PaperPilotReproductionEvidence:
    """Promote only an exact certified identity after a passed pilot_10."""
    evidence = PaperPilotReproductionEvidence(
        component_id=track.component_id,
        recipe_id=recipe.recipe_id,
        paper_ids=track.paper_ids,
        adapter_hash=track.adapter_hash,
        snapshot_hash=research.snapshot_hash,
        acceptance_protocol_hash=acceptance_protocol_hash,
        maturity_protocol_hash=track.maturity_protocol_hash,
        pilot_3=pilot_3,
        pilot_10=pilot_10,
    )
    path = Path(output_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    evidence.to_yaml(path, exclude_none=True, sort_keys=False)

    registry = ComponentMaturityRegistry(registry_path)
    contract = load_component_contract(track.component_id)
    effective, resolution, overlay = registry.resolve(
        contract,
        adapter_hash=track.adapter_hash,
        ultralytics_version=track.ultralytics_version,
        protocol_hash=track.maturity_protocol_hash,
    )
    if overlay is None or resolution.status != "applied":
        raise RuntimeError(
            f"certified {track.component_id} maturity overlay is no longer valid"
        )
    if effective.maturity == "pilot_reproduced":
        return evidence
    if effective.maturity != "gpu_certified":
        raise RuntimeError(
            f"{track.component_id} pilot promotion requires gpu_certified; got "
            + effective.maturity
        )
    artifact = maturity_artifact(
        component_id=track.component_id,
        target_maturity="pilot_reproduced",
        artifact_path=path,
        status="passed",
        producer="PaperAutoOptimizationAcceptanceSuite",
        protocol_hash=track.maturity_protocol_hash,
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
        reason=f"verified matched {track.component_id} pilot_3 and pilot_10",
        artifact=artifact,
    )
    registry.record_contract(
        promoted,
        adapter_hash=track.adapter_hash,
        code_commit=overlay.code_commit,
        ultralytics_version=track.ultralytics_version,
        protocol_hash=track.maturity_protocol_hash,
    )
    return evidence


__all__ = [
    "PaperPilotReproductionEvidence",
    "promote_component_pilot_reproduced",
    "promote_sampling_pilot_reproduced",
]
