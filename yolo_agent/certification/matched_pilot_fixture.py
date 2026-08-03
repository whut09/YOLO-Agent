"""Matched pilot fixture emitted after real adapter GPU certification."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from yolo_agent.certification.component_schemas import (
    ComponentCertificationReport,
    ComponentGPUCertificationEvidence,
)
from yolo_agent.certification.paper_adapter_factory_schemas import (
    AdapterCertificationIdentity,
)
from yolo_agent.core.yaml_io import YAMLModelMixin


class MatchedPilotArm(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["baseline_control", "candidate"]
    model: str
    dataset: str
    fixture_manifest_hash: str
    seed: int
    epochs: int = Field(ge=1)
    batch: int = Field(ge=1)
    imgsz: Literal[640] = 640
    ultralytics_version: str
    eval_protocol_hash: str
    adapter_hash: str | None = None


class MatchedPilotCertificationFixture(BaseModel, YAMLModelMixin):
    """Protocol-only fixture; it is not paired local outcome evidence."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "matched_component_pilot_fixture.v1"
    component_id: str
    certification_protocol_hash: str
    baseline: MatchedPilotArm
    candidate: MatchedPilotArm
    local_metric_claim_allowed: Literal[False] = False
    maturity_ceiling: Literal["gpu_certified"] = "gpu_certified"
    fixture_hash: str = ""

    @model_validator(mode="after")
    def _validate_match(self) -> "MatchedPilotCertificationFixture":
        fields = (
            "model",
            "dataset",
            "fixture_manifest_hash",
            "seed",
            "epochs",
            "batch",
            "imgsz",
            "ultralytics_version",
            "eval_protocol_hash",
        )
        mismatched = [
            name
            for name in fields
            if getattr(self.baseline, name) != getattr(self.candidate, name)
        ]
        if mismatched:
            raise ValueError("matched pilot fixture mismatch: " + ", ".join(mismatched))
        if self.baseline.adapter_hash is not None or not self.candidate.adapter_hash:
            raise ValueError("only the matched pilot candidate may bind an adapter hash")
        expected = self.calculate_hash()
        if self.fixture_hash and self.fixture_hash != expected:
            raise ValueError("matched pilot fixture hash mismatch")
        self.fixture_hash = expected
        return self

    def calculate_hash(self) -> str:
        payload = self.model_dump(mode="json", exclude={"fixture_hash"})
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        ).hexdigest()


class MatchedPilotFixtureBuilder:
    def build(
        self,
        *,
        report: ComponentCertificationReport,
        identity: AdapterCertificationIdentity,
        model: str,
        data: str,
        output: Path,
    ) -> MatchedPilotCertificationFixture:
        if report.status != "passed" or report.final_maturity != "gpu_certified":
            raise ValueError("matched pilot fixture requires passed gpu certification")
        if (
            report.component_id != identity.component_id
            or report.adapter_hash != identity.adapter_hash
            or report.protocol_hash != identity.protocol_hash
            or report.ultralytics_version != identity.ultralytics_version
        ):
            raise ValueError("GPU report identity does not match reusable adapter")
        evidence_path = report.generated_paths.get("gpu_evidence")
        if evidence_path is None or not evidence_path.is_file():
            raise ValueError("matched pilot fixture requires GPU evidence artifact")
        evidence = ComponentGPUCertificationEvidence.from_yaml(evidence_path)
        protocol = evidence.gpu_protocol
        eval_hash = _eval_protocol_hash(
            model=model,
            fixture_manifest_hash=protocol.fixture_manifest_hash,
            imgsz=protocol.imgsz,
        )
        common = {
            "model": model,
            "dataset": data,
            "fixture_manifest_hash": protocol.fixture_manifest_hash,
            "seed": 0,
            "epochs": protocol.initial_epochs,
            "batch": protocol.batch,
            "imgsz": protocol.imgsz,
            "ultralytics_version": identity.ultralytics_version,
            "eval_protocol_hash": eval_hash,
        }
        fixture = MatchedPilotCertificationFixture(
            component_id=identity.component_id,
            certification_protocol_hash=identity.protocol_hash,
            baseline=MatchedPilotArm(role="baseline_control", **common),
            candidate=MatchedPilotArm(
                role="candidate", adapter_hash=identity.adapter_hash, **common
            ),
        )
        fixture.to_yaml(output, exclude_none=True, sort_keys=False)
        return fixture


def _eval_protocol_hash(
    *, model: str, fixture_manifest_hash: str, imgsz: int
) -> str:
    payload = {
        "schema_version": "matched_component_eval_protocol.v1",
        "model": model,
        "fixture_manifest_hash": fixture_manifest_hash,
        "imgsz": imgsz,
        "metric_source": "coco_post_eval",
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


__all__ = [
    "MatchedPilotArm",
    "MatchedPilotCertificationFixture",
    "MatchedPilotFixtureBuilder",
]
