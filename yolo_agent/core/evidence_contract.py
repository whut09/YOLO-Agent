"""Evidence contracts and gates for trustworthy loop decisions."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_serializer

from yolo_agent.core.evidence_index import EvidenceIndex
from yolo_agent.core.experiment_graph import Evidence


NO_EVIDENCE_WARNING = "No evidence, do not trust this result."
EvidenceKind = Literal["artifact", "metric", "config"]

METRIC_NAMES = {
    "map",
    "mAP",
    "mAP_small",
    "map50",
    "map50_95",
    "precision",
    "recall",
    "latency",
    "latency_ms",
    "model_size",
    "model_size_mb",
    "fps",
    "robustness",
    "robustness_score",
    "localization_error_rate",
    "false_negative_count",
    "false_positive_count",
    "label_quality_score",
}
ARTIFACT_ALIASES = {
    "label_quality_report": ["annotation_advice", "annotation_advice.json", "label_quality_report.json"],
    "dataset_report": ["dataset_report", "dataset_report.json"],
    "smoke_result": ["smoke_result", "smoke_result.json"],
    "loop_report": ["loop_report", "loop_report.json", "loop_diagnosis", "loop_diagnosis.json"],
    "loop_plan": ["loop_plan", "loop_plan.yaml"],
    "candidate_plan": ["candidate_plan", "candidate_plan.yaml", "plan.yaml"],
    "ablation_plan": ["ablation_plan", "ablation_plan.yaml"],
    "dataset_version": ["dataset_manifest", "manifest.json"],
    "dataset_manifest": ["dataset_manifest", "manifest.json"],
}


class EvidenceRequirement(BaseModel):
    """One required evidence item."""

    name: str
    kind: EvidenceKind
    required: bool = True
    description: str = ""

    @classmethod
    def from_name(cls, name: str) -> "EvidenceRequirement":
        """Infer requirement kind from a compact evidence name."""
        kind: EvidenceKind = "metric" if name in METRIC_NAMES else "artifact"
        return cls(name=name, kind=kind)


class EvidenceStatus(BaseModel):
    """Presence/absence status for one evidence requirement."""

    name: str
    kind: EvidenceKind
    required: bool = True
    present: bool = False
    path: Path | None = None
    value: Any = None
    message: str = ""

    @field_serializer("path")
    def serialize_path(self, value: Path | None) -> str | None:
        """Serialize paths portably."""
        return value.as_posix() if value is not None else None


class EvidenceGateResult(BaseModel):
    """Batch evidence gate result."""

    ok: bool
    trusted: bool
    statuses: list[EvidenceStatus]
    missing_required: list[str] = Field(default_factory=list)
    warning: str | None = None


class EvidenceContract(BaseModel):
    """Stage-level evidence requirements that can be evaluated independently."""

    requirements: list[EvidenceRequirement] = Field(default_factory=list)

    @classmethod
    def from_names(cls, names: list[str]) -> "EvidenceContract":
        """Build a contract from compact evidence names."""
        return cls(requirements=[EvidenceRequirement.from_name(name) for name in names])

    def evaluate(
        self,
        evidence: Evidence,
        artifacts: dict[str, Path] | None = None,
        config: dict[str, Any] | None = None,
    ) -> EvidenceGateResult:
        """Evaluate this contract against run evidence."""
        return EvidenceGate(self.requirements).evaluate(
            evidence=evidence,
            artifacts=artifacts,
            config=config,
        )


class EvidenceGate:
    """Evaluate whether a run has the evidence required for trusted decisions."""

    def __init__(self, requirements: list[EvidenceRequirement | str]) -> None:
        self.requirements = [
            requirement if isinstance(requirement, EvidenceRequirement) else EvidenceRequirement.from_name(requirement)
            for requirement in requirements
        ]

    def evaluate(
        self,
        evidence: Evidence,
        artifacts: dict[str, Path] | None = None,
        config: dict[str, Any] | None = None,
    ) -> EvidenceGateResult:
        """Evaluate requirements against EvidenceStore output and loop artifacts."""
        artifact_index = _artifact_index(evidence, artifacts or {})
        config_data = config or evidence.config
        statuses = [
            _evaluate_requirement(requirement, evidence, artifact_index, config_data)
            for requirement in self.requirements
        ]
        missing_required = [
            status.name
            for status in statuses
            if status.required and not status.present
        ]
        ok = not missing_required
        return EvidenceGateResult(
            ok=ok,
            trusted=ok,
            statuses=statuses,
            missing_required=missing_required,
            warning=None if ok else NO_EVIDENCE_WARNING,
        )


def default_loop_evidence_requirements(extra: list[str] | None = None) -> list[EvidenceRequirement]:
    """Return default loop-level evidence contract."""
    names = [
        "dataset_report",
        "label_quality_report",
        "smoke_result",
        "latency_ms",
        "map50",
        "recall",
    ]
    names.extend(extra or [])
    return [EvidenceRequirement.from_name(name) for name in list(dict.fromkeys(names))]


def _evaluate_requirement(
    requirement: EvidenceRequirement,
    evidence: Evidence,
    artifacts: dict[str, Path],
    config: dict[str, Any],
) -> EvidenceStatus:
    if requirement.kind == "metric":
        value = evidence.metrics.get(requirement.name)
        metric_record = EvidenceIndex(evidence.metric_records).select_one(
            metric_name=requirement.name,
            verified=True,
        )
        present = value is not None or metric_record is not None
        return EvidenceStatus(
            name=requirement.name,
            kind=requirement.kind,
            required=requirement.required,
            present=present,
            value=value if value is not None else (metric_record.value if metric_record is not None else None),
            message="metric present" if present else f"Missing metric: {requirement.name}",
        )
    if requirement.kind == "config":
        value = config.get(requirement.name)
        return EvidenceStatus(
            name=requirement.name,
            kind=requirement.kind,
            required=requirement.required,
            present=value is not None,
            value=value,
            message="config present" if value is not None else f"Missing config: {requirement.name}",
        )

    path = _find_artifact(requirement.name, artifacts)
    return EvidenceStatus(
        name=requirement.name,
        kind=requirement.kind,
        required=requirement.required,
        present=path is not None and path.is_file(),
        path=path,
        message="artifact present" if path is not None and path.is_file() else f"Missing artifact: {requirement.name}",
    )


def _artifact_index(evidence: Evidence, loop_artifacts: dict[str, Path]) -> dict[str, Path]:
    artifacts: dict[str, Path] = {}
    artifacts.update(loop_artifacts)
    artifacts.update(evidence.artifacts)
    for entry in evidence.artifact_manifest:
        aliases = {entry.name, entry.path.name, entry.path.stem}
        if entry.verify():
            for alias in aliases:
                artifacts[alias] = entry.path
            continue
        for alias in aliases:
            artifacts.pop(alias, None)
    for key, path in list(artifacts.items()):
        artifacts[Path(key).stem] = path
        artifacts[path.name] = path
        artifacts[path.stem] = path
    return artifacts


def _find_artifact(name: str, artifacts: dict[str, Path]) -> Path | None:
    for candidate in [name, *ARTIFACT_ALIASES.get(name, [])]:
        path = artifacts.get(candidate)
        if path is not None:
            return path
    return None
