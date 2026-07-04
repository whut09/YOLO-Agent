"""Stage contracts for the loop harness state machine."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from yolo_agent.core.artifact_manifest import ArtifactManifestEntry
from yolo_agent.core.evidence_contract import EvidenceContract, EvidenceGateResult
from yolo_agent.core.experiment_graph import Evidence
from yolo_agent.core.loop_state import LoopStage


RetryBackoff = Literal["none", "linear", "exponential"]
ArtifactFreshness = Literal["any", "current_run"]


class RetryPolicy(BaseModel):
    """Retry behavior for a loop stage."""

    max_attempts: int = Field(default=1, ge=1)
    backoff: RetryBackoff = "none"


class StageContractCheck(BaseModel):
    """Validation result for one stage contract."""

    ok: bool
    missing_required: list[str] = Field(default_factory=list)
    missing_evidence: list[str] = Field(default_factory=list)
    invalid_artifacts: list[str] = Field(default_factory=list)
    evidence_gate: EvidenceGateResult | None = None
    warnings: list[str] = Field(default_factory=list)


class ArtifactContract(BaseModel):
    """Trust contract for one artifact dependency."""

    model_config = ConfigDict(populate_by_name=True)

    schema_name: str | None = Field(default=None, alias="schema")
    sha_required: bool = True
    freshness: ArtifactFreshness = "current_run"

    def validate(
        self,
        name: str,
        manifest_entries: list[ArtifactManifestEntry],
        current_run_dir: Path,
    ) -> list[str]:
        """Return validation errors for one artifact."""
        entry = _latest_manifest_entry(name, manifest_entries)
        if entry is None:
            return [f"{name}: missing manifest entry"]
        errors: list[str] = []
        if self.sha_required and not entry.verify():
            errors.append(f"{name}: sha256 verification failed")
        if self.freshness == "current_run" and not _is_relative_to(entry.path, current_run_dir):
            errors.append(f"{name}: artifact is not from current run")
        if self.schema_name:
            schema_error = _validate_artifact_schema(entry.path, self.schema_name)
            if schema_error:
                errors.append(f"{name}: {schema_error}")
        return errors


class StageContract(BaseModel):
    """Executable contract for one loop stage."""

    id: LoopStage
    description: str = ""
    requires: list[str] = Field(default_factory=list)
    provides: list[str] = Field(default_factory=list)
    evidence_required: list[str] = Field(default_factory=list)
    block_on_missing: bool = True
    block_on_missing_evidence: bool = True
    retry_policy: RetryPolicy = Field(default_factory=RetryPolicy)
    producer_artifacts: dict[str, str] = Field(default_factory=dict)
    artifact_contract: dict[str, ArtifactContract] = Field(default_factory=dict)

    def check(
        self,
        available: set[str],
        manifest_entries: list[ArtifactManifestEntry] | None = None,
        current_run_dir: Path | str | None = None,
        evidence: Evidence | None = None,
        evidence_artifacts: dict[str, Path] | None = None,
        evidence_config: dict[str, Any] | None = None,
    ) -> StageContractCheck:
        """Check whether required inputs and trusted evidence are available."""
        missing = [requirement for requirement in self.requires if requirement not in available]
        invalid_artifacts = self._check_artifact_contracts(
            available=available,
            manifest_entries=manifest_entries or [],
            current_run_dir=Path(current_run_dir) if current_run_dir is not None else None,
        )
        evidence_gate = self._check_evidence_contract(
            evidence=evidence,
            evidence_artifacts=evidence_artifacts,
            evidence_config=evidence_config,
        )
        missing_evidence = evidence_gate.missing_required if evidence_gate is not None else []
        if not missing and not missing_evidence and not invalid_artifacts:
            return StageContractCheck(ok=True, evidence_gate=evidence_gate)
        warnings = [f"Missing required input for {self.id}: {item}" for item in missing]
        warnings.extend(f"Missing required evidence for {self.id}: {item}" for item in missing_evidence)
        warnings.extend(f"Invalid artifact for {self.id}: {item}" for item in invalid_artifacts)
        input_ok = not missing or not self.block_on_missing
        evidence_ok = not missing_evidence or not self.block_on_missing_evidence
        return StageContractCheck(
            ok=not invalid_artifacts and input_ok and evidence_ok,
            missing_required=missing,
            missing_evidence=missing_evidence,
            invalid_artifacts=invalid_artifacts,
            evidence_gate=evidence_gate,
            warnings=warnings,
        )

    def _check_artifact_contracts(
        self,
        available: set[str],
        manifest_entries: list[ArtifactManifestEntry],
        current_run_dir: Path | None,
    ) -> list[str]:
        if current_run_dir is None:
            return []
        errors: list[str] = []
        for name, contract in self.artifact_contract.items():
            if name not in self.requires or name not in available:
                continue
            errors.extend(contract.validate(name, manifest_entries, current_run_dir))
        return errors

    def _check_evidence_contract(
        self,
        evidence: Evidence | None,
        evidence_artifacts: dict[str, Path] | None,
        evidence_config: dict[str, Any] | None,
    ) -> EvidenceGateResult | None:
        if evidence is None or not self.evidence_required:
            return None
        return EvidenceContract.from_names(self.evidence_required).evaluate(
            evidence=evidence,
            artifacts=evidence_artifacts or {},
            config=evidence_config,
        )


class LoopStageContracts(BaseModel):
    """Stage contracts loaded from loop policy YAML."""

    stages: list[StageContract]
    policy_budget: dict[str, Any] = Field(default_factory=dict)

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


def _latest_manifest_entry(
    name: str,
    entries: list[ArtifactManifestEntry],
) -> ArtifactManifestEntry | None:
    for entry in reversed(entries):
        if entry.name == name:
            return entry
    for entry in reversed(entries):
        if entry.path.name == name:
            return entry
    aliases = {Path(name).name, Path(name).stem}
    for entry in reversed(entries):
        entry_aliases = {entry.path.name, entry.path.stem}
        if aliases & entry_aliases:
            return entry
    return None


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def _validate_artifact_schema(path: Path, schema: str) -> str | None:
    model = _schema_model(schema)
    if model is None:
        return f"unknown schema {schema}"
    try:
        model.model_validate(_read_structured_artifact(path))
    except Exception as exc:  # pragma: no cover - pydantic/json/yaml error details vary
        return f"schema {schema} validation failed: {exc}"
    return None


def _read_structured_artifact(path: Path) -> Any:
    suffix = path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        with path.open("r", encoding="utf-8-sig") as file:
            return yaml.safe_load(file) or {}
    if suffix == ".json":
        with path.open("r", encoding="utf-8-sig") as file:
            return json.load(file)
    raise ValueError(f"unsupported schema validation format: {path.suffix}")


def _schema_model(schema: str) -> type[BaseModel] | None:
    if schema == "DatasetReport":
        from yolo_agent.tools.dataset_stats import DatasetReport

        return DatasetReport
    if schema == "AnnotationAdviceReport":
        from yolo_agent.agents.annotation_advisor import AnnotationAdviceReport

        return AnnotationAdviceReport
    if schema == "ErrorDrivenLoopReport":
        from yolo_agent.agents.error_driven_loop import ErrorDrivenLoopReport

        return ErrorDrivenLoopReport
    if schema == "LoopPolicyEvaluationReport":
        from yolo_agent.agents.loop_policy_evaluator import LoopPolicyEvaluationReport

        return LoopPolicyEvaluationReport
    if schema == "ExperimentPlan":
        from yolo_agent.core.experiment_graph import ExperimentPlan

        return ExperimentPlan
    if schema == "CandidatePlan":
        from yolo_agent.agents.candidate_generator import CandidatePlan

        return CandidatePlan
    if schema == "AblationPlan":
        from yolo_agent.agents.ablation_planner import AblationPlan

        return AblationPlan
    if schema == "SmokeRunResult":
        from yolo_agent.tools.smoke_runner import SmokeRunResult

        return SmokeRunResult
    if schema == "ActiveLearningPlan":
        from yolo_agent.agents.active_learning import ActiveLearningPlan

        return ActiveLearningPlan
    if schema == "LabelingManifest":
        from yolo_agent.agents.active_learning import LabelingManifest

        return LabelingManifest
    if schema == "EvidenceGateResult":
        from yolo_agent.core.evidence_contract import EvidenceGateResult

        return EvidenceGateResult
    return None
