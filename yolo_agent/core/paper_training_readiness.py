"""Final offline authorization gate before a real paper training run.

The gate is intentionally file-backed and side-effect free with respect to
training.  It does not probe CUDA, create assignments, or infer readiness
from a paper count.  It joins the independently produced evidence artifacts
and makes missing or stale evidence explicit for every paper.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from yolo_agent.agents.asha_scheduler import ASHAStudy
from yolo_agent.certification.paper_readiness import PaperReadinessReport
from yolo_agent.core.yaml_io import YAMLModelMixin
from yolo_agent.research.paper_asset_schemas import PaperAssetRegistry
from yolo_agent.research.paper_execution_requirement_schemas import (
    PaperExecutionRequirementsMatrix,
)
from yolo_agent.research.paper_execution_schemas import PaperExecutionInventory


TRAINING_READINESS_SCHEMA_VERSION = "paper_training_readiness.v1"
_RUNNABLE_TRIAL_STATUSES = {
    "waiting",
    "running",
    "promotion_pending",
    "full_pending_confirmation",
    "confirmation_pending",
}
_INFERENCE_PREFIX = "inference."


class PaperTrainingReadinessRecord(BaseModel):
    """Final authorization state for one paper identity."""

    model_config = ConfigDict(extra="forbid")

    paper_id: str
    profile_id: str
    mechanism_id: str | None = None
    recipe_id: str | None = None
    execution_fingerprint: str
    readiness_state: str
    asha_eligibility: bool
    training_allowed: bool = False
    inference_only: bool = False
    cpu_checks_passed: bool = False
    runtime_checks_passed: bool = False
    matched_control_ready: bool = False
    asset_available: bool = False
    asha_trial_id: str | None = None
    asha_assignment_ids: list[str] = Field(default_factory=list)
    blocker: str | None = None
    recovery_action: str | None = None


class PaperTrainingReadinessReport(BaseModel, YAMLModelMixin):
    """Persistent, non-training authorization report."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = TRAINING_READINESS_SCHEMA_VERSION
    status: str
    training_allowed: bool = False
    training_started: bool = False
    gpu_probe: str = "not_run"
    inventory_path: str
    requirements_path: str
    assets_path: str
    readiness_path: str
    asha_path: str
    inventory_hash: str
    requirements_file_hash: str
    asset_registry_hash: str
    readiness_report_hash: str
    paper_count: int
    asha_eligible_count: int = 0
    asha_registered_count: int = 0
    runnable_assignment_count: int = 0
    pre_registered_count: int = 0
    blocked_count: int = 0
    deferred_count: int = 0
    registration_failures_by_paper_id: dict[str, list[str]] = Field(default_factory=dict)
    records: list[PaperTrainingReadinessRecord] = Field(default_factory=list)
    blockers: list[str] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    report_hash: str = ""

    @model_validator(mode="after")
    def validate_report(self) -> "PaperTrainingReadinessReport":
        ids = [item.paper_id for item in self.records]
        if len(ids) != len(set(ids)):
            raise ValueError("paper training readiness contains duplicate paper IDs")
        if ids != sorted(ids):
            raise ValueError("paper training readiness records must be sorted")
        if self.paper_count != len(ids):
            raise ValueError("paper_count must equal record count")
        if self.asha_eligible_count != sum(item.asha_eligibility for item in self.records):
            raise ValueError("asha_eligible_count does not match records")
        if self.pre_registered_count != sum(
            item.asha_trial_id is not None for item in self.records
        ):
            raise ValueError("pre_registered_count does not match records")
        if self.training_started:
            raise ValueError("paper training readiness cannot report training_started")
        if self.gpu_probe != "not_run":
            raise ValueError("paper training readiness must not probe GPU")
        if self.report_hash and self.report_hash != self.calculate_hash():
            raise ValueError("paper training readiness report hash mismatch")
        return self

    def calculate_hash(self) -> str:
        payload = self.model_dump(mode="json", exclude={"report_hash", "generated_at"})
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    def with_hash(self) -> "PaperTrainingReadinessReport":
        return self.model_copy(update={"report_hash": self.calculate_hash()})


def build_paper_training_readiness(
    *,
    inventory_path: Path | str,
    requirements_path: Path | str,
    assets_path: Path | str,
    readiness_path: Path | str,
    asha_path: Path | str,
    output_path: Path | str = Path("runs/paper-readiness/paper_training_readiness.yaml"),
    expected_paper_count: int = 83,
) -> PaperTrainingReadinessReport:
    """Join all preflight artifacts and return the final training gate."""
    inventory_file = _existing_file(inventory_path, "inventory")
    requirements_file = _existing_file(requirements_path, "requirements")
    assets_file = _existing_file(assets_path, "assets")
    readiness_file = _existing_file(readiness_path, "readiness")
    asha_file = _existing_file(asha_path, "asha")

    inventory = PaperExecutionInventory.from_yaml(inventory_file)
    requirements = PaperExecutionRequirementsMatrix.from_yaml(requirements_file)
    assets = PaperAssetRegistry.from_yaml(assets_file)
    readiness = PaperReadinessReport.from_yaml(readiness_file)
    study = ASHAStudy.from_yaml(asha_file)

    if inventory.compatible_paper_count != expected_paper_count:
        raise ValueError(
            f"training readiness requires {expected_paper_count} papers; "
            f"got {inventory.compatible_paper_count}"
        )
    inventory_ids = {item.paper_id for item in inventory.records}
    for label, ids in (
        ("requirements", {item.paper_id for item in requirements.requirements}),
        ("assets", {item.paper_id for item in assets.records}),
        ("readiness", {item.paper_id for item in readiness.records}),
    ):
        if ids != inventory_ids:
            missing = sorted(inventory_ids - ids)
            extra = sorted(ids - inventory_ids)
            raise ValueError(
                f"{label} paper coverage mismatch: missing={missing}, extra={extra}"
            )
    if requirements.source_inventory_hash != inventory.inventory_hash:
        raise ValueError("requirements source inventory hash does not match inventory")
    if assets.source_inventory_hash != inventory.inventory_hash:
        raise ValueError("asset registry source inventory hash does not match inventory")
    if assets.source_requirements_hash != _file_hash(requirements_file):
        raise ValueError("asset registry source requirements hash is stale")
    if readiness.inventory_hash != inventory.inventory_hash:
        raise ValueError("readiness report inventory hash does not match inventory")
    requirements_hash = _semantic_hash(requirements)
    if readiness.requirements_hash not in {"missing", requirements_hash}:
        raise ValueError("readiness report requirements hash does not match requirements")
    if readiness.asset_registry_hash not in {"missing", assets.registry_hash, assets.calculate_hash()}:
        raise ValueError("readiness report asset registry hash does not match assets")

    inventory_by_id = {item.paper_id: item for item in inventory.records}
    requirement_by_id = {item.paper_id: item for item in requirements.requirements}
    asset_by_id = {item.paper_id: item for item in assets.records}
    readiness_by_id = {item.paper_id: item for item in readiness.records}
    trials_by_fingerprint: dict[str, list[Any]] = {}
    for trial in study.trials:
        trials_by_fingerprint.setdefault(trial.execution_fingerprint, []).append(trial)

    records: list[PaperTrainingReadinessRecord] = []
    failure_by_paper: dict[str, list[str]] = {}
    for paper_id in sorted(inventory_ids):
        item = inventory_by_id[paper_id]
        requirement = requirement_by_id[paper_id]
        asset = asset_by_id[paper_id]
        preflight = readiness_by_id[paper_id]
        inference_only = _is_inference_only(item, requirement, preflight)
        candidate_trials = _matching_trials(
            item.execution_fingerprint,
            trials_by_fingerprint,
        )
        active_trial = next(
            (
                trial
                for trial in candidate_trials
                if trial.readiness_state == "asha_eligible"
                and trial.status in _RUNNABLE_TRIAL_STATUSES
                and not trial.readiness_blockers
            ),
            None,
        )
        assignments = [
            assignment
            for assignment in study.assignments
            if assignment.trial_id in {trial.trial_id for trial in candidate_trials}
            and assignment.status in {"issued", "running"}
        ]
        trial = active_trial or (candidate_trials[0] if candidate_trials else None)
        blocker = _paper_blocker(
            item=item,
            requirement=requirement,
            asset=asset,
            preflight=preflight,
            inference_only=inference_only,
            active_trial=active_trial,
        )
        allowed = active_trial is not None and blocker is None and not inference_only
        if blocker:
            failure_by_paper.setdefault(paper_id, []).append(blocker)
        recovery = _recovery_action(blocker)
        records.append(
            PaperTrainingReadinessRecord(
                paper_id=paper_id,
                profile_id=item.profile_id,
                mechanism_id=(
                    item.paper_specific_mechanism_ids[0]
                    if item.paper_specific_mechanism_ids
                    else None
                ),
                recipe_id=item.recipe_ids[0] if item.recipe_ids else None,
                execution_fingerprint=item.execution_fingerprint,
                readiness_state=preflight.readiness_state or "blocked",
                asha_eligibility=preflight.asha_eligibility,
                training_allowed=allowed,
                inference_only=inference_only,
                cpu_checks_passed=preflight.cpu_checks_passed is True,
                runtime_checks_passed=preflight.runtime_checks_passed is True,
                matched_control_ready=preflight.matched_control_readiness.passed,
                asset_available=asset.availability == "available",
                asha_trial_id=trial.trial_id if trial else None,
                asha_assignment_ids=[item.assignment_id for item in assignments],
                blocker=blocker,
                recovery_action=recovery,
            )
        )

    records.sort(key=lambda item: item.paper_id)
    eligible_count = sum(item.asha_eligibility for item in records)
    registered = len(
        {
            item.asha_trial_id
            for item in records
            if item.asha_trial_id is not None
            and any(
                trial.trial_id == item.asha_trial_id
                and trial.readiness_state == "asha_eligible"
                and trial.status in _RUNNABLE_TRIAL_STATUSES
                for trial in study.trials
            )
        }
    )
    runnable_assignments = sum(
        assignment.status in {"issued", "running"}
        for assignment in study.assignments
    )
    blockers = sorted(
        {
            item.blocker
            for item in records
            if item.blocker
        }
    )
    training_allowed = bool(eligible_count and all(item.training_allowed for item in records if item.asha_eligibility))
    report = PaperTrainingReadinessReport(
        status="ready" if training_allowed else "blocked",
        training_allowed=training_allowed,
        inventory_path=str(inventory_file),
        requirements_path=str(requirements_file),
        assets_path=str(assets_file),
        readiness_path=str(readiness_file),
        asha_path=str(asha_file),
        inventory_hash=inventory.inventory_hash,
        requirements_file_hash=_file_hash(requirements_file),
        asset_registry_hash=assets.registry_hash or assets.calculate_hash(),
        readiness_report_hash=readiness.report_hash or readiness.calculate_hash(),
        paper_count=len(records),
        asha_eligible_count=eligible_count,
        asha_registered_count=registered,
        runnable_assignment_count=runnable_assignments,
        pre_registered_count=sum(item.asha_trial_id is not None for item in records),
        blocked_count=sum(item.blocker is not None for item in records),
        deferred_count=sum(
            item.readiness_state == "pre_registered" and item.blocker is None
            for item in records
        ),
        registration_failures_by_paper_id=failure_by_paper,
        records=records,
        blockers=blockers,
    ).with_hash()
    report.to_yaml(output_path, exclude_none=True, sort_keys=False)
    return report


def _existing_file(path: Path | str, label: str) -> Path:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise ValueError(f"{label} artifact does not exist: {resolved}")
    return resolved


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _semantic_hash(model: BaseModel) -> str:
    payload = model.model_dump(mode="json")
    payload.pop("generated_at", None)
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _matching_trials(
    fingerprint: str,
    by_fingerprint: dict[str, list[Any]],
) -> list[Any]:
    # Paper provenance is not execution identity.  A paper may have several
    # independent implementations, so falling back to paper_id could
    # authorize the wrong trial.
    return list(by_fingerprint.get(fingerprint, []))


def _is_inference_only(item: Any, requirement: Any, preflight: Any) -> bool:
    return bool(
        preflight.inference_only
        or requirement.execution_route == "inference"
        or any(str(component).startswith(_INFERENCE_PREFIX) for component in item.canonical_component_ids)
    )


def _paper_blocker(
    *, item: Any, requirement: Any, asset: Any, preflight: Any,
    inference_only: bool, active_trial: Any,
) -> str | None:
    if inference_only:
        return "inference_only_not_training_candidate"
    if not preflight.asha_eligibility or preflight.readiness_state != "asha_eligible":
        return preflight.exact_blocker or "readiness_report_not_asha_eligible"
    if not preflight.cpu_checks_passed:
        return "cpu_readiness_incomplete"
    if not preflight.runtime_checks_passed:
        return "runtime_readiness_incomplete"
    if not preflight.matched_control_readiness.passed:
        return "matched_baseline_not_ready"
    if not requirement.training_candidate_allowed or requirement.execution_route != "training":
        return requirement.exact_blocker or "training_route_not_allowed"
    if asset.availability != "available":
        return asset.exact_blocker or "paper_assets_unavailable"
    if active_trial is None:
        return "asha_eligible_paper_missing_runnable_trial"
    trial_blocker = _trial_identity_blocker(
        item=item,
        requirement=requirement,
        preflight=preflight,
        asset=asset,
        trial=active_trial,
    )
    if trial_blocker:
        return trial_blocker
    return None


def _trial_identity_blocker(
    *, item: Any, requirement: Any, preflight: Any, asset: Any, trial: Any
) -> str | None:
    """Recheck persisted trial identity instead of trusting an old ASHA file."""
    control = trial.baseline_control_node
    if control is None:
        return "asha_trial_matched_control_missing"
    source_metadata = _node_metadata(trial.source_node)
    control_metadata = _node_metadata(control)
    source_protocol = _first_value(
        source_metadata, "baseline_protocol_hash", "run_protocol_hash", "protocol_hash"
    )
    control_protocol = _first_value(
        control_metadata, "baseline_protocol_hash", "run_protocol_hash", "protocol_hash"
    )
    expected_protocols = {
        value
        for value in (
            requirement.protocol_hash,
            preflight.protocol_hash,
            asset.protocol_hash,
        )
        if value and value != "missing"
    }
    if len(expected_protocols) > 1:
        return "paper_protocol_evidence_mismatch"
    expected_protocol = next(iter(expected_protocols), "")
    if not source_protocol or not control_protocol:
        return "asha_trial_protocol_hash_missing"
    if source_protocol != control_protocol:
        return "asha_trial_matched_baseline_protocol_mismatch"
    if expected_protocol and source_protocol != expected_protocol:
        return "asha_trial_protocol_hash_mismatch"

    source_dataset = _first_value(
        source_metadata, "dataset_manifest_hash", "dataset_manifest_sha256"
    )
    control_dataset = _first_value(
        control_metadata, "dataset_manifest_hash", "dataset_manifest_sha256"
    )
    if not source_dataset or not control_dataset:
        return "asha_trial_dataset_manifest_hash_missing"
    if source_dataset != control_dataset:
        return "asha_trial_matched_baseline_dataset_mismatch"
    if preflight.dataset_manifest_hash not in {"missing", source_dataset}:
        return "asha_trial_dataset_manifest_hash_mismatch"

    source_split = _first_value(source_metadata, "split", "evaluation_split")
    control_split = _first_value(control_metadata, "split", "evaluation_split")
    if not source_split or not control_split:
        return "asha_trial_split_missing"
    if source_split != control_split:
        return "asha_trial_matched_baseline_split_mismatch"
    source_imgsz = _node_imgsz(trial.source_node)
    control_imgsz = _node_imgsz(control)
    if source_imgsz != 640 or control_imgsz != 640:
        return "asha_trial_imgsz_must_be_640"
    return None


def _node_metadata(node: Any) -> dict[str, object]:
    command = getattr(node, "command_spec", None)
    return dict(getattr(command, "metadata", {}) or {})


def _first_value(metadata: dict[str, object], *keys: str) -> str:
    for key in keys:
        value = metadata.get(key)
        if value is not None and str(value).strip() and str(value) != "unknown":
            return str(value)
    return ""


def _node_imgsz(node: Any) -> int | None:
    metadata = _node_metadata(node)
    raw = metadata.get("imgsz")
    command = getattr(node, "command_spec", None)
    for argument in getattr(command, "args", []):
        if str(argument).startswith("imgsz="):
            raw = str(argument).split("=", 1)[1]
            break
    try:
        return int(raw) if raw is not None else None
    except (TypeError, ValueError):
        return None


def _recovery_action(blocker: str | None) -> str | None:
    if blocker is None:
        return None
    if "teacher" in blocker:
        return "provide_frozen_teacher_checkpoint_and_rebuild_readiness"
    if "domain" in blocker:
        return "provide_distinct_source_target_domain_assets"
    if "manifest" in blocker or "hard_negative" in blocker:
        return "recover_train_hard_negative_evidence"
    if "baseline" in blocker or "control" in blocker:
        return "generate_matched_baseline_control"
    if "asha" in blocker:
        return "register_eligible_paper_cohort"
    return "repair_paper_readiness_evidence"


__all__ = [
    "PaperTrainingReadinessRecord",
    "PaperTrainingReadinessReport",
    "build_paper_training_readiness",
]
