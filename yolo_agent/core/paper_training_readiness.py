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
import yaml

from yolo_agent.agents.asha_scheduler import ASHAStudy
from yolo_agent.certification.paper_readiness import PaperReadinessReport
from yolo_agent.core.matched_baseline import assess_matched_control_plan
from yolo_agent.core.yaml_io import YAMLModelMixin
from yolo_agent.research.paper_asset_schemas import PaperAssetRegistry
from yolo_agent.research.paper_execution_requirement_schemas import (
    PaperExecutionRequirementsMatrix,
)
from yolo_agent.research.paper_execution_schemas import PaperExecutionInventory
from yolo_agent.research.paper_asset_dependencies import (
    requires_domain_assets,
    requires_hard_negative_replay,
    requires_teacher_checkpoint,
)


TRAINING_READINESS_SCHEMA_VERSION = "paper_training_readiness.v1"
_RUNNABLE_TRIAL_STATUSES = {
    "waiting",
    "running",
    "promotion_pending",
    "full_pending_confirmation",
    "confirmation_pending",
}
_INFERENCE_PREFIX = "inference."
_FINAL_COHORT_STAT_FIELDS = (
    "total_papers",
    "trainable_fingerprints",
    "evidence_bootstrap_fingerprints",
    "teacher_ready_fingerprints",
    "matched_controls_planned",
    "asha_trials_registered",
    "external_domain_blocked",
    "implementation_blocked",
    "inference_only",
)


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
    disposition: str = "blocked_runtime"
    implementation_complete: bool = False
    exact_reproduction_possible: bool = False
    actual_trained: bool = False
    training_allowed: bool = False
    inference_only: bool = False
    cpu_checks_passed: bool = False
    runtime_checks_passed: bool = False
    matched_control_ready: bool = False
    matched_control_plan_ready: bool = False
    matched_control_result_ready: bool = False
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
    inventory_count: int = 0
    # These are the final gate's public cohort counters.  They are distinct
    # from per-paper counts because several papers may share one execution.
    total_papers: int = Field(default=0, ge=0)
    trainable_fingerprints: int = Field(default=0, ge=0)
    evidence_bootstrap_fingerprints: int = Field(default=0, ge=0)
    teacher_ready_fingerprints: int = Field(default=0, ge=0)
    matched_controls_planned: int = Field(default=0, ge=0)
    asha_trials_registered: int = Field(default=0, ge=0)
    external_domain_blocked: int = Field(default=0, ge=0)
    implementation_blocked: int = Field(default=0, ge=0)
    inference_only: int = Field(default=0, ge=0)
    implementation_complete_count: int = 0
    cpu_ready_count: int = 0
    runtime_ready_count: int = 0
    matched_control_ready_count: int = 0
    matched_control_plan_ready_count: int = 0
    matched_control_result_ready_count: int = 0
    asha_eligible_count: int = 0
    asha_registered_count: int = 0
    runnable_assignment_count: int = 0
    pre_registered_count: int = 0
    blocked_count: int = 0
    evidence_recovery_count: int = 0
    deferred_count: int = 0
    inference_only_count: int = 0
    actual_trained_count: int = 0
    exact_reproduction_count: int = 0
    training_cohort_fingerprints: list[str] = Field(default_factory=list)
    registration_failures_by_paper_id: dict[str, list[str]] = Field(default_factory=dict)
    records: list[PaperTrainingReadinessRecord] = Field(default_factory=list)
    blockers: list[str] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    report_hash: str = ""

    @model_validator(mode="after")
    def validate_report(self) -> "PaperTrainingReadinessReport":
        fields_present = set(self.model_fields_set)
        legacy_stats = not any(
            field in fields_present for field in _FINAL_COHORT_STAT_FIELDS
        )
        legacy_matched_control = "matched_control_plan_ready_count" not in fields_present
        if legacy_stats:
            self._populate_legacy_cohort_stats()
        ids = [item.paper_id for item in self.records]
        if len(ids) != len(set(ids)):
            raise ValueError("paper training readiness contains duplicate paper IDs")
        if ids != sorted(ids):
            raise ValueError("paper training readiness records must be sorted")
        if self.paper_count != len(ids):
            raise ValueError("paper_count must equal record count")
        if self.inventory_count != self.paper_count:
            raise ValueError("inventory_count must equal paper_count")
        if self.total_papers != self.paper_count:
            raise ValueError("total_papers must equal paper_count")
        if self.training_started:
            raise ValueError("paper training readiness cannot report training_started")
        if self.gpu_probe != "not_run":
            raise ValueError("paper training readiness must not probe GPU")
        eligible_fingerprints = {
            item.execution_fingerprint
            for item in self.records
            if item.asha_eligibility
        }
        if self.asha_eligible_count != len(eligible_fingerprints):
            raise ValueError("asha_eligible_count does not match records")
        if self.trainable_fingerprints != len(eligible_fingerprints):
            raise ValueError("trainable_fingerprints does not match records")
        if self.training_allowed != bool(self.trainable_fingerprints):
            raise ValueError(
                "training_allowed must reflect trainable fingerprints"
            )
        if self.asha_trials_registered != self.asha_registered_count:
            raise ValueError(
                "asha_trials_registered must match asha_registered_count"
            )
        if self.training_cohort_fingerprints != sorted(
            set(self.training_cohort_fingerprints)
        ):
            raise ValueError("training cohort fingerprints must be sorted and unique")
        if set(self.training_cohort_fingerprints) != eligible_fingerprints:
            raise ValueError(
                "training cohort must contain exactly the eligible fingerprints"
            )
        if self.implementation_complete_count != sum(
            item.implementation_complete for item in self.records
        ):
            raise ValueError("implementation_complete_count does not match records")
        if self.cpu_ready_count != sum(
            item.cpu_checks_passed for item in self.records
        ):
            raise ValueError("cpu_ready_count does not match records")
        if self.runtime_ready_count != sum(
            item.runtime_checks_passed
            and item.readiness_state in {"runtime_ready", "asha_eligible"}
            for item in self.records
        ):
            raise ValueError("runtime_ready_count does not match records")
        if self.matched_control_ready_count != sum(
            item.matched_control_ready for item in self.records
        ):
            raise ValueError("matched_control_ready_count does not match records")
        if self.matched_control_plan_ready_count != sum(
            item.matched_control_plan_ready for item in self.records
        ):
            raise ValueError("matched_control_plan_ready_count does not match records")
        if self.matched_control_result_ready_count != sum(
            item.matched_control_result_ready for item in self.records
        ):
            raise ValueError("matched_control_result_ready_count does not match records")
        pre_registered_trials = {
            item.asha_trial_id
            for item in self.records
            if item.asha_trial_id is not None
        }
        if self.pre_registered_count != len(pre_registered_trials):
            raise ValueError("pre_registered_count does not match records")
        if self.evidence_recovery_count != sum(
            item.disposition == "evidence_recovery" for item in self.records
        ):
            raise ValueError("evidence_recovery_count does not match records")
        if self.inference_only_count != sum(item.inference_only for item in self.records):
            raise ValueError("inference_only_count does not match records")
        if self.actual_trained_count != sum(item.actual_trained for item in self.records):
            raise ValueError("actual_trained_count does not match records")
        if self.exact_reproduction_count != sum(
            item.exact_reproduction_possible for item in self.records
        ):
            raise ValueError("exact_reproduction_count does not match records")
        if self.blocked_count != sum(
            item.disposition in {"blocked_runtime", "incompatible"}
            for item in self.records
        ):
            raise ValueError("blocked_count does not match records")
        if self.deferred_count != sum(
            item.disposition == "deferred_budget" for item in self.records
        ):
            raise ValueError("deferred_count does not match records")
        if self.report_hash and self.report_hash != self.calculate_hash():
            if not legacy_stats or self.report_hash != self.calculate_hash(
                legacy_matched_control=legacy_matched_control,
                legacy_stats=True,
            ):
                raise ValueError("paper training readiness report hash mismatch")
        return self

    def _populate_legacy_cohort_stats(self) -> None:
        """Project pre-cohort reports into the new public counter names."""
        self.total_papers = self.paper_count
        self.trainable_fingerprints = self.asha_eligible_count
        self.evidence_bootstrap_fingerprints = len(
            {
                item.execution_fingerprint
                for item in self.records
                if item.disposition == "evidence_recovery"
            }
        )
        self.teacher_ready_fingerprints = 0
        self.matched_controls_planned = self.matched_control_plan_ready_count
        self.asha_trials_registered = self.asha_registered_count
        self.external_domain_blocked = len(
            {
                item.execution_fingerprint
                for item in self.records
                if item.blocker and "domain" in item.blocker.lower()
            }
        )
        self.implementation_blocked = len(
            {
                item.execution_fingerprint
                for item in self.records
                if item.blocker
                and any(
                    marker in item.blocker.lower()
                    for marker in ("adapter", "implementation")
                )
            }
        )
        self.inference_only = self.inference_only_count

    def calculate_hash(
        self,
        *,
        legacy_matched_control: bool = False,
        legacy_stats: bool = False,
    ) -> str:
        payload = self.model_dump(mode="json", exclude={"report_hash", "generated_at"})
        if legacy_stats:
            for field in _FINAL_COHORT_STAT_FIELDS:
                payload.pop(field, None)
        if legacy_matched_control:
            payload.pop("matched_control_plan_ready_count", None)
            payload.pop("matched_control_result_ready_count", None)
            for record in payload["records"]:
                record.pop("matched_control_plan_ready", None)
                record.pop("matched_control_result_ready", None)
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
    trainable_fingerprints: set[str] = set()
    evidence_bootstrap_fingerprints: set[str] = set()
    teacher_ready_fingerprints: set[str] = set()
    matched_control_ids: set[str] = set()
    asha_trial_ids: set[str] = set()
    external_domain_fingerprints: set[str] = set()
    implementation_blocked_fingerprints: set[str] = set()
    for paper_id in sorted(inventory_ids):
        item = inventory_by_id[paper_id]
        requirement = requirement_by_id[paper_id]
        asset = asset_by_id[paper_id]
        preflight = readiness_by_id[paper_id]
        inference_only = _is_inference_only(item, requirement, preflight)
        mock_evidence = _has_mock_evidence(preflight, asset)
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
                and _trial_has_candidate_node(trial)
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
            mock_evidence=mock_evidence,
        )
        allowed = active_trial is not None and blocker is None and not inference_only
        disposition = _final_disposition(
            preflight=preflight,
            requirement=requirement,
            inference_only=inference_only,
            blocker=blocker,
        )
        implementation_complete = _implementation_complete(
            item=item,
            requirement=requirement,
        )
        fingerprint = item.execution_fingerprint
        mechanisms = set(item.paper_specific_mechanism_ids)
        mechanisms.update(requirement.paper_specific_mechanism_ids)
        if active_trial is not None:
            asha_trial_ids.add(active_trial.trial_id)
        if allowed:
            trainable_fingerprints.add(fingerprint)
        if any(
            candidate_trial.matched_control_plan_ready
            and candidate_trial.matched_control_plan is not None
            for candidate_trial in candidate_trials
        ):
            for candidate_trial in candidate_trials:
                if (
                    candidate_trial.matched_control_plan_ready
                    and candidate_trial.matched_control_plan is not None
                ):
                    matched_control_ids.add(
                        candidate_trial.matched_control_plan.baseline_node_id
                    )
        if (
            requires_teacher_checkpoint(mechanisms)
            and _teacher_asset_ready(asset)
            and not mock_evidence
        ):
            teacher_ready_fingerprints.add(fingerprint)
        if _is_evidence_bootstrap_candidate(
            requirement=requirement,
            preflight=preflight,
            blocker=blocker,
        ):
            evidence_bootstrap_fingerprints.add(fingerprint)
        if _is_external_domain_blocker(
            mechanisms=mechanisms,
            asset=asset,
            blocker=blocker,
        ):
            external_domain_fingerprints.add(fingerprint)
        if _is_implementation_blocker(
            requirement=requirement,
            implementation_complete=implementation_complete,
            inference_only=inference_only,
            blocker=blocker,
        ):
            implementation_blocked_fingerprints.add(fingerprint)
        actual_trained = _trial_has_production_training(trial)
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
                asha_eligibility=allowed,
                training_allowed=allowed,
                disposition=disposition,
                implementation_complete=implementation_complete,
                exact_reproduction_possible=item.exact_reproduction_possible,
                actual_trained=actual_trained,
                inference_only=inference_only,
                cpu_checks_passed=(
                    preflight.cpu_checks_passed is True and not mock_evidence
                ),
                runtime_checks_passed=(
                    preflight.runtime_checks_passed is True and not mock_evidence
                ),
                matched_control_ready=(
                    bool(active_trial and active_trial.matched_control_plan_ready)
                    and not mock_evidence
                ),
                matched_control_plan_ready=(
                    bool(active_trial and active_trial.matched_control_plan_ready)
                    and not mock_evidence
                ),
                matched_control_result_ready=(
                    bool(active_trial and active_trial.matched_control_result_ready)
                    and not mock_evidence
                ),
                asset_available=_asset_available_for_scheduling(asset),
                asha_trial_id=trial.trial_id if trial else None,
                asha_assignment_ids=[item.assignment_id for item in assignments],
                blocker=blocker,
                recovery_action=recovery,
            )
        )

    records.sort(key=lambda item: item.paper_id)
    eligible_count = len(
        {
            item.execution_fingerprint
            for item in records
            if item.asha_eligibility
        }
    )
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
    paper_trial_ids = {
        trial.trial_id for trial in study.trials if trial.paper_ids
    }
    runnable_assignments = sum(
        assignment.status in {"issued", "running"}
        and assignment.trial_id in paper_trial_ids
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
        inventory_count=len(records),
        implementation_complete_count=sum(
            item.implementation_complete for item in records
        ),
        cpu_ready_count=sum(item.cpu_checks_passed for item in records),
        runtime_ready_count=sum(
            item.runtime_checks_passed
            and item.readiness_state in {"runtime_ready", "asha_eligible"}
            for item in records
        ),
        matched_control_ready_count=sum(
            item.matched_control_ready for item in records
        ),
        matched_control_plan_ready_count=sum(
            item.matched_control_plan_ready for item in records
        ),
        matched_control_result_ready_count=sum(
            item.matched_control_result_ready for item in records
        ),
        asha_eligible_count=eligible_count,
        asha_registered_count=registered,
        runnable_assignment_count=runnable_assignments,
        pre_registered_count=len(
            {
                item.asha_trial_id
                for item in records
                if item.asha_trial_id is not None
            }
        ),
        blocked_count=sum(
            item.disposition in {"blocked_runtime", "incompatible"}
            for item in records
        ),
        evidence_recovery_count=sum(
            item.disposition == "evidence_recovery" for item in records
        ),
        deferred_count=sum(
            item.disposition == "deferred_budget" for item in records
        ),
        inference_only_count=sum(item.inference_only for item in records),
        actual_trained_count=sum(item.actual_trained for item in records),
        exact_reproduction_count=sum(
            item.exact_reproduction_possible for item in records
        ),
        training_cohort_fingerprints=sorted(trainable_fingerprints),
        registration_failures_by_paper_id=failure_by_paper,
        records=records,
        blockers=blockers,
        total_papers=len(records),
        trainable_fingerprints=len(trainable_fingerprints),
        evidence_bootstrap_fingerprints=len(evidence_bootstrap_fingerprints),
        teacher_ready_fingerprints=len(teacher_ready_fingerprints),
        matched_controls_planned=len(matched_control_ids),
        asha_trials_registered=len(asha_trial_ids),
        external_domain_blocked=len(external_domain_fingerprints),
        implementation_blocked=len(implementation_blocked_fingerprints),
        inference_only=sum(item.inference_only for item in records),
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


def _trial_has_candidate_node(trial: Any) -> bool:
    """Require an actual candidate source node before authorizing training."""
    source_node = getattr(trial, "source_node", None)
    if source_node is None:
        return False
    candidate = getattr(source_node, "candidate_config", None)
    if candidate is None or not str(getattr(candidate, "candidate_id", "")).strip():
        return False
    metadata = _node_metadata(source_node)
    return not bool(metadata.get("matched_baseline_control"))


def _is_inference_only(item: Any, requirement: Any, preflight: Any) -> bool:
    return bool(
        preflight.inference_only
        or requirement.execution_route == "inference"
        or any(str(component).startswith(_INFERENCE_PREFIX) for component in item.canonical_component_ids)
    )


def _paper_blocker(
    *, item: Any, requirement: Any, asset: Any, preflight: Any,
    inference_only: bool, active_trial: Any, mock_evidence: bool,
) -> str | None:
    if inference_only:
        return "inference_only_not_training_candidate"
    if mock_evidence:
        return "mock_evidence_not_production_authorization"
    if not requirement.training_candidate_allowed or requirement.execution_route != "training":
        return requirement.exact_blocker or "training_route_not_allowed"
    asset_blocker = _required_asset_blocker(
        item=item,
        requirement=requirement,
        asset=asset,
    )
    if asset_blocker:
        return asset_blocker
    if not preflight.asha_eligibility or preflight.readiness_state != "asha_eligible":
        return preflight.exact_blocker or "readiness_report_not_asha_eligible"
    if not preflight.cpu_checks_passed:
        return "cpu_readiness_incomplete"
    if not preflight.runtime_checks_passed:
        return "runtime_readiness_incomplete"
    if not preflight.matched_control_plan_readiness.passed:
        return "matched_control_plan_not_ready"
    if not _asset_available_for_scheduling(asset):
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


def _has_mock_evidence(preflight: Any, asset: Any) -> bool:
    """Keep test backends and fixture artifacts outside production counts."""
    for source in (preflight, asset):
        fields = source.model_dump(mode="json")
        if _contains_mock_value(fields):
            return True
    return False


def _teacher_asset_ready(asset: Any) -> bool:
    """Return true only when the registry has a real teacher path and hash."""
    return bool(
        getattr(asset, "teacher_checkpoint", None)
        and getattr(asset, "teacher_sha256", None)
    )


def _is_evidence_bootstrap_candidate(
    *, requirement: Any, preflight: Any, blocker: str | None
) -> bool:
    text = str(blocker or "").lower()
    return bool(
        requirement.current_disposition == "evidence_recovery"
        or preflight.final_disposition == "evidence_recovery"
        or "evidence" in text
        or "hard_negative" in text
    )


def _is_external_domain_blocker(
    *, mechanisms: set[str], asset: Any, blocker: str | None
) -> bool:
    if not requires_domain_assets(mechanisms):
        return False
    text = str(blocker or "").lower()
    return bool(
        any(
            marker in text
            for marker in ("domain", "source_target", "target_domain", "source_domain")
        )
        or not getattr(asset, "source_dataset_manifest", None)
        or not getattr(asset, "target_dataset_manifest", None)
        or getattr(asset, "source_dataset_manifest", None)
        == getattr(asset, "target_dataset_manifest", None)
    )


def _is_implementation_blocker(
    *,
    requirement: Any,
    implementation_complete: bool,
    inference_only: bool,
    blocker: str | None,
) -> bool:
    if inference_only:
        return False
    text = str(blocker or "").lower()
    return bool(
        not implementation_complete
        or requirement.current_disposition == "implementation_request"
        or any(
            marker in text
            for marker in (
                "adapter",
                "implementation",
                "paper_specific",
                "route_incomplete",
            )
        )
    )


def _contains_mock_value(value: Any, *, field_name: str = "") -> bool:
    if isinstance(value, dict):
        return any(
            _contains_mock_value(item, field_name=str(key))
            for key, item in value.items()
        )
    if isinstance(value, list):
        return any(_contains_mock_value(item, field_name=field_name) for item in value)
    if not isinstance(value, str):
        return False
    normalized = value.strip().lower()
    if normalized in {"mock", "fixture", "fixtures", "pytest_fixture"}:
        return True
    if any(
        marker in normalized
        for marker in (
            "mock_backend",
            "offline_mock",
            "mock_evidence",
            "fixture_evidence",
            "pytest_fixture",
        )
    ):
        return True
    if field_name.endswith(("_path", "_checkpoint", "_manifest", "_artifact", "_config")):
        try:
            return any(
                part.lower() in {"mock", "fixture", "fixtures"}
                for part in Path(value).parts
            )
        except (OSError, ValueError):
            return False
    return False


def _required_asset_blocker(*, item: Any, requirement: Any, asset: Any) -> str | None:
    """Apply asset-type gates independently of a producer's readiness label."""
    mechanisms = set(item.paper_specific_mechanism_ids)
    mechanisms.update(requirement.paper_specific_mechanism_ids)
    if requires_teacher_checkpoint(mechanisms):
        if not asset.teacher_checkpoint or not asset.teacher_sha256:
            return "teacher_checkpoint_missing"

    if requires_domain_assets(mechanisms):
        if not asset.source_dataset_manifest or not asset.target_dataset_manifest:
            return "domain_source_target_missing"
        if asset.source_dataset_manifest == asset.target_dataset_manifest:
            return "domain_source_target_must_differ"

    if requires_hard_negative_replay(mechanisms):
        if not asset.hard_negative_manifest:
            return "hard_negative_train_manifest_missing"
        manifest_blocker = _hard_negative_manifest_blocker(
            Path(asset.hard_negative_manifest)
        )
        if manifest_blocker:
            return manifest_blocker

    return None


def _asset_available_for_scheduling(asset: Any) -> bool:
    """Baseline result is produced after scheduling, so it is not an asset gate."""
    if asset.availability == "available":
        return True
    blockers = {
        item.strip()
        for item in str(asset.exact_blocker or "").split(";")
        if item.strip()
    }
    baseline_only = {
        "matched_baseline_artifact_missing",
        "matched baseline artifact missing",
    }
    ignored_summary = {"runtime readiness evidence is incomplete"}
    return bool(blockers & baseline_only) and blockers <= baseline_only | ignored_summary


def _hard_negative_manifest_blocker(path: Path) -> str | None:
    """Reject replay files that are present but not explicitly train-only."""
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, yaml.YAMLError) as exc:
        return f"hard_negative_manifest_invalid:{type(exc).__name__}"
    if not isinstance(payload, dict):
        return "hard_negative_manifest_invalid:root_not_mapping"
    if (payload.get("source_split") or payload.get("split")) != "train":
        return "hard_negative_manifest_not_train_split"
    entries = payload.get("records", payload.get("samples", []))
    if isinstance(entries, list) and any(
        isinstance(entry, dict) and entry.get("split") not in {None, "train"}
        for entry in entries
    ):
        return "hard_negative_manifest_contains_non_train_sample"
    return None


def _final_disposition(
    *, preflight: Any, requirement: Any, inference_only: bool, blocker: str | None
) -> str:
    """Convert intermediate evidence into one final, enumerated disposition."""
    if inference_only:
        return "incompatible"
    if preflight.final_disposition == "incompatible" or not requirement.compatible_with_yolo26:
        return "incompatible"
    if preflight.final_disposition == "evidence_recovery":
        return "evidence_recovery"
    if requirement.current_disposition == "implementation_request":
        return "implementation_request"
    if requirement.current_disposition == "deferred_budget":
        return "deferred_budget"
    if blocker:
        return "blocked_runtime"
    return "runtime_ready"


def _implementation_complete(*, item: Any, requirement: Any) -> bool:
    """Count implementation metadata, never CPU fixtures or training results."""
    mechanisms = set(item.paper_specific_mechanism_ids)
    unresolved = any(
        mechanism.startswith("paper.unresolved") for mechanism in mechanisms
    )
    return bool(
        mechanisms
        and not unresolved
        and item.recipe_ids
        and requirement.required_adapter
        and requirement.required_changed_variables
        and requirement.required_runtime_payload
        and requirement.execution_route in {"training", "inference"}
    )


def _trial_has_production_training(trial: Any) -> bool:
    """Count only terminal evidence from a non-fixture execution record."""
    if trial is None or trial.status not in {"eliminated", "confirmed", "failed"}:
        return False
    if not trial.observations or _is_mock_trial(trial):
        return False
    return True


def _is_mock_trial(trial: Any) -> bool:
    values = [str(getattr(trial, "source_run_id", ""))]
    values.extend(str(value) for value in _node_metadata(trial.source_node).values())
    text = " ".join(values).lower()
    return any(token in text for token in ("mock", "fixture", "pytest"))


def _trial_identity_blocker(
    *, item: Any, requirement: Any, preflight: Any, asset: Any, trial: Any
) -> str | None:
    """Recheck persisted trial identity instead of trusting an old ASHA file."""
    control = trial.baseline_control_node
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
    assessment = assess_matched_control_plan(
        trial.source_node,
        control,
        required_protocol_hash=expected_protocol or None,
    )
    if not assessment.matched_control_plan_ready or assessment.plan is None:
        return "matched_control_plan_invalid:" + ",".join(assessment.blockers)
    if trial.matched_control_plan is None or not trial.matched_control_plan_ready:
        return "asha_trial_matched_control_plan_not_persisted"
    if trial.matched_control_plan.plan_hash != assessment.plan.plan_hash:
        return "asha_trial_matched_control_plan_hash_mismatch"
    if preflight.dataset_manifest_hash not in {
        "missing",
        assessment.plan.dataset_manifest_hash,
    }:
        return "asha_trial_dataset_manifest_hash_mismatch"
    return None


def _node_metadata(node: Any) -> dict[str, object]:
    command = getattr(node, "command_spec", None)
    return dict(getattr(command, "metadata", {}) or {})


def _node_value(node: Any, *keys: str) -> str:
    metadata = _node_metadata(node)
    value = _first_value(metadata, *keys)
    if value:
        return value
    command = getattr(node, "command_spec", None)
    for argument in getattr(command, "args", []):
        text = str(argument)
        for key in keys:
            prefix = f"{key}="
            if text.startswith(prefix):
                return text[len(prefix) :]
    return ""


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
    if "hard_negative" in blocker:
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
