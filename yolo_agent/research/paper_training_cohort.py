"""Build the current-data paper training cohort without changing evidence."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TYPE_CHECKING, Any

from yolo_agent.research.paper_training_cohort_schemas import (
    PaperTrainingCohort,
    PaperTrainingCohortRecord,
)

if TYPE_CHECKING:
    from yolo_agent.certification.paper_readiness import PaperReadinessReport
    from yolo_agent.research.paper_asset_schemas import PaperAssetRegistry
    from yolo_agent.research.paper_execution_requirement_schemas import (
        PaperExecutionRequirementsMatrix,
    )
    from yolo_agent.research.paper_execution_schemas import PaperExecutionInventory


class PaperTrainingCohortBuilder:
    """Classify every paper while authorizing only the current executable set.

    The builder intentionally does not require a completed matched-baseline
    result.  A valid control *plan* is enough to schedule the first paired
    run; the paired result is produced only after baseline and candidate
    execution complete.
    """

    def build(
        self,
        *,
        inventory_path: Path | str,
        requirements_path: Path | str,
        assets_path: Path | str,
        readiness_path: Path | str,
        asha_path: Path | str,
        output_path: Path | str = Path(
            "runs/paper-readiness/paper_training_cohort.yaml"
        ),
        expected_paper_count: int = 83,
    ) -> PaperTrainingCohort:
        inventory_file = _existing_file(inventory_path, "inventory")
        requirements_file = _existing_file(requirements_path, "requirements")
        assets_file = _existing_file(assets_path, "assets")
        readiness_file = _existing_file(readiness_path, "readiness")
        asha_file = _existing_file(asha_path, "asha")

        from yolo_agent.agents.asha_scheduler import ASHAStudy
        from yolo_agent.certification.paper_readiness import PaperReadinessReport
        from yolo_agent.research.paper_asset_schemas import PaperAssetRegistry
        from yolo_agent.research.paper_execution_requirement_schemas import (
            PaperExecutionRequirementsMatrix,
        )
        from yolo_agent.research.paper_execution_schemas import PaperExecutionInventory

        inventory = PaperExecutionInventory.from_yaml(inventory_file)
        requirements = PaperExecutionRequirementsMatrix.from_yaml(requirements_file)
        assets = PaperAssetRegistry.from_yaml(assets_file)
        readiness = PaperReadinessReport.from_yaml(readiness_file)
        study = ASHAStudy.from_yaml(asha_file)
        _validate_inputs(
            inventory=inventory,
            requirements=requirements,
            assets=assets,
            readiness=readiness,
            requirements_file=requirements_file,
            expected_paper_count=expected_paper_count,
        )

        requirements_by_id = {item.paper_id: item for item in requirements.requirements}
        assets_by_id = {item.paper_id: item for item in assets.records}
        readiness_by_id = {item.paper_id: item for item in readiness.records}
        trials_by_fingerprint: dict[str, list[Any]] = {}
        for trial in study.trials:
            trials_by_fingerprint.setdefault(trial.execution_fingerprint, []).append(trial)

        paper_ids_by_fingerprint: dict[str, list[str]] = {}
        for item in inventory.records:
            paper_ids_by_fingerprint.setdefault(item.execution_fingerprint, []).append(
                item.paper_id
            )

        records: list[PaperTrainingCohortRecord] = []
        for item in inventory.records:
            requirement = requirements_by_id[item.paper_id]
            asset = assets_by_id[item.paper_id]
            preflight = readiness_by_id[item.paper_id]
            trials = trials_by_fingerprint.get(item.execution_fingerprint, [])
            trial = _preferred_trial(trials)
            records.append(
                _classify(
                    item=item,
                    requirement=requirement,
                    asset=asset,
                    preflight=preflight,
                    paper_ids=paper_ids_by_fingerprint[item.execution_fingerprint],
                    trial=trial,
                )
            )

        records.sort(key=lambda item: item.paper_id)
        eligible_fingerprints = sorted(
            {
                item.execution_fingerprint
                for item in records
                if item.asha_eligible
            }
        )
        cohort = PaperTrainingCohort(
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
            category_counts={
                category: sum(item.category == category for item in records)
                for category in (
                    "coco_supervised_ready",
                    "teacher_bootstrap_ready",
                    "evidence_bootstrap_ready",
                    "inference_only",
                    "requires_external_domain",
                    "implementation_blocked",
                )
            },
            executable_fingerprint_count=len(eligible_fingerprints),
            training_cohort_fingerprints=eligible_fingerprints,
            training_allowed=bool(eligible_fingerprints),
            records=records,
        ).with_hash()
        cohort.to_yaml(output_path, exclude_none=True, sort_keys=False)
        return cohort


def build_paper_training_cohort(**kwargs: object) -> PaperTrainingCohort:
    """Functional entry point for tools, tests, and future CLI integration."""

    return PaperTrainingCohortBuilder().build(**kwargs)  # type: ignore[arg-type]


def _classify(
    *,
    item: Any,
    requirement: Any,
    asset: Any,
    preflight: Any,
    paper_ids: list[str],
    trial: Any,
) -> PaperTrainingCohortRecord:
    mechanisms = set(item.canonical_component_ids)
    mechanisms.update(item.paper_specific_mechanism_ids)
    mechanism_text = " ".join(mechanisms)
    inference_only = bool(
        preflight.inference_only
        or requirement.execution_route == "inference"
        or any(str(value).startswith("inference.") for value in mechanisms)
    )
    implementation_complete = _implementation_complete(item, requirement)
    cpu_ready = preflight.cpu_checks_passed is True
    runtime_ready = preflight.runtime_checks_passed is True
    plan_check = getattr(
        preflight,
        "matched_control_plan_readiness",
        preflight.matched_control_readiness,
    )
    matched_plan_ready = bool(
        (plan_check and plan_check.passed)
        or getattr(trial, "matched_control_plan_ready", False)
    )
    matched_result_ready = bool(
        getattr(preflight, "matched_control_result_readiness", None)
        and preflight.matched_control_result_readiness.passed
    )
    mock_evidence = _contains_mock_value(preflight.model_dump(mode="json")) or _contains_mock_value(
        asset.model_dump(mode="json")
    )
    asset_blocker = _asset_blocker(item, requirement, asset)
    runtime_ready_for_cohort = runtime_ready or _baseline_result_only_failure(preflight)
    base_blocker = _base_blocker(
        requirement=requirement,
        preflight=preflight,
        implementation_complete=implementation_complete,
        cpu_ready=cpu_ready,
        runtime_ready=runtime_ready_for_cohort,
        matched_plan_ready=matched_plan_ready,
        mock_evidence=mock_evidence,
    )
    blocker = asset_blocker or base_blocker
    is_domain = bool(
        requirement.required_domain_assets
        or any(str(value).startswith("domain_adaptation.") for value in mechanisms)
    )
    is_distillation = bool(
        requirement.required_teacher_assets
        or any(str(value).startswith("distillation.") for value in mechanisms)
    )
    is_hard_negative = "hard_negative" in mechanism_text or bool(
        requirement.required_manifest_assets
    )
    teacher_ready = is_distillation and bool(
        asset.teacher_checkpoint and asset.teacher_sha256 and not asset_blocker
    )
    evidence_ready = is_hard_negative and bool(
        asset.hard_negative_manifest and not asset_blocker
    )

    if inference_only:
        category = "inference_only"
        candidate = eligible = False
        blocker = "inference_only_not_training_candidate"
    elif is_domain:
        category = "requires_external_domain"
        candidate = eligible = False
        blocker = asset_blocker or "domain_source_target_required"
    elif is_hard_negative and not evidence_ready:
        category = "evidence_bootstrap_ready"
        candidate = eligible = False
        blocker = asset_blocker or "hard_negative_train_manifest_missing"
    elif not implementation_complete:
        category = "implementation_blocked"
        candidate = eligible = False
        blocker = blocker or "paper_adapter_or_route_incomplete"
    elif is_distillation and not teacher_ready:
        category = "implementation_blocked"
        candidate = eligible = False
        blocker = blocker or "teacher_checkpoint_missing"
    elif blocker:
        category = "implementation_blocked"
        candidate = eligible = False
    else:
        category = "teacher_bootstrap_ready" if is_distillation else "coco_supervised_ready"
        candidate = eligible = True

    return PaperTrainingCohortRecord(
        paper_id=item.paper_id,
        paper_ids=paper_ids,
        profile_id=item.profile_id,
        method_profile_ids=[item.profile_id],
        mechanism_ids=sorted(mechanisms),
        recipe_ids=list(item.recipe_ids),
        execution_fingerprint=item.execution_fingerprint,
        category=category,
        training_candidate=candidate,
        asha_eligible=eligible,
        pre_registered=bool(trial or getattr(preflight, "pre_registered", False)),
        cpu_checks_passed=cpu_ready,
        runtime_checks_passed=runtime_ready_for_cohort,
        matched_control_plan_ready=matched_plan_ready,
        matched_control_result_ready=matched_result_ready,
        teacher_ready=teacher_ready,
        evidence_ready=evidence_ready,
        inference_only=inference_only,
        asha_trial_id=getattr(trial, "trial_id", None),
        blocker=None if eligible else blocker,
        recovery_action=None if eligible else _recovery_action(blocker),
    )


def _implementation_complete(item: Any, requirement: Any) -> bool:
    specific = list(item.paper_specific_mechanism_ids)
    return bool(
        specific
        and not any(str(value).startswith("paper.unresolved") for value in specific)
        and item.recipe_ids
        and requirement.required_adapter
        and requirement.required_changed_variables
        and requirement.required_runtime_payload
        and requirement.execution_route in {"training", "inference"}
    )


def _asset_blocker(item: Any, requirement: Any, asset: Any) -> str | None:
    mechanisms = set(item.paper_specific_mechanism_ids)
    mechanisms.update(requirement.paper_specific_mechanism_ids)
    if requirement.required_teacher_assets or any(
        str(value).startswith("distillation.") for value in mechanisms
    ):
        if not asset.teacher_checkpoint or not asset.teacher_sha256:
            return "teacher_checkpoint_missing"
    if requirement.required_domain_assets or any(
        str(value).startswith("domain_adaptation.") for value in mechanisms
    ):
        if not asset.source_dataset_manifest or not asset.target_dataset_manifest:
            return "domain_source_target_missing"
        if Path(asset.source_dataset_manifest).resolve() == Path(
            asset.target_dataset_manifest
        ).resolve():
            return "domain_source_target_must_differ"
    if requirement.required_manifest_assets or any(
        "hard_negative" in str(value) for value in mechanisms
    ):
        if not asset.hard_negative_manifest:
            return "hard_negative_train_manifest_missing"
    return None


def _base_blocker(
    *,
    requirement: Any,
    preflight: Any,
    implementation_complete: bool,
    cpu_ready: bool,
    runtime_ready: bool,
    matched_plan_ready: bool,
    mock_evidence: bool,
) -> str | None:
    if mock_evidence:
        return "mock_evidence_not_production_authorization"
    if not requirement.compatible_with_yolo26:
        return "incompatible_with_yolo26"
    if requirement.execution_route == "inference":
        return "inference_only_not_training_candidate"
    # Requirements are often generated before the first control plan and are
    # therefore labelled blocked_runtime/evidence_recovery.  Once the actual
    # adapter, payload, checks, and control plan are present, those stale
    # planning labels must not prevent this current-data cohort from forming.
    if requirement.execution_route not in {
        "training",
        "blocked_runtime",
        "evidence_recovery",
    }:
        return requirement.exact_blocker or "training_route_not_allowed"
    if requirement.exact_blocker and requirement.exact_blocker not in {
        "runtime readiness evidence is incomplete",
        "matched baseline artifact missing",
        "matched_baseline_artifact_missing",
        "teacher_checkpoint_missing",
        "teacher_checkpoint_sha256_missing",
        "hard_negative_train_manifest_missing",
    } and not requirement.training_candidate_allowed:
        return requirement.exact_blocker
    if not implementation_complete:
        return "paper_adapter_or_route_incomplete"
    if not cpu_ready:
        return _check_blocker(preflight, "cpu_readiness_incomplete")
    if not runtime_ready:
        return _check_blocker(preflight, "runtime_readiness_incomplete")
    if not matched_plan_ready:
        return _check_blocker(preflight, "matched_control_plan_not_ready")
    return None


def _baseline_result_only_failure(preflight: Any) -> bool:
    """Allow scheduling when the only stale failure is the post-run result.

    Older readiness reports folded ``matched_baseline_artifact_missing`` into
    dataset/runtime checks.  A control plan is intentionally created before
    that result exists, so this one post-run artifact is not a cohort blocker.
    Any other failed runtime check remains blocking.
    """
    runtime_checks = (
        preflight.dataset_evidence_result,
        preflight.domain_evidence_result,
        preflight.manifest_evidence_result,
        preflight.protocol_evidence_result,
        preflight.teacher_evidence_result,
        preflight.graph_evidence_result,
    )
    failures = [item for item in runtime_checks if not item.passed]
    if not failures:
        return False
    return all(
        item.blocker in {
            "matched_baseline_artifact_missing",
            "matched_baseline_result_missing",
            "matched baseline artifact missing",
        }
        for item in failures
    )


def _check_blocker(preflight: Any, fallback: str) -> str:
    blocker = getattr(preflight, "exact_blocker", None)
    if blocker and blocker != "matched_control_result_pending":
        return blocker
    return fallback


def _preferred_trial(trials: list[Any]) -> Any | None:
    for trial in trials:
        if trial.readiness_state == "asha_eligible":
            return trial
    return trials[0] if trials else None


def _recovery_action(blocker: str | None) -> str:
    text = blocker or ""
    if "teacher" in text:
        return "provide_frozen_teacher_checkpoint_and_rebuild_readiness"
    if "domain" in text:
        return "provide_distinct_source_target_domain_assets"
    if "hard_negative" in text or "manifest" in text:
        return "recover_train_hard_negative_evidence"
    if "control" in text or "baseline" in text:
        return "generate_matched_baseline_control_plan"
    if "cpu" in text or "adapter" in text or "route" in text:
        return "implement_and_materialize_paper_adapter"
    return "repair_paper_training_readiness"


def _validate_inputs(
    *,
    inventory: PaperExecutionInventory,
    requirements: PaperExecutionRequirementsMatrix,
    assets: PaperAssetRegistry,
    readiness: PaperReadinessReport,
    requirements_file: Path,
    expected_paper_count: int,
) -> None:
    if inventory.compatible_paper_count != expected_paper_count:
        raise ValueError(
            f"cohort requires {expected_paper_count} papers; "
            f"got {inventory.compatible_paper_count}"
        )
    expected = {item.paper_id for item in inventory.records}
    for label, actual in (
        ("requirements", {item.paper_id for item in requirements.requirements}),
        ("assets", {item.paper_id for item in assets.records}),
        ("readiness", {item.paper_id for item in readiness.records}),
    ):
        if actual != expected:
            raise ValueError(f"{label} paper coverage does not match inventory")
    if requirements.source_inventory_hash != inventory.inventory_hash:
        raise ValueError("requirements source inventory hash is stale")
    if assets.source_inventory_hash != inventory.inventory_hash:
        raise ValueError("asset registry source inventory hash is stale")
    if assets.source_requirements_hash != _file_hash(requirements_file):
        raise ValueError("asset registry source requirements hash is stale")
    if readiness.inventory_hash != inventory.inventory_hash:
        raise ValueError("readiness report inventory hash is stale")


def _existing_file(path: Path | str, label: str) -> Path:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise ValueError(f"{label} artifact does not exist: {resolved}")
    return resolved


def _contains_mock_value(value: Any, *, field_name: str = "") -> bool:
    """Keep test fixtures out of production cohort authorization."""
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
        for marker in ("mock_backend", "offline_mock", "mock_evidence", "fixture_evidence", "pytest_fixture")
    ):
        return True
    if field_name.endswith(("_path", "_checkpoint", "_manifest", "_artifact", "_config")):
        try:
            return any(part.lower() in {"mock", "fixture", "fixtures"} for part in Path(value).parts)
        except (OSError, ValueError):
            return False
    return False


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = ["PaperTrainingCohortBuilder", "build_paper_training_cohort"]
