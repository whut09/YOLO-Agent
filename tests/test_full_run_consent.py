"""Offline tests for scoped full-run authorization and baseline sequencing."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import json

from yolo_agent.adapters.ultralytics.baseline_acceptance import BaselineAcceptanceResult
from yolo_agent.agents.auto_optimization_loop import _trusted_full_run_authorization
from yolo_agent.agents.optimize_runner import _baseline_nodes
from yolo_agent.core.full_run_consent import FullRunConsentDriver
from yolo_agent.core.optimization_objective import OptimizationObjective, OptimizationObjectiveStatus


def _objective(**updates: object) -> OptimizationObjective:
    values: dict[str, object] = {
        "baseline_run_id": "coco-yolo26n",
        "baseline_candidate_id": "yolo26n_coco_baseline_full",
        "baseline_protocol_hash": "baseline-protocol-v1",
        "max_gpu_hours": 24.0,
    }
    values.update(updates)
    return OptimizationObjective.model_validate(values)


def test_consent_grant_and_valid_reuse(tmp_path: Path) -> None:
    driver = FullRunConsentDriver(tmp_path / "runs" / "coco-yolo26n")
    objective = _objective()

    granted = driver.grant(
        run_id="coco-yolo26n",
        objective=objective,
        dataset_manifest_sha256="dataset-v1",
    )
    decision = driver.validate(
        run_id="coco-yolo26n",
        objective=objective,
        dataset_manifest_sha256="dataset-v1",
        objective_status=OptimizationObjectiveStatus(
            objective_hash=objective.objective_hash,
            primary_metric=objective.primary_metric,
            gpu_hours_used=6.5,
            gpu_budget_remaining=17.5,
        ),
    )

    assert granted.state == "active"
    assert decision.allowed is True
    assert decision.gpu_hours_used == 6.5
    assert decision.gpu_hours_remaining == 17.5


def test_consent_invalidates_when_scope_changes(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "coco-yolo26n"
    driver = FullRunConsentDriver(run_dir)
    objective = _objective()
    driver.grant(
        run_id="coco-yolo26n",
        objective=objective,
        dataset_manifest_sha256="dataset-v1",
    )

    changed_objective = driver.validate(
        run_id="coco-yolo26n",
        objective=_objective(target_absolute_delta=0.03),
        dataset_manifest_sha256="dataset-v1",
    )
    assert changed_objective.allowed is False
    assert changed_objective.reason == "full_run_consent_objective_changed"

    driver.grant(
        run_id="coco-yolo26n",
        objective=objective,
        dataset_manifest_sha256="dataset-v1",
    )
    changed_manifest = driver.validate(
        run_id="coco-yolo26n",
        objective=objective,
        dataset_manifest_sha256="dataset-v2",
    )
    assert changed_manifest.reason == "full_run_consent_dataset_manifest_changed"

    driver.grant(
        run_id="coco-yolo26n",
        objective=objective,
        dataset_manifest_sha256="dataset-v1",
    )
    changed_protocol = driver.validate(
        run_id="coco-yolo26n",
        objective=_objective(baseline_protocol_hash="baseline-protocol-v2"),
        dataset_manifest_sha256="dataset-v1",
    )
    assert changed_protocol.reason == "full_run_consent_objective_changed"


def test_consent_rejects_budget_change_and_exhaustion(tmp_path: Path) -> None:
    driver = FullRunConsentDriver(tmp_path / "runs" / "coco-yolo26n")
    objective = _objective()
    driver.grant(
        run_id="coco-yolo26n",
        objective=objective,
        dataset_manifest_sha256="dataset-v1",
    )

    budget_changed = driver.validate(
        run_id="coco-yolo26n",
        objective=_objective(max_gpu_hours=30.0),
        dataset_manifest_sha256="dataset-v1",
    )
    assert budget_changed.allowed is False
    assert budget_changed.reason == "full_run_consent_objective_changed"

    driver.grant(
        run_id="coco-yolo26n",
        objective=objective,
        dataset_manifest_sha256="dataset-v1",
    )
    exhausted = driver.validate(
        run_id="coco-yolo26n",
        objective=objective,
        dataset_manifest_sha256="dataset-v1",
        objective_status=OptimizationObjectiveStatus(
            objective_hash=objective.objective_hash,
            primary_metric=objective.primary_metric,
            gpu_hours_used=24.0,
            gpu_budget_remaining=0.0,
        ),
    )
    assert exhausted.allowed is False
    assert exhausted.reason == "gpu_budget_exhausted"
    assert exhausted.consent is not None and exhausted.consent.state == "exhausted"


def test_baseline_full_sequence_uses_seed_one_then_two_and_three() -> None:
    full = _baseline_nodes("coco", "yolo26n.pt", "baseline_full", "coco2017")
    confirm = _baseline_nodes("coco", "yolo26n.pt", "baseline_confirm", "coco2017")

    assert [node.seed for node in full] == [1]
    assert [node.seed for node in confirm] == [2, 3]
    assert len({node.node_id for node in [*full, *confirm]}) == 3


def test_candidate_full_requires_consent_and_trusted_baseline(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "coco-yolo26n"
    objective = _objective()
    context = SimpleNamespace(
        run_id="coco-yolo26n",
        run_dir=run_dir,
        dataset_manifest_sha256="dataset-v1",
        artifact_path=lambda name: run_dir / "artifacts" / name,
    )
    allowed, reason = _trusted_full_run_authorization(context, objective, None)
    assert allowed is False
    assert reason == "full_run_consent_missing"

    FullRunConsentDriver(run_dir).grant(
        run_id="coco-yolo26n",
        objective=objective,
        dataset_manifest_sha256="dataset-v1",
    )
    allowed, reason = _trusted_full_run_authorization(context, objective, None)
    assert allowed is False
    assert reason == "baseline_acceptance_missing"

    acceptance = BaselineAcceptanceResult(
        baseline_trusted=True,
        accepted_seed_count=3,
        actual_dataset_manifest_sha256="dataset-v1",
    )
    acceptance_path = context.artifact_path("baseline_acceptance.json")
    acceptance_path.parent.mkdir(parents=True, exist_ok=True)
    acceptance_path.write_text(json.dumps(acceptance.model_dump(mode="json")), encoding="utf-8")

    allowed, reason = _trusted_full_run_authorization(context, objective, None)
    assert allowed is True
    assert reason == "trusted_full_run_authorized"
