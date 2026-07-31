from __future__ import annotations

import hashlib
import json
from pathlib import Path

from yolo_agent.agents.decision_bundle import DecisionContext
from yolo_agent.agents.paper_candidate_orchestrator import PaperCandidateEvidence
from yolo_agent.agents.paper_recipe_materialization.schemas import (
    PaperRecipeCandidateInput,
)
from yolo_agent.agents.paper_recipe_materialization_gate import (
    PaperRecipeMaterializationGate,
)
from yolo_agent.components.compatibility import CompatibilityResult
from yolo_agent.components.contracts import load_contracts
from yolo_agent.components.maturity import ComponentMaturityArtifact
from yolo_agent.components.maturity_registry import (
    ComponentMaturityRegistry,
    adapter_source_hash,
)
from yolo_agent.components.maturity_registry_schemas import ComponentEvidenceOverlay
from yolo_agent.recipes.paper_priors import RecipePrior, RecipePriorEvidence
from yolo_agent.research.awesome_snapshot_builder import AwesomeSnapshotBuilder
from yolo_agent.research.method_profiles import PaperMethodCoverageReport
from yolo_agent.research.snapshot import load_research_snapshot
from yolo_agent.resources import ResourcePaths
from tests.paired_result_helpers import verified_paired_result
from tests.paper_materialization_fixtures import (
    PROTOCOL_HASH,
    budget,
    error_fact,
    node,
    objective,
)


COMPONENT_ID = "sampling.small_object"
PAPER_ID = "paper-small-object-sampling"


def test_awesome_paper_recipe_reaches_pilot_10_through_certified_gates(
    tmp_path: Path,
) -> None:
    catalog = _write_catalog(tmp_path / "awesome")
    registry = _maturity_registry(tmp_path)
    research_root = tmp_path / "research"
    built = AwesomeSnapshotBuilder(
        research_root,
        maturity_registry=registry,
        maturity_protocol_hash="component-cert-v1",
        maturity_ultralytics_version="test-ultralytics",
    ).build(source=catalog, source_commit="awesome-commit-1")
    assert built.status == "completed", built.errors

    loaded = load_research_snapshot(research_root, built.snapshot_path)
    assert loaded is not None
    snapshot, snapshot_dir = loaded
    assert snapshot.paper_intelligence == "available"
    contract = next(
        item
        for item in load_contracts(snapshot_dir / "component_contracts.yaml")
        if item.component_id == COMPONENT_ID
    )
    assert contract.can_execute is True
    coverage = PaperMethodCoverageReport.from_yaml(
        snapshot_dir / "paper_method_coverage.yaml"
    )
    profile = next(item for item in coverage.profiles if item.paper_id == PAPER_ID)
    decision = next(item for item in coverage.decisions if item.paper_id == PAPER_ID)
    assert decision.decision == "reuse_existing_adapter"
    assert decision.reusable_adapter_ids == [COMPONENT_ID]

    run_id = "paper-auto"
    gate = PaperRecipeMaterializationGate(
        tmp_path / "runs" / run_id,
        base_run_id=run_id,
    )
    candidates = [
        _candidate(
            index,
            snapshot_hash=snapshot.snapshot_hash,
            profile=profile,
            decision=decision,
        )
        for index in range(1, 4)
    ]
    result = gate.materialize(
        run_id=run_id,
        decision_context=DecisionContext(
            run_id=run_id,
            research_snapshot_hash=snapshot.snapshot_hash,
            research_snapshot_verified=True,
            paper_intelligence="available",
        ),
        research_snapshot=snapshot,
        candidates=candidates,
        current_error_facts=[error_fact(run_id=run_id)],
        component_contracts={COMPONENT_ID: contract},
        objective=objective().model_copy(update={"baseline_run_id": run_id}),
        budget=budget(),
        round_index=1,
    )
    assert result.action == "queue_assignment"
    assert result.scalar_hpo_enabled is False
    assert sorted(result.registration["registered"]) == [
        "paper-sampling-1",
        "paper-sampling-2",
        "paper-sampling-3",
    ]
    assert result.execution_queue is not None
    assert result.execution_queue["metadata"]["source_authority"] == (
        "RoundExecutionPlan"
    )
    assert any(line.startswith("Adapter hash: ") for line in result.terminal_lines)
    assert "Maturity: sampling.small_object=smoke_passed" in result.terminal_lines

    orchestrator = gate.orchestrator
    for delta in (0.03, 0.02, 0.01):
        step = orchestrator.next_step()
        assert step.action == "queue_assignment"
        assert step.assignment is not None
        assert step.assignment.stage_id == "pilot_3"
        assert step.round_plan is not None
        assert step.queue is not None
        assert len(step.queue.items) == 2
        assert step.round_plan.scheduler_mode == "external_asha"
        assert step.queue.metadata["source_authority"] == "RoundExecutionPlan"
        assert step.queue.metadata["scheduler_mode"] == "external_asha"
        assert all("imgsz=640" in item.command.display() for item in step.queue.items)
        assert all(
            item.command.metadata.get("post_eval_required")
            for item in step.queue.items
        )
        assert all(
            item.command.metadata.get("paired_evidence_required")
            for item in step.queue.items
        )
        candidate_item, control_item = _paired_queue_items(step)
        for field in (
            "dataset_manifest_sha256",
            "subset_manifest_sha256",
            "batch_policy_hash",
            "eval_protocol_hash",
            "protocol_hash",
            "ultralytics_version",
            "epochs",
            "seed",
        ):
            assert candidate_item.command.metadata[field] == control_item.command.metadata[field]
        assert step.adapter_identity["adapter_ids"] == [COMPONENT_ID]
        assert step.adapter_identity["adapter_hashes"][COMPONENT_ID]
        assert step.adapter_identity["component_maturity"] == {
            COMPONENT_ID: "smoke_passed"
        }
        update = orchestrator.record_result(
            _complete_evidence(step, delta=delta, target_improved=False)
        )
        assert update.evidence_complete is True

    pilot_10 = orchestrator.next_step()
    assert pilot_10.action == "queue_assignment"
    assert pilot_10.assignment is not None
    assert pilot_10.assignment.stage_id == "pilot_10"
    assert pilot_10.assignment.candidate_id == "paper-sampling-1"
    update = orchestrator.record_result(
        _complete_evidence(pilot_10, delta=0.025, target_improved=True)
    )
    assert update.evidence_complete is True
    observation = orchestrator.scheduler.study.trial(
        pilot_10.assignment.trial_id
    ).observation("pilot_10")
    assert observation is not None
    assert observation.paired_result_verified is True
    assert orchestrator.scheduler.study.rungs[1].stage_id == "pilot_10"


def _write_catalog(root: Path) -> Path:
    path = root / "data" / "papers.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([
            {
                "paper_id": PAPER_ID,
                "title": "Small Object Sampling Fixture",
                "year": 2025,
                "category": "Small, Aerial, and Oriented Detection",
                "summary": "Small-object sampling improves recall for tiny instances.",
                "task_families": ["small_object_detection"],
                "detector_family": "one_stage",
                "component_ids": ["small_object_sampling"],
                "applicability": "direct_adapter_candidate",
                "harness_hints": ["Use only when AP_small and small-object recall are low."],
            }
        ], indent=2),
        encoding="utf-8",
    )
    return path


def _maturity_registry(tmp_path: Path) -> ComponentMaturityRegistry:
    contract = load_contracts(
        ResourcePaths.COMPONENTS_DIR / "sampling" / "small_object_sampling.yaml"
    )[0]
    artifact_types = {
        "runtime_integrated": "runtime_payload",
        "unit_tested": "unit_test_report",
        "smoke_passed": "smoke_report",
    }
    artifacts: list[ComponentMaturityArtifact] = []
    for target, artifact_type in artifact_types.items():
        path = tmp_path / "maturity" / f"{target}.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{target}: passed\n", encoding="utf-8")
        artifacts.append(ComponentMaturityArtifact(
            component_id=COMPONENT_ID,
            target_maturity=target,  # type: ignore[arg-type]
            artifact_type=artifact_type,  # type: ignore[arg-type]
            artifact_path=path,
            artifact_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            status="passed",
            producer="offline-state-machine-test",
            protocol_hash="component-cert-v1",
        ))
    registry = ComponentMaturityRegistry(tmp_path / "component_maturity_registry.yaml")
    registry.upsert(ComponentEvidenceOverlay(
        component_id=COMPONENT_ID,
        adapter_hash=adapter_source_hash(contract),
        code_commit="test-commit",
        ultralytics_version="test-ultralytics",
        protocol_hash="component-cert-v1",
        artifacts=artifacts,
    ))
    return registry


def _candidate(index: int, *, snapshot_hash: str, profile, decision):
    candidate_id = f"paper-sampling-{index}"
    prior = RecipePrior(
        prior_id=f"prior-paper-sampling-{index}",
        research_snapshot_hash=snapshot_hash,
        paper_ids=[PAPER_ID],
        component_ids=[COMPONENT_ID],
        target_error_facts=[{"fact_type": "area_metric", "subject": "small"}],
        target_metrics=["ap_small", "recall"],
        suggested_changed_variables=["data.sampling_policy"],
        baseline_protocol={"imgsz": 640, "protocol_hashes": [PROTOCOL_HASH]},
        evidence_prior=[RecipePriorEvidence(
            paper_id=PAPER_ID,
            claim="Small-object sampling may improve recall.",
            source_location="catalog:summary",
            evidence_level="paper_claim",
        )],
        expected_paper_effect={"ap_small": "unknown"},
        implementation_status="smoke_passed",
        yolo26_compatibility="compatible",
        required_adapter=["SmallObjectSamplingAdapter"],
        confidence=0.8 - index * 0.01,
        source_locations=["catalog:summary"],
    )
    source = node(candidate_id)
    source.candidate_config = source.candidate_config.model_copy(update={
        "components": [COMPONENT_ID],
        "action_id": f"materialized_{prior.prior_id}",
        "target_error_facts": prior.target_error_facts,
    })
    source.changed_variables = {"data.sampling_policy": f"bounded-small-{index}"}
    return PaperRecipeCandidateInput(
        prior=prior,
        method_profile=profile,
        implementation_decision=decision,
        compatibility=CompatibilityResult(ok=True),
        source_node=source,
        matched_control_node=node(f"control-{index}", control=True),
        component_family=f"sampling-{index}",
        bucket="exploration" if index == 3 else "exploitation",
    )


def _complete_evidence(step, *, delta: float, target_improved: bool) -> PaperCandidateEvidence:
    assert step.assignment is not None
    assert step.round_plan is not None
    candidate_item, _ = _paired_queue_items(step)
    paired = verified_paired_result(
        candidate_id=step.assignment.candidate_id,
        node_id=candidate_item.experiment_node.node_id,
        delta=delta,
        target_improved=target_improved,
        protocol_hash=candidate_item.command.metadata["protocol_hash"],
    )
    return PaperCandidateEvidence(
        assignment_id=step.assignment.assignment_id,
        post_eval_complete=True,
        error_facts_complete=True,
        paired_result=paired,
        target_error_improved_count=1 if target_improved else 0,
        diagnosis_gate_passed=(
            True if step.assignment.stage_id == "pilot_10" else None
        ),
    )


def _paired_queue_items(step):
    assert step.queue is not None
    candidate = next(
        item
        for item in step.queue.items
        if not item.command.metadata.get("matched_baseline_control")
    )
    control = next(
        item
        for item in step.queue.items
        if item.command.metadata.get("matched_baseline_control")
    )
    return candidate, control
