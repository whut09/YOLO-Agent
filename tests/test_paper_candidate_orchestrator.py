from __future__ import annotations

import json
from pathlib import Path

from yolo_agent.agents.candidate_generator import CandidateConfig
from yolo_agent.agents.decision_bundle import DecisionContext
from yolo_agent.agents.paper_candidate_orchestrator import (
    PaperCandidateEvidence,
    PaperCandidateOrchestrator,
    PaperCandidateOrchestratorConfig,
    PaperCandidateSubmission,
)
from yolo_agent.agents.paper_component_gate import PaperComponentGateResult
from yolo_agent.agents.recipe_critic import RecipeCriticReport
from yolo_agent.core.command_spec import CommandSpec
from yolo_agent.core.experiment_graph import ExperimentNode
from yolo_agent.recipes.paper_priors import RecipePrior, RecipePriorEvidence
from yolo_agent.recipes.schemas import AtomicRecipe
from yolo_agent.research.snapshot import ResearchSnapshot, research_snapshot_hash
from tests.paired_result_helpers import verified_paired_result


SNAPSHOT_PAYLOAD = {
    "schema_version": "research_snapshot.v1",
    "papers_version": "papers-v1",
    "component_registry_version": "components-v1",
    "recipe_registry_version": "recipes-v1",
    "classifications_version": "classifications-v1",
    "extractions_version": "extractions-v1",
    "compatibility_version": "compatibility-v1",
    "reproduction_queue_version": "reproduction-v1",
    "paper_count": 1,
    "component_count": 1,
    "recipe_count": 1,
}
SNAPSHOT_HASH = research_snapshot_hash(SNAPSHOT_PAYLOAD)


def _snapshot(hash_value: str = SNAPSHOT_HASH) -> ResearchSnapshot:
    return ResearchSnapshot(
        **SNAPSHOT_PAYLOAD,
        snapshot_hash=hash_value,
        paper_intelligence="available",
        frozen=True,
    )


def _node(candidate_id: str, *, control: bool = False, imgsz: int = 640) -> ExperimentNode:
    command = CommandSpec.ultralytics_train(
        model="yolo26n.pt",
        data="coco.yaml",
        project="runs/ultralytics",
        name=candidate_id,
        epochs=3,
        imgsz=imgsz,
        batch=48,
        metadata={
            "objective_hash": "objective-1",
            "protocol_hash": "protocol-640",
            "matched_baseline_control": control,
        },
    )
    return ExperimentNode(
        node_id=f"node_{candidate_id}",
        candidate_config=CandidateConfig(
            candidate_id=candidate_id,
            base_model="yolo26n",
            scale="n",
            framework="ultralytics",
        ),
        data_version="coco2017",
        command=command.display(),
        command_spec=command,
        changed_variables={} if control else {"data.sampler": candidate_id},
    )


def _submission(
    candidate_id: str,
    *,
    bucket: str = "exploitation",
    family: str | None = None,
    round_index: int = 1,
    gate_eligible: bool = True,
    critic_accepted: bool = True,
    matched_control: bool = True,
    imgsz: int = 640,
    snapshot_hash: str = SNAPSHOT_HASH,
) -> PaperCandidateSubmission:
    component_id = f"sampling.{candidate_id}"
    prior = RecipePrior(
        prior_id=f"prior-{candidate_id}",
        research_snapshot_hash=SNAPSHOT_HASH,
        paper_ids=[f"paper-{candidate_id}"],
        component_ids=[component_id],
        target_error_facts=[{"fact_type": "area_metric", "subject": "small"}],
        target_metrics=["ap_small"],
        suggested_changed_variables=["data.sampler"],
        baseline_protocol={"imgsz": 640},
        evidence_prior=[RecipePriorEvidence(
            paper_id=f"paper-{candidate_id}",
            claim="small-object sampling may improve AP_small",
            source_location="summary",
            evidence_level="paper_claim",
        )],
        expected_paper_effect={"ap_small": "unknown"},
        implementation_status="smoke_passed",
        yolo26_compatibility="compatible",
        confidence=0.8,
        source_locations=["summary"],
    )
    recipe = AtomicRecipe(
        recipe_id=f"recipe-{candidate_id}",
        version="v1",
        target_error_facts=[{"fact_type": "area_metric", "subject": "small"}],
        target_metrics=["ap_small"],
        component_ids=[component_id],
        train_overrides={"imgsz": 640},
        fixed_variables={"imgsz": 640},
        primary_changed_variable="data.sampler",
        evidence_prior=[{"paper_id": f"paper-{candidate_id}", "evidence_level": "paper_claim"}],
        stop_conditions=["stop when paired AP_small does not improve"],
        maturity="smoke_passed",
    )
    return PaperCandidateSubmission(
        decision_context=DecisionContext(
            run_id="base-run",
            research_snapshot_hash=snapshot_hash,
            research_snapshot_verified=True,
            paper_intelligence="available",
        ),
        research_snapshot=_snapshot(),
        recipe_prior=prior,
        recipe=recipe,
        eligibility=PaperComponentGateResult(
            eligible=gate_eligible,
            decision="eligible" if gate_eligible else "blocked",
            blocked_by=[] if gate_eligible else ["test_rejection"],
            changed_variables=["data.sampler"],
            paper_prior=[{"paper_id": f"paper-{candidate_id}"}],
            local_evidence=[{"matched_baseline": True}],
            execution_class="pilot_candidate" if gate_eligible else "paper_only",
            eligibility_token=f"token-{candidate_id}" if gate_eligible else None,
        ),
        critic=RecipeCriticReport(
            recipe_id=recipe.recipe_id,
            decision="accepted" if critic_accepted else "rejected",
            accepted=critic_accepted,
        ),
        source_node=_node(candidate_id, imgsz=imgsz),
        matched_control_node=_node(f"baseline-{candidate_id}", control=True) if matched_control else None,
        component_family=family or f"family-{candidate_id}",
        bucket=bucket,
        round_index=round_index,
    )


def _complete_evidence(step, delta: float, *, target_improved: bool = False) -> PaperCandidateEvidence:
    paired = verified_paired_result(
        candidate_id=step.assignment.candidate_id,
        node_id=step.round_plan.execution_nodes[-1].node_id,
        delta=delta,
        target_improved=target_improved,
    )
    return PaperCandidateEvidence(
        assignment_id=step.assignment.assignment_id,
        post_eval_complete=True,
        error_facts_complete=True,
        paired_result=paired,
        target_error_improved_count=1 if target_improved else 0,
        diagnosis_gate_passed=True if step.assignment.stage_id != "pilot_3" else None,
    )


def test_paper_candidates_follow_complete_asha_state_machine(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "base-run"
    orchestrator = PaperCandidateOrchestrator(run_dir, base_run_id="base-run")
    report = orchestrator.register_cohort([
        _submission("a"),
        _submission("b", bucket="exploration"),
        _submission("c"),
    ])
    assert sorted(report.registered) == ["a", "b", "c"]

    pilot_steps = []
    for delta in (0.03, 0.02, 0.01):
        step = orchestrator.next_step()
        pilot_steps.append(step)
        assert step.action == "queue_assignment"
        assert step.assignment.stage_id == "pilot_3"
        assert step.queue.metadata["source_authority"] == "RoundExecutionPlan"
        assert step.queue.metadata["scheduler_mode"] == "external_asha"
        assert len(step.queue.items) == 2
        assert all("imgsz=640" in item.command.display() for item in step.queue.items)
        assert all(item.command.metadata.get("paper_prior_id") is None or item.command.metadata["paper_prior_id"].startswith("prior-") for item in step.queue.items)
        update = orchestrator.record_result(_complete_evidence(step, delta))
        assert update.evidence_complete is True

    pilot_10 = orchestrator.next_step()
    assert pilot_10.action == "queue_assignment"
    assert pilot_10.assignment.stage_id == "pilot_10"
    assert pilot_10.assignment.candidate_id == "a"
    orchestrator.record_result(_complete_evidence(pilot_10, 0.025, target_improved=True))

    recommendation = orchestrator.next_step()
    assert recommendation.action == "full_candidate_recommendation"
    assert recommendation.recommended_candidate_ids == ["a"]
    assert (run_dir.parent / "policy_memory.jsonl").is_file()
    assert (run_dir / "artifacts" / "paper_component_reproduction.yaml").is_file()
    assert (run_dir / "artifacts" / "paper_recipe_report.yaml").is_file()
    ledger = [json.loads(line) for line in (run_dir / "artifacts" / "decision_ledger.jsonl").read_text().splitlines()]
    assert {item["decision_type"] for item in ledger} == {
        "paper_candidate_registration",
        "paper_candidate_assignment",
        "paper_candidate_result",
    }


def test_incomplete_evidence_queues_recovery_without_training(tmp_path: Path) -> None:
    orchestrator = PaperCandidateOrchestrator(tmp_path / "run", base_run_id="base")
    orchestrator.register_cohort([_submission("a"), _submission("b"), _submission("c")])
    step = orchestrator.next_step()
    update = orchestrator.record_result(PaperCandidateEvidence(
        assignment_id=step.assignment.assignment_id,
        post_eval_complete=False,
        error_facts_complete=False,
    ))
    assert update.evidence_complete is False
    recovery = orchestrator.next_step()
    assert recovery.action == "evidence_recovery"
    assert recovery.queue.metadata["source_authority"] == "RoundExecutionPlan"
    assert recovery.queue.metadata["evidence_recovery_only"] is True
    assert all(item.command.command_type != "train" for item in recovery.queue.items)


def test_registration_rejects_untrusted_inputs_and_respects_budget(tmp_path: Path) -> None:
    orchestrator = PaperCandidateOrchestrator(
        tmp_path / "run",
        base_run_id="base",
        config=PaperCandidateOrchestratorConfig(max_registered_candidates=3, exploitation_ratio=2 / 3),
    )
    report = orchestrator.register_cohort([
        _submission("gate", gate_eligible=False),
        _submission("critic", critic_accepted=False),
        _submission("control", matched_control=False),
        _submission("size", imgsz=672),
        _submission("snapshot", snapshot_hash="other"),
        _submission("exploit-1"),
        _submission("exploit-2"),
        _submission("exploit-3"),
        _submission("explore", bucket="exploration"),
    ])
    assert set(report.rejected) == {"gate", "critic", "control", "size", "snapshot"}
    assert sorted(report.registered) == ["exploit-1", "exploit-2", "explore"]
    assert report.deferred["exploit-3"] == "deferred_by_exploit_explore_budget"


def test_family_cooldown_and_minimum_cohort_are_enforced(tmp_path: Path) -> None:
    orchestrator = PaperCandidateOrchestrator(tmp_path / "run", base_run_id="base")
    orchestrator.state.family_last_round["sampling"] = 2
    cooldown = orchestrator.register_cohort([
        _submission("cooldown", family="sampling", round_index=3),
        _submission("a", round_index=3),
        _submission("b", round_index=3),
    ])
    assert "cooldown" in cooldown.deferred
    for delta in (0.03, 0.02):
        step = orchestrator.next_step()
        orchestrator.record_result(_complete_evidence(step, delta))
    waiting = orchestrator.next_step()
    assert waiting.action == "awaiting_pilot_3_cohort"
    assert "2/3" in waiting.reason


def test_policy_evaluation_artifact_has_no_queue_authority(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True)
    (run_dir / "policy_evaluation.yaml").write_text("accepted: [unsafe_candidate]\n", encoding="utf-8")
    orchestrator = PaperCandidateOrchestrator(run_dir, base_run_id="base")
    assert orchestrator.next_step().action == "idle"
    assert not (run_dir / "execution_queue.yaml").exists()


def test_configured_cohort_floor_is_applied_to_asha(tmp_path: Path) -> None:
    orchestrator = PaperCandidateOrchestrator(
        tmp_path / "run",
        base_run_id="base",
        config=PaperCandidateOrchestratorConfig(
            min_pilot_3_cohort=4,
            max_registered_candidates=4,
            exploitation_ratio=1.0,
        ),
    )
    orchestrator.register_cohort([_submission(value) for value in ("a", "b", "c", "d")])
    for delta in (0.04, 0.03, 0.02):
        step = orchestrator.next_step()
        orchestrator.record_result(_complete_evidence(step, delta))
    fourth = orchestrator.next_step()
    assert fourth.assignment.stage_id == "pilot_3"
    orchestrator.record_result(_complete_evidence(fourth, 0.01))
    assert orchestrator.next_step().assignment.stage_id == "pilot_10"
