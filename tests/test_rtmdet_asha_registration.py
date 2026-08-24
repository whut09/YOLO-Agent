from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.maturity_helpers import with_smoke_artifact
from tests.neck_fixtures import neck_contracts, neck_node, neck_recipes
from yolo_agent.agents.asha_scheduler import ASHAScheduler
from yolo_agent.agents.auto_optimization_loop import _register_guarded_pilot_trials
from yolo_agent.agents.candidate_generator import CandidateConfig
from yolo_agent.agents.orchestrator import LoopOrchestrator
from yolo_agent.agents.paper_proposal_ledger import PaperCandidateCoverage
from yolo_agent.components.adapters.runtime import AdapterRuntimePayload
from yolo_agent.components.execution_bridge import ComponentExecutionBridge
from yolo_agent.core.command_spec import CommandSpec
from yolo_agent.core.experiment_graph import ExperimentNode
from yolo_agent.core.round_execution_plan import RoundExecutionPlan
from yolo_agent.core.run_context import RunContext


COMPONENT_ID = "neck.rtmdet_large_kernel"
RECIPE_ID = "yolo26_rtmdet_large_kernel_neck"
PROTOCOL_HASH = "rtmdet-matched-protocol"


def _registered_candidate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    certification_allowed: bool,
) -> tuple[int, ASHAScheduler, RunContext, ExperimentNode]:
    recipe = next(item for item in neck_recipes() if item.recipe_id == RECIPE_ID)
    candidate = neck_node(recipe, tmp_path)
    candidate.candidate_config = candidate.candidate_config.model_copy(
        update={
            "target_error_facts": [{"fact_type": "localization_error"}],
        }
    )
    runtime = ComponentExecutionBridge().prepare(
        recipe=recipe,
        node=candidate,
        contracts={
            COMPONENT_ID: with_smoke_artifact(neck_contracts()[COMPONENT_ID]),
        },
        training_config=dict(recipe.train_overrides),
        workspace=tmp_path / "runtime",
        protocol_hash=PROTOCOL_HASH,
    )
    assert runtime.status == "executable", runtime.blocked_by
    assert runtime.node.command_spec is not None
    runtime.node.command_spec.metadata.update(
        {
            "paper_readiness_state": "asha_eligible",
            "paper_readiness_blockers": "[]",
        }
    )

    control_command = CommandSpec.ultralytics_train(
        model="yolo26n.pt",
        data=tmp_path / "coco.yaml",
        project=tmp_path / "runs",
        name="matched-control",
        epochs=3,
        imgsz=640,
    ).model_copy(
        update={
            "metadata": {
                "matched_baseline_control": True,
                "baseline_protocol_hash": PROTOCOL_HASH,
            }
        }
    )
    control = ExperimentNode(
        node_id="node_matched_control",
        candidate_config=CandidateConfig(
            candidate_id="matched_baseline_control",
            base_model="yolo26n.pt",
            scale="n",
            framework="ultralytics",
        ),
        data_version="coco2017",
        seed=1,
        command=control_command.display(),
        command_spec=control_command,
    )
    context = RunContext(
        run_id="rtmdet-asha",
        run_root=tmp_path / "run-root",
        task_path=tmp_path / "task.yaml",
        data_yaml=tmp_path / "coco.yaml",
    )
    child = LoopOrchestrator(context)
    RoundExecutionPlan(
        run_id=context.run_id,
        round_id="round-1",
        deferred_nodes=[control, runtime.node],
    ).to_yaml(context.artifact_path("round_execution_plan.yaml"))
    monkeypatch.setattr(
        "yolo_agent.agents.auto_optimization_loop.ComponentQueueCertificationGate.evaluate",
        lambda *args, **kwargs: SimpleNamespace(
            allowed=certification_allowed,
            blockers=(
                []
                if certification_allowed
                else ["model_graph_resource_guard_failed:latency"]
            ),
            report_path=None,
            report_hash="rtmdet-certification",
        ),
    )
    monkeypatch.setattr(
        "yolo_agent.agents.auto_optimization_loop.validate_certified_runtime_node",
        lambda node: [],
    )
    scheduler = ASHAScheduler.create(context.run_id)
    registered = _register_guarded_pilot_trials(
        scheduler,
        child,
        [runtime.node],
    )
    return registered, scheduler, context, runtime.node


def test_rtmdet_candidate_registers_with_matched_control(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registered, scheduler, context, candidate = _registered_candidate(
        tmp_path,
        monkeypatch,
        certification_allowed=True,
    )

    assert registered == 1
    assert [trial.candidate_id for trial in scheduler.study.trials] == [RECIPE_ID]
    trial = scheduler.study.trials[0]
    assert trial.baseline_control_node is not None
    assert trial.target_error_facts == [{"fact_type": "localization_error"}]
    assert candidate.command_spec is not None
    payload = AdapterRuntimePayload.read(
        Path(candidate.command_spec.metadata["adapter_runtime_payload_path"]),
        verify_imports=False,
    )
    options = payload.model_graph_plugin[0].options
    assert options["graph_identity"]["target_node"] == "terminal_native_detect"
    assert options["graph_identity"]["preserves_one_to_one_head"] is True
    assert payload.rollback_plan.actions
    coverage = PaperCandidateCoverage.from_yaml(
        context.artifact_path("paper_candidate_coverage.yaml")
    )
    assert coverage.records[0].disposition == "queued"
    assert coverage.records[0].canonical_component_ids == [COMPONENT_ID]


def test_rtmdet_resource_guard_failure_remains_a_blocked_candidate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registered, scheduler, context, _ = _registered_candidate(
        tmp_path,
        monkeypatch,
        certification_allowed=False,
    )

    assert registered == 0
    assert len(scheduler.study.trials) == 1
    assert scheduler.study.trials[0].readiness_state == "pre_registered"
    assert scheduler.study.trials[0].status == "needs_evidence"
    assert scheduler.study.trials[0].pending_stage is None
    coverage = PaperCandidateCoverage.from_yaml(
        context.artifact_path("paper_candidate_coverage.yaml")
    )
    assert coverage.records[0].disposition == "blocked_runtime"
    assert coverage.records[0].reason_codes == [
        "model_graph_resource_guard_failed:latency"
    ]
