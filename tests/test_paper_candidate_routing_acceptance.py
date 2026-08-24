"""End-to-end acceptance for overall-mAP paper candidate routing."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from yolo_agent.agents.asha_scheduler import ASHAObservation, ASHAScheduler
from yolo_agent.agents.auto_optimization_loop import (
    _mark_paper_candidate_disposition,
    _register_guarded_pilot_trials,
)
from yolo_agent.agents.candidate_generator import CandidateConfig
from yolo_agent.agents.orchestrator import LoopOrchestrator
from yolo_agent.agents.paper_proposal_ledger import (
    PaperCandidateCoverage,
    PaperCandidateCoverageLedger,
    planned_recipe_disposition,
)
from yolo_agent.core.command_spec import CommandSpec
from yolo_agent.core.execution_fingerprint import execution_fingerprint
from yolo_agent.core.experiment_graph import ExperimentNode, MetricEvidence
from yolo_agent.core.matched_baseline import paired_metric_delta
from yolo_agent.core.optimization_objective import OptimizationObjective
from yolo_agent.core.round_execution_plan import (
    RoundExecutionPlan,
    build_round_execution_plan,
)
from yolo_agent.core.run_context import RunContext
from tests.test_overall_map_paper_routing_acceptance import (
    TARGET_COMPONENTS,
    _fact,
    _plan,
    _runtime_ready_registry,
)


PROTOCOL_HASH = "protocol-640"
DATASET_HASH = "coco2017-manifest"

IMPROVE_MAP_11_FACTS = [
    ("background_false_positive_class", "person"),
    ("high_confidence_false_positive", "person"),
    ("class_confusion_pair", "person:bicycle"),
    ("confidence_localization_mismatch", "overall"),
    ("localization_error", "overall"),
    ("localization_heavy_class", "overall"),
    ("assignment_conflict", "overall"),
    ("duplicate_prediction", "person"),
    ("class_low_ap", "long_tail_classes"),
    ("representation_gap", "overall"),
    ("capacity_gap", "overall"),
    ("scale_variation", "overall"),
    ("feature_relation_gap", "overall"),
]

COUPLED_ABLATION = [
    {"name": "baseline", "components": []},
    {"name": "A", "components": ["loss.hard_negative_classification"]},
    {"name": "B", "components": ["sampling.hard_negative_replay"]},
    {
        "name": "A+B",
        "components": [
            "loss.hard_negative_classification",
            "sampling.hard_negative_replay",
        ],
    },
]


@pytest.fixture
def improve_map_11_diagnosis(tmp_path: Path):  # type: ignore[no-untyped-def]
    contracts, registry = _runtime_ready_registry()
    facts = [
        _fact(fact_type, subject, severity="high")
        for fact_type, subject in IMPROVE_MAP_11_FACTS
    ]
    return _plan(tmp_path, contracts, registry, facts), registry


def test_overall_map_cohort_survives_planner_ledger_plan_and_asha(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    improve_map_11_diagnosis,
) -> None:  # type: ignore[no-untyped-def]
    paper_plan, recipe_registry = improve_map_11_diagnosis
    planned_by_component = _planned_recipes_by_component(
        paper_plan.candidate_inventory,
        recipe_registry,
    )

    assert TARGET_COMPONENTS <= set(planned_by_component)
    assert all(
        planned_by_component[component_id].matched_error_fact_ids
        for component_id in TARGET_COMPONENTS
    )

    child, objective = _orchestrator(tmp_path, run_id="improve-map-11-acceptance")
    baseline = _candidate_node(
        tmp_path,
        candidate_id="matched_baseline_control",
        component_ids=[],
        changed_variables={},
        matched_control=True,
    )
    atomic_nodes = [
        _candidate_node(
            tmp_path,
            candidate_id=f"paper_{component_id.replace('.', '_')}",
            component_ids=[component_id],
            changed_variables={f"runtime.{component_id}": "enabled"},
            recipe_id=planned_by_component[component_id].recipe_id,
            recipe_version=planned_by_component[component_id].version,
        )
        for component_id in sorted(TARGET_COMPONENTS)
    ]
    coupled_nodes = _coupled_nodes(tmp_path)
    small_object = _candidate_node(
        tmp_path,
        candidate_id="paper_small_object_only",
        component_ids=["sampling.small_object"],
        changed_variables={"data.small_object_sampling": "enabled"},
        recipe_id="yolo26_small_object_sampling",
    )
    candidate_nodes = [*atomic_nodes, *coupled_nodes, small_object]

    ledger = PaperCandidateCoverageLedger(
        child.context.artifact_path("paper_candidate_coverage.yaml"),
        run_id=child.context.run_id,
        protocol_hash=objective.baseline_protocol_hash,
    )
    ledger.upsert_many(
        _planner_ledger_record(
            child.context.run_id,
            node,
            paper_plan=paper_plan,
            recipe_registry=recipe_registry,
        )
        for node in candidate_nodes
    )

    round_plan = build_round_execution_plan(
        run_id=child.context.run_id,
        nodes=candidate_nodes,
        baseline_control_node=baseline,
        coupled_node_ids={node.node_id for node in coupled_nodes},
        objective_hash=objective.objective_hash,
        primary_metric="map50_95",
    )
    round_plan.to_yaml(child.context.artifact_path("round_execution_plan.yaml"))
    _allow_mock_runtime_readiness(monkeypatch)

    scheduler = ASHAScheduler.create(child.context.run_id)
    registered = _register_guarded_pilot_trials(
        scheduler,
        child,
        candidate_nodes,
    )

    eligible_nodes = [*atomic_nodes, *coupled_nodes]
    eligible_ids = {
        node.candidate_config.candidate_id
        for node in eligible_nodes
    }
    assert registered == len(eligible_nodes)
    trial_ids = {trial.candidate_id for trial in scheduler.study.trials}
    assert eligible_ids <= trial_ids
    blocked_small = scheduler.study.trial(
        f"{child.context.run_id}:paper_small_object_only"
    )
    assert blocked_small.readiness_state == "pre_registered"
    assert blocked_small.status == "needs_evidence"
    assert blocked_small.pending_stage is None
    assert all(trial.baseline_control_node is not None for trial in scheduler.study.trials)

    persisted_plan = RoundExecutionPlan.from_yaml(
        child.context.artifact_path("round_execution_plan.yaml")
    )
    plan_candidate_ids = {
        node.candidate_config.candidate_id
        for node in persisted_plan.deferred_nodes
        if not node.command_spec.metadata.get("matched_baseline_control")
    }
    assert {node.candidate_config.candidate_id for node in candidate_nodes} <= plan_candidate_ids
    coupled_ablation_ids = {
        item.candidate_id
        for item in persisted_plan.ablation_nodes
        if item.candidate_id.startswith("paper_hard_negative__")
    }
    assert coupled_ablation_ids == {
        "paper_hard_negative__a",
        "paper_hard_negative__b",
        "paper_hard_negative__a_b",
    }
    assert all(
        item.valid
        for item in persisted_plan.ablation_nodes
        if item.candidate_id in coupled_ablation_ids
    )
    assert all(
        json.loads(node.command_spec.metadata["internal_ablation_plan"])
        == COUPLED_ABLATION
        for node in coupled_nodes
    )

    coverage = PaperCandidateCoverage.from_yaml(ledger.path)
    assert len(coverage.records) == len(candidate_nodes)
    assert len(coverage.current_by_fingerprint) == len(candidate_nodes)
    assert all(record.disposition for record in coverage.records)
    dispositions = {
        record.candidate_id: record.disposition
        for record in coverage.records
    }
    assert dispositions["paper_small_object_only"] == "incompatible"
    assert {
        dispositions[node.candidate_config.candidate_id]
        for node in eligible_nodes
    } == {"queued"}
    assert child.context.metadata["asha_registration_summary"] == {
        "considered": len(candidate_nodes),
        "registered": len(eligible_nodes),
        "newly_registered": len(eligible_nodes),
        "already_registered": 0,
        "queued": len(eligible_nodes),
        "deferred": 0,
        "terminal_rejections": 1,
        "retryable_rejections": 0,
    }


@pytest.mark.parametrize(
    ("baseline_overrides", "expected_reason"),
    [
        ({"protocol_hash": "old-protocol"}, "protocol_hash_mismatch"),
        ({"split": "train2017"}, "split_mismatch"),
    ],
)
def test_protocol_or_split_mismatch_never_produces_paired_delta(
    baseline_overrides: dict[str, object],
    expected_reason: str,
) -> None:
    candidate = _metric_record(role="current_observation", value=0.42)
    baseline = _metric_record(
        role="baseline_reference",
        value=0.40,
        **baseline_overrides,
    )

    control, delta = paired_metric_delta(candidate, [baseline])

    assert delta is None
    assert control.matched is False
    assert expected_reason in control.mismatch_reasons


def test_mock_candidate_failure_and_old_protocol_recovery_are_isolated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    child, objective = _orchestrator(tmp_path, run_id="routing-isolation")
    baseline = _candidate_node(
        tmp_path,
        candidate_id="matched_baseline_control",
        component_ids=[],
        changed_variables={},
        matched_control=True,
    )
    failed = _candidate_node(
        tmp_path,
        candidate_id="paper_quality_correlation",
        component_ids=["loss.quality.correlation"],
        changed_variables={"loss.quality.correlation": "enabled"},
    )
    retained = _candidate_node(
        tmp_path,
        candidate_id="paper_rtmdet_large_kernel",
        component_ids=["neck.rtmdet_large_kernel"],
        changed_variables={"neck.rtmdet_large_kernel": "enabled"},
    )
    ledger = PaperCandidateCoverageLedger(
        child.context.artifact_path("paper_candidate_coverage.yaml"),
        run_id=child.context.run_id,
        protocol_hash=objective.baseline_protocol_hash,
    )
    ledger.upsert_many(
        _runtime_ledger_record(child.context.run_id, node)
        for node in (failed, retained)
    )
    build_round_execution_plan(
        run_id=child.context.run_id,
        nodes=[failed, retained],
        baseline_control_node=baseline,
        objective_hash=objective.objective_hash,
    ).to_yaml(child.context.artifact_path("round_execution_plan.yaml"))
    _allow_mock_runtime_readiness(monkeypatch)
    scheduler = ASHAScheduler.create(child.context.run_id)
    assert _register_guarded_pilot_trials(
        scheduler,
        child,
        [failed, retained],
    ) == 2

    trial_ids_before = {trial.trial_id for trial in scheduler.study.trials}
    MockRoutingBackend.fail_candidate(
        scheduler,
        trial_id=f"{child.context.run_id}:{failed.candidate_config.candidate_id}",
    )
    _mark_paper_candidate_disposition(
        child,
        failed,
        disposition="blocked_runtime",
        reasons=["mock_candidate_training_failed"],
        source_stage="mock_backend",
    )

    assert {trial.trial_id for trial in scheduler.study.trials} == trial_ids_before
    assert scheduler.study.trial(
        f"{child.context.run_id}:{failed.candidate_config.candidate_id}"
    ).status == "failed"
    assert scheduler.study.trial(
        f"{child.context.run_id}:{retained.candidate_config.candidate_id}"
    ).status == "waiting"
    coverage = ledger.read()
    assert len(coverage.records) == 2
    assert {
        record.candidate_id: record.disposition
        for record in coverage.records
    } == {
        "paper_quality_correlation": "blocked_runtime",
        "paper_rtmdet_large_kernel": "queued",
    }

    old_candidate = _metric_record(
        role="current_observation",
        value=0.43,
        run_id="old-run",
        protocol_hash="old-protocol",
    )
    old_baseline = _metric_record(
        role="baseline_reference",
        value=0.40,
        run_id="old-run",
        protocol_hash="old-protocol",
    )
    old_control, old_delta = paired_metric_delta(old_candidate, [old_baseline])
    assert old_control.matched is True
    assert old_delta is not None
    assert old_delta.match_key.protocol_hash != objective.baseline_protocol_hash

    _mark_paper_candidate_disposition(
        child,
        retained,
        disposition="evidence_recovery",
        reasons=["old_run_protocol_mismatch", "isolated_run_required"],
        source_stage="paired_evidence_import",
    )
    recovered = ledger.read().current_by_fingerprint[
        execution_fingerprint(retained)
    ]
    assert recovered.disposition == "evidence_recovery"
    assert recovered.reason_codes == [
        "old_run_protocol_mismatch",
        "isolated_run_required",
    ]
    assert len(scheduler.study.trials) == 2


def test_zero_asha_registration_requires_persisted_candidate_blocker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    child, objective = _orchestrator(tmp_path, run_id="zero-registration-blocked")
    baseline = _candidate_node(
        tmp_path,
        candidate_id="matched_baseline_control",
        component_ids=[],
        changed_variables={},
        matched_control=True,
    )
    candidate = _candidate_node(
        tmp_path,
        candidate_id="paper_quality_missing_evidence",
        component_ids=["loss.quality.correlation"],
        changed_variables={"loss.quality.correlation": "enabled"},
    )
    candidate.candidate_config.target_error_facts = []
    ledger = PaperCandidateCoverageLedger(
        child.context.artifact_path("paper_candidate_coverage.yaml"),
        run_id=child.context.run_id,
        protocol_hash=objective.baseline_protocol_hash,
    )
    ledger.upsert(_runtime_ledger_record(child.context.run_id, candidate))
    build_round_execution_plan(
        run_id=child.context.run_id,
        nodes=[candidate],
        baseline_control_node=baseline,
        objective_hash=objective.objective_hash,
    ).to_yaml(child.context.artifact_path("round_execution_plan.yaml"))
    _allow_mock_runtime_readiness(monkeypatch)

    registered = _register_guarded_pilot_trials(
        ASHAScheduler.create(child.context.run_id),
        child,
        [candidate],
    )

    assert registered == 0
    assert child.context.metadata["asha_registration_all_candidates_dispositioned"] is True
    assert child.context.metadata["asha_registration_summary"] == {
        "considered": 1,
        "registered": 0,
        "newly_registered": 0,
        "already_registered": 0,
        "queued": 0,
        "deferred": 0,
        "terminal_rejections": 0,
        "retryable_rejections": 1,
    }
    record = ledger.read().records[0]
    assert record.disposition == "evidence_recovery"
    assert record.reason_codes == ["target_error_facts_missing"]


def test_zero_asha_registration_rejects_silent_disposition_drop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    child, objective = _orchestrator(tmp_path, run_id="zero-registration-silent-drop")
    baseline = _candidate_node(
        tmp_path,
        candidate_id="matched_baseline_control",
        component_ids=[],
        changed_variables={},
        matched_control=True,
    )
    candidate = _candidate_node(
        tmp_path,
        candidate_id="paper_quality_silently_dropped",
        component_ids=["loss.quality.correlation"],
        changed_variables={"loss.quality.correlation": "enabled"},
    )
    candidate.candidate_config.target_error_facts = []
    PaperCandidateCoverageLedger(
        child.context.artifact_path("paper_candidate_coverage.yaml"),
        run_id=child.context.run_id,
        protocol_hash=objective.baseline_protocol_hash,
    ).upsert(_runtime_ledger_record(child.context.run_id, candidate))
    build_round_execution_plan(
        run_id=child.context.run_id,
        nodes=[candidate],
        baseline_control_node=baseline,
        objective_hash=objective.objective_hash,
    ).to_yaml(child.context.artifact_path("round_execution_plan.yaml"))
    _allow_mock_runtime_readiness(monkeypatch)
    monkeypatch.setattr(
        "yolo_agent.agents.auto_optimization_loop._mark_paper_candidate_disposition",
        lambda *args, **kwargs: None,
    )

    with pytest.raises(RuntimeError, match="lack a terminal paper proposal disposition"):
        _register_guarded_pilot_trials(
            ASHAScheduler.create(child.context.run_id),
            child,
            [candidate],
        )


class MockRoutingBackend:
    """CPU-only backend signal used to exercise isolated ASHA failure handling."""

    @staticmethod
    def fail_candidate(scheduler: ASHAScheduler, *, trial_id: str) -> None:
        scheduler.report(
            trial_id,
            ASHAObservation(
                stage_id="pilot_3",
                node_id=f"node_{trial_id.replace(':', '_')}",
                seed=1,
                evidence_complete=False,
                failure_reason="mock_candidate_training_failed",
            ),
        )


def _planned_recipes_by_component(planned_recipes, recipe_registry):  # type: ignore[no-untyped-def]
    result = {}
    for planned in planned_recipes:
        recipe = recipe_registry.get(planned.recipe_id, planned.version)
        if recipe is None:
            continue
        for component_id in recipe.component_ids:
            if component_id in TARGET_COMPONENTS:
                result.setdefault(component_id, planned)
    return result


def _orchestrator(
    tmp_path: Path,
    *,
    run_id: str,
) -> tuple[LoopOrchestrator, OptimizationObjective]:
    context = RunContext(
        run_id=run_id,
        run_root=tmp_path / "runs",
        task_path=tmp_path / "task.yaml",
        data_yaml=tmp_path / "coco.yaml",
        dataset_version="coco2017",
        dataset_manifest_sha256=DATASET_HASH,
    )
    child = LoopOrchestrator(context)
    objective = OptimizationObjective(
        goal_description="Improve overall mAP",
        primary_metric="map50_95",
        baseline_run_id=run_id,
        baseline_candidate_id="matched_baseline_control",
        baseline_protocol_hash=PROTOCOL_HASH,
    )
    objective_path = context.artifact_path("optimization_objective.yaml")
    objective.to_yaml(objective_path)
    context.metadata["optimization_objective_path"] = objective_path.as_posix()
    return child, objective


def _candidate_node(
    tmp_path: Path,
    *,
    candidate_id: str,
    component_ids: list[str],
    changed_variables: dict[str, object],
    recipe_id: str | None = None,
    recipe_version: str = "v1.0.0",
    matched_control: bool = False,
    combination_id: str | None = None,
) -> ExperimentNode:
    metadata: dict[str, object] = {
        "matched_baseline_control": matched_control,
        "matched_pilot_required": not matched_control,
        "run_protocol_hash": PROTOCOL_HASH,
        "baseline_protocol_hash": PROTOCOL_HASH,
        "protocol_hash": PROTOCOL_HASH,
        "dataset_manifest_sha256": DATASET_HASH,
        "split": "val2017",
        "fidelity": "pilot_3",
    }
    if not matched_control:
        metadata.update(
            {
                "adapter_runtime_entrypoint": (
                    "yolo_agent.adapters.ultralytics.runtime_entrypoint"
                ),
                "component_recipe_id": recipe_id or candidate_id,
                "component_recipe_version": recipe_version,
                "paper_readiness_state": "asha_eligible",
                "paper_readiness_blockers": "[]",
            }
        )
    if combination_id is not None:
        metadata.update(
            {
                "ablation_combination_id": combination_id,
                "coupling_reason": (
                    "Hard-negative classification and replay target complementary "
                    "false-positive evidence."
                ),
                "coupling_source_papers": json.dumps(["paper:hard-negative"]),
                "internal_ablation_plan": json.dumps(COUPLED_ABLATION),
            }
        )
    command = CommandSpec.ultralytics_train(
        model="yolo26n.pt",
        data=tmp_path / "coco.yaml",
        project=tmp_path / "ultralytics",
        name=candidate_id,
        epochs=3,
        imgsz=640,
        batch=16,
        seed=1,
        metadata=metadata,
    )
    return ExperimentNode(
        node_id=f"node_{candidate_id}",
        candidate_config=CandidateConfig(
            candidate_id=candidate_id,
            base_model="yolo26n.pt",
            scale="n",
            framework="ultralytics",
            components=component_ids,
            action_domain="paper",
            action_id=recipe_id or candidate_id,
            search_tier="method",
            target_error_facts=(
                []
                if matched_control
                else [{"fact_type": "localization_error", "subject": "overall"}]
            ),
        ),
        data_version="coco2017",
        seed=1,
        command=command.display(),
        command_spec=command,
        changed_variables=changed_variables,
    )


def _coupled_nodes(tmp_path: Path) -> list[ExperimentNode]:
    return [
        _candidate_node(
            tmp_path,
            candidate_id=f"paper_hard_negative__{suffix}",
            component_ids=components,
            changed_variables=changed_variables,
            recipe_id="yolo26_hard_negative_pair",
            combination_id=combination_id,
        )
        for suffix, combination_id, components, changed_variables in (
            (
                "a",
                "A",
                ["loss.hard_negative_classification"],
                {"loss.hard_negative_classification": "enabled"},
            ),
            (
                "b",
                "B",
                ["sampling.hard_negative_replay"],
                {"sampling.hard_negative_replay": "enabled"},
            ),
            (
                "a_b",
                "A+B",
                [
                    "loss.hard_negative_classification",
                    "sampling.hard_negative_replay",
                ],
                {
                    "loss.hard_negative_classification": "enabled",
                    "sampling.hard_negative_replay": "enabled",
                },
            ),
        )
    ]


def _runtime_ledger_record(run_id: str, node: ExperimentNode):  # type: ignore[no-untyped-def]
    metadata = node.command_spec.metadata
    return planned_recipe_disposition(
        run_id=run_id,
        round_index=1,
        recipe_id=str(metadata.get("component_recipe_id") or node.candidate_config.action_id),
        recipe_version=str(metadata.get("component_recipe_version") or "v1.0.0"),
        component_ids=node.candidate_config.components,
        decision="selected",
        reasons=[],
        related_papers=[f"paper:{node.candidate_config.candidate_id}"],
        method_profile_ids=[f"profile:{node.candidate_config.candidate_id}"],
        matched_error_fact_ids=["fact:overall-map"],
        execution_fingerprint=execution_fingerprint(node),
        candidate_id=node.candidate_config.candidate_id,
        source_stage="paper_recipe_planner",
    )


def _metric_record(
    *,
    role: str,
    value: float,
    run_id: str = "routing-run",
    **overrides: object,
) -> MetricEvidence:
    values: dict[str, object] = {
        "candidate_id": (
            "matched_baseline_control"
            if role == "baseline_reference"
            else "paper_candidate"
        ),
        "node_id": (
            "node_matched_baseline_control"
            if role == "baseline_reference"
            else "node_paper_candidate"
        ),
        "run_id": run_id,
        "origin_run_id": run_id,
        "evidence_role": role,
        "inheritance_depth": 0,
        "dataset_manifest_sha256": DATASET_HASH,
        "protocol_hash": PROTOCOL_HASH,
        "subset_manifest_sha256": "pilot-subset",
        "seed": 1,
        "epochs": 3,
        "fidelity": "pilot_3",
        "batch_policy_hash": "batch-16",
        "ultralytics_version": "mock",
        "imgsz": 640,
        "eval_protocol_hash": "coco-eval",
        "split": "val2017",
        "metric_name": "map50_95",
        "value": value,
        "source": "mock_backend",
        "verified": True,
    }
    values.update(overrides)
    return MetricEvidence.model_validate(values)


def _planner_ledger_record(
    run_id: str,
    node: ExperimentNode,
    *,
    paper_plan,
    recipe_registry,
):  # type: ignore[no-untyped-def]
    metadata = node.command_spec.metadata
    recipe_id = str(metadata.get("component_recipe_id") or node.candidate_config.action_id)
    recipe_version = str(metadata.get("component_recipe_version") or "v1.0.0")
    planned = next(
        (
            item
            for item in paper_plan.candidate_inventory
            if item.recipe_id == recipe_id and item.version == recipe_version
        ),
        None,
    )
    raw_ablation = metadata.get("internal_ablation_plan")
    ablation = json.loads(raw_ablation) if isinstance(raw_ablation, str) else []
    raw_papers = metadata.get("coupling_source_papers")
    coupling_papers = json.loads(raw_papers) if isinstance(raw_papers, str) else []
    return planned_recipe_disposition(
        run_id=run_id,
        round_index=11,
        recipe_id=recipe_id,
        recipe_version=recipe_version,
        component_ids=node.candidate_config.components,
        decision="selected",
        reasons=[],
        related_papers=(
            planned.related_papers
            if planned is not None and planned.related_papers
            else coupling_papers or [f"paper:{recipe_id}"]
        ),
        method_profile_ids=(
            planned.related_method_profile_ids
            if planned is not None
            else [f"profile:{recipe_id}"]
        ),
        matched_error_fact_ids=(
            planned.matched_error_fact_ids
            if planned is not None
            else ["fact:overall-map"]
        ),
        execution_fingerprint=execution_fingerprint(node),
        candidate_id=node.candidate_config.candidate_id,
        combination_id=(
            str(metadata["ablation_combination_id"])
            if metadata.get("ablation_combination_id") is not None
            else None
        ),
        coupling_reason=(
            str(metadata["coupling_reason"])
            if metadata.get("coupling_reason") is not None
            else None
        ),
        coupling_source_papers=coupling_papers,
        internal_ablation_plan=ablation,
        source_stage="paper_recipe_planner",
    )


def _allow_mock_runtime_readiness(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "yolo_agent.agents.auto_optimization_loop.AutomaticRuntimeReadinessGate.evaluate_node",
        lambda self, node: SimpleNamespace(
            allowed=True,
            blockers=[],
            artifact_path=None,
        ),
    )
    monkeypatch.setattr(
        "yolo_agent.agents.auto_optimization_loop.ComponentQueueCertificationGate.evaluate",
        lambda *args, **kwargs: SimpleNamespace(
            allowed=True,
            blockers=[],
            report_path=None,
            report_hash="mock-certified",
        ),
    )
    monkeypatch.setattr(
        "yolo_agent.agents.auto_optimization_loop.validate_certified_runtime_node",
        lambda node: [],
    )
    monkeypatch.setattr(
        "yolo_agent.agents.auto_optimization_loop._distillation_runtime_blockers",
        lambda node, control: [],
    )
