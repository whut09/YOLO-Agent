from pathlib import Path
from types import SimpleNamespace
import hashlib

import pytest

from yolo_agent.agents.asha_scheduler import ASHAScheduler
from yolo_agent.agents.auto_optimization_loop import _register_guarded_pilot_trials
from yolo_agent.agents.candidate_generator import CandidateConfig
from yolo_agent.agents.orchestrator import LoopOrchestrator
from yolo_agent.agents.paper_proposal_ledger import PaperCandidateCoverage
from yolo_agent.components.adapters.base import AdapterContext
from yolo_agent.components.adapters.runtime import AdapterRuntimePayload
from yolo_agent.components.adapters.distillation.yolo26_distillation import (
    YOLO26DistillationAdapter,
)
from yolo_agent.components.contracts import ComponentContract
from yolo_agent.core.command_spec import CommandSpec
from yolo_agent.core.experiment_graph import ExperimentNode, MetricEvidence
from yolo_agent.core.paired_experiment import build_paired_experiment_result
from yolo_agent.core.round_execution_plan import RoundExecutionPlan
from yolo_agent.core.run_context import RunContext


def _nodes(tmp_path: Path) -> tuple[ExperimentNode, ExperimentNode, Path, Path]:
    teacher = tmp_path / "yolo26s.pt"
    student = tmp_path / "yolo26n.pt"
    dataset = tmp_path / "coco.yaml"
    teacher.write_bytes(b"teacher")
    student.write_bytes(b"student")
    dataset.write_text("path: coco\n", encoding="utf-8")
    contract = ComponentContract(
        component_id="distillation.yolo26_teacher_student",
        display_name="Distillation",
        category="distillation",
        implementation_path=(
            "yolo_agent.components.adapters.distillation.yolo26_distillation"
        ),
        adapter_class="YOLO26DistillationAdapter",
        maturity="smoke_passed",
        fixed_imgsz_compatible=True,
    )
    payload = YOLO26DistillationAdapter().build_runtime_payload(
        AdapterContext(
            contract=contract,
            detector_family="yolo26",
            imgsz=640,
            workspace=tmp_path,
            options={
                "teacher": str(teacher),
                "student": str(student),
                "teacher_data": str(dataset),
                "student_data": str(dataset),
                "teacher_split": "train",
                "student_split": "train",
            },
        ),
        protocol_hash="protocol-1",
        base_command=[
            "yolo",
            "detect",
            "train",
            f"model={student}",
            f"data={dataset}",
            "imgsz=640",
        ],
        generated_config={},
    )
    payload_path = payload.write(tmp_path / "runtime" / "payload.yaml")
    command = CommandSpec.ultralytics_train(
        model=student,
        data=dataset,
        project=tmp_path / "runs",
        name="distillation",
        epochs=3,
        imgsz=640,
    ).model_copy(
        update={
            "metadata": {
                "adapter_runtime_payload_path": payload_path.as_posix(),
                "adapter_runtime_protocol_hash": "protocol-1",
                "dataset_manifest_hash": hashlib.sha256(dataset.read_bytes()).hexdigest(),
                "split": "train",
                "fidelity": "pilot_3",
                "seed_policy": "42",
                "adapter_runtime_entrypoint": (
                    "yolo_agent.adapters.ultralytics.runtime_entrypoint"
                ),
                "matched_pilot_required": True,
                "component_recipe_id": "yolo26n_distillation",
                "component_recipe_version": "v1.0.0",
                "paper_readiness_state": "asha_eligible",
                "paper_readiness_blockers": "[]",
            }
        }
    )
    candidate = ExperimentNode(
        node_id="distillation-candidate-node",
        candidate_config=CandidateConfig(
            candidate_id="distillation-candidate",
            base_model=str(student),
            scale="n",
            framework="ultralytics",
            components=["distillation.yolo26_teacher_student"],
            target_error_facts=[{"fact_type": "capacity_gap"}],
        ),
        data_version="coco",
        command_spec=command,
        command=command.display(),
    )
    control_command = CommandSpec.ultralytics_train(
        model=student,
        data=dataset,
        project=tmp_path / "runs",
        name="control",
        epochs=3,
        imgsz=640,
    ).model_copy(
        update={
            "metadata": {
                "matched_baseline_control": True,
                "baseline_protocol_hash": "protocol-1",
                "dataset_manifest_hash": hashlib.sha256(dataset.read_bytes()).hexdigest(),
                "split": "train",
                "fidelity": "pilot_3",
                "seed_policy": "42",
            }
        }
    )
    control = ExperimentNode(
        node_id="matched-control-node",
        candidate_config=CandidateConfig(
            candidate_id="matched_baseline_control",
            base_model=str(student),
            scale="n",
            framework="ultralytics",
        ),
        data_version="coco",
        command_spec=control_command,
        command=control_command.display(),
    )
    return candidate, control, teacher, dataset


def _register(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    fault: str | None,
) -> tuple[int, ASHAScheduler, RunContext]:
    candidate, control, teacher, dataset = _nodes(tmp_path)
    if fault == "missing_teacher":
        teacher.unlink()
    elif fault == "teacher_hash":
        teacher.write_bytes(b"replaced-teacher")
    elif fault == "dataset_hash":
        dataset.write_text("path: changed-coco\n", encoding="utf-8")
    elif fault == "split":
        _rewrite_student_split(candidate, "val")
    context = RunContext(
        run_id="distillation-asha",
        run_root=tmp_path / "run-root",
        task_path=tmp_path / "task.yaml",
        data_yaml=tmp_path / "data.yaml",
    )
    child = LoopOrchestrator(context)
    RoundExecutionPlan(
        run_id=context.run_id,
        round_id="round-1",
        deferred_nodes=[control, candidate],
    ).to_yaml(context.artifact_path("round_execution_plan.yaml"))
    monkeypatch.setattr(
        "yolo_agent.agents.auto_optimization_loop.ComponentQueueCertificationGate.evaluate",
        lambda *args, **kwargs: SimpleNamespace(
            allowed=True,
            blockers=[],
            report_path=None,
            report_hash="certified",
        ),
    )
    monkeypatch.setattr(
        "yolo_agent.agents.auto_optimization_loop.validate_certified_runtime_node",
        lambda node: [],
    )
    monkeypatch.setattr(
        "yolo_agent.agents.auto_optimization_loop.AutomaticRuntimeReadinessGate.evaluate_node",
        lambda self, node: SimpleNamespace(allowed=True),
    )
    scheduler = ASHAScheduler.create(context.run_id)
    registered = _register_guarded_pilot_trials(
        scheduler,
        child,
        [candidate],
    )
    return registered, scheduler, context


def test_distillation_registers_with_asha_when_protocol_is_bound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registered, scheduler, _ = _register(
        tmp_path,
        monkeypatch,
        fault=None,
    )
    assert registered == 1
    assert [trial.candidate_id for trial in scheduler.study.trials] == [
        "distillation-candidate"
    ]
    assert scheduler.study.trials[0].baseline_control_node is not None


def test_missing_teacher_is_kept_as_evidence_recovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registered, scheduler, context = _register(
        tmp_path,
        monkeypatch,
        fault="missing_teacher",
    )
    assert registered == 0
    assert len(scheduler.study.trials) == 1
    assert scheduler.study.trials[0].readiness_state == "pre_registered"
    assert scheduler.study.trials[0].status == "needs_evidence"
    assert scheduler.study.trials[0].pending_stage is None
    coverage = PaperCandidateCoverage.from_yaml(
        context.artifact_path("paper_candidate_coverage.yaml")
    )
    assert coverage.records[0].disposition == "evidence_recovery"
    assert any(
        reason.startswith("teacher_checkpoint_missing:")
        for reason in coverage.records[0].reason_codes
    )


@pytest.mark.parametrize(
    ("fault", "reason"),
    [
        ("teacher_hash", "teacher_checkpoint_sha256_mismatch"),
        ("dataset_hash", "distillation_dataset_hash_mismatch"),
        ("split", "distillation_protocol_invalid:"),
    ],
)
def test_invalid_distillation_binding_is_kept_as_blocked_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fault: str,
    reason: str,
) -> None:
    registered, scheduler, context = _register(
        tmp_path,
        monkeypatch,
        fault=fault,
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
    assert any(
        item.startswith(reason)
        for item in coverage.records[0].reason_codes
    )


def _rewrite_student_split(candidate: ExperimentNode, split: str) -> None:
    assert candidate.command_spec is not None
    payload_path = Path(
        candidate.command_spec.metadata["adapter_runtime_payload_path"]
    )
    payload = AdapterRuntimePayload.read(payload_path, verify_imports=False)
    reference = payload.loss_plugin[0]
    options = {**reference.options, "student_split": split}
    payload.model_copy(
        update={
            "loss_plugin": [reference.model_copy(update={"options": options})],
        }
    ).write(payload_path)


def _metric(name: str, value: float, *, baseline: bool) -> MetricEvidence:
    return MetricEvidence(
        run_id="paired-run",
        origin_run_id="paired-run",
        candidate_id="baseline" if baseline else "distillation-candidate",
        node_id="control" if baseline else "candidate",
        metric_name=name,
        value=value,
        verified=True,
        evidence_role="baseline_reference" if baseline else "current_observation",
        protocol_hash="protocol-1",
        dataset_manifest_sha256="dataset",
        subset_manifest_sha256="subset",
        seed=42,
        epochs=3,
        fidelity="pilot_3",
        batch_policy_hash="batch",
        ultralytics_version="9.0.0",
        imgsz=640,
        eval_protocol_hash="eval",
        split="runtime" if name in {"latency_ms", "model_size_mb"} else "val2017",
    )


def test_paired_map_waits_for_candidate_and_matched_baseline() -> None:
    candidate_only = [
        _metric("map50_95", 0.41, baseline=False),
        _metric("latency_ms", 15.0, baseline=False),
        _metric("model_size_mb", 5.2, baseline=False),
    ]
    incomplete = build_paired_experiment_result(
        run_id="paired-run",
        candidate_id="distillation-candidate",
        candidate_node_id="candidate",
        metric_records=candidate_only,
        error_facts=[],
    )
    assert incomplete.verified is False
    assert "map50_95" not in incomplete.metric_deltas

    complete = build_paired_experiment_result(
        run_id="paired-run",
        candidate_id="distillation-candidate",
        candidate_node_id="candidate",
        metric_records=[
            *candidate_only,
            _metric("map50_95", 0.39, baseline=True),
            _metric("latency_ms", 14.8, baseline=True),
            _metric("model_size_mb", 5.2, baseline=True),
        ],
        error_facts=[],
    )
    assert complete.verified is True
    assert complete.metric_deltas["map50_95"].paired_delta == pytest.approx(0.02)
