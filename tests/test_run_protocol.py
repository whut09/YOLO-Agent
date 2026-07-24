"""Run protocol identity tests."""

from __future__ import annotations

from pathlib import Path

from yolo_agent.adapters.ultralytics.training import UltralyticsTrainingConfig
from yolo_agent.agents.asha_scheduler import ASHAScheduler
from yolo_agent.core.experiment_graph import ExperimentPlan
from yolo_agent.core.run_context import RunContext
from yolo_agent.core.run_protocol import RunProtocolVersion, build_run_protocol_version


def _context(tmp_path: Path) -> RunContext:
    context = RunContext(
        run_id="run-1",
        run_root=tmp_path / "runs",
        task_path=tmp_path / "task.yaml",
        data_yaml=tmp_path / "data.yaml",
        dataset_version="coco2017",
        dataset_manifest_sha256="dataset-sha",
    )
    context.ensure_dirs()
    return context


def test_run_protocol_hash_covers_required_execution_identity(tmp_path: Path) -> None:
    context = _context(tmp_path)
    config = UltralyticsTrainingConfig(data=context.data_yaml, model="yolo26n.pt", budget_profile="pilot")

    protocol = build_run_protocol_version(
        model="yolo26n.pt",
        context=context,
        training_config=config,
        profile="pilot",
        seed=42,
        code_version="commit-1",
        ultralytics_version="9.0.0",
    )

    assert protocol.dataset_manifest_sha256 == "dataset-sha"
    assert protocol.subset_manifest_sha256
    assert protocol.imgsz == 640
    assert protocol.epochs == 10
    assert protocol.seed == 42
    assert protocol.batch_policy_hash
    assert protocol.eval_protocol_hash
    assert protocol.ultralytics_version == "9.0.0"
    assert protocol.code_version == "commit-1"
    assert len(protocol.protocol_hash) == 64

    changed_seed = protocol.model_copy(update={"seed": 43, "protocol_hash": ""})
    assert changed_seed.semantic_hash() != protocol.protocol_hash


def test_run_context_exposes_event_log_path(tmp_path: Path) -> None:
    """Run event logging should use one stable path beneath the run directory."""
    context = _context(tmp_path)

    assert context.events_path == context.run_dir / "events.jsonl"


def test_run_protocol_serializes_through_plan_and_asha(tmp_path: Path) -> None:
    protocol = RunProtocolVersion(
        model="yolo26n.pt",
        dataset_version="coco2017",
        dataset_manifest_sha256="dataset",
        subset_manifest_sha256="subset",
        imgsz=640,
        epochs=3,
        seed=42,
        batch_policy={"mode": "auto"},
        batch_policy_hash="batch",
        ultralytics_version="9.0.0",
        eval_protocol={"split": "val"},
        eval_protocol_hash="eval",
        code_version="commit",
        profile="pilot",
    )
    protocol_path = protocol.to_yaml(tmp_path / "run_protocol.yaml")
    plan = ExperimentPlan(plan_id="plan", run_protocol_hash=protocol.protocol_hash)
    plan.to_yaml(tmp_path / "plan.yaml")
    scheduler = ASHAScheduler.create("run-1")
    scheduler.study.run_protocol_hash = protocol.protocol_hash
    scheduler.study.to_yaml(tmp_path / "asha.yaml")

    assert RunProtocolVersion.from_yaml(protocol_path).protocol_hash == protocol.protocol_hash
    assert ExperimentPlan.from_yaml(tmp_path / "plan.yaml").run_protocol_hash == protocol.protocol_hash
    assert ASHAScheduler.create("other").study.run_protocol_hash is None
    assert scheduler.study.from_yaml(tmp_path / "asha.yaml").run_protocol_hash == protocol.protocol_hash
