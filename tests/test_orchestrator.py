"""Run orchestrator tests."""

from __future__ import annotations

from pathlib import Path

import yaml

from yolo_agent.agents.orchestrator import LoopOrchestrator
from yolo_agent.cli import main
from yolo_agent.core.artifact_manifest import ArtifactManifest, sha256_file
from yolo_agent.core.dataset_versioning import DatasetVersionManifest
from yolo_agent.core.decision_ledger import DecisionLedger
from yolo_agent.core.execution_queue import ExecutionQueue
from yolo_agent.core.event_log import EventLog
from yolo_agent.core.loop_state import LoopState
from yolo_agent.core.run_context import RunContext
from yolo_agent.core.run_lineage import RunLineageStore


def _make_task(path: Path) -> Path:
    task_path = path / "task.yaml"
    task_path.write_text(
        yaml.safe_dump(
            {
                "task_type": "detect",
                "scene": "infrared_small_target",
                "class_names": ["target"],
                "primary_metric": {"name": "recall"},
                "secondary_metrics": [{"name": "map50_95"}, {"name": "latency_ms", "goal": "minimize"}],
                "max_latency_ms": 30,
                "max_model_size_mb": 20,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return task_path


def _make_dataset(root: Path) -> Path:
    image_dir = root / "images" / "train"
    label_dir = root / "labels" / "train"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    (image_dir / "img1.jpg").write_bytes(b"image-1")
    (image_dir / "img2.jpg").write_bytes(b"image-2")
    (label_dir / "img1.txt").write_text("0 0.5 0.5 0.03 0.03\n", encoding="utf-8")
    (label_dir / "img2.txt").write_text("", encoding="utf-8")
    data_yaml = root / "data.yaml"
    data_yaml.write_text(
        "\n".join(
            [
                "path: .",
                "scene: infrared_small_target",
                "train: images/train",
                "names:",
                "  - target",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return data_yaml


def _make_errors(path: Path) -> Path:
    errors_path = path / "errors.yaml"
    errors_path.write_text(
        yaml.safe_dump(
            {
                "errors": [
                    {"error_type": "small_object_miss", "count": 4, "severity": "high"},
                    {"error_type": "background_confusion", "count": 2, "severity": "medium"},
                ]
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return errors_path


def _make_metrics(path: Path) -> Path:
    metrics_path = path / "metrics.csv"
    metrics_path.write_text(
        (
            "metric,value\n"
            "map50,0.6\n"
            "mAP_small,0.4\n"
            "precision,0.8\n"
            "recall,0.7\n"
            "false_negative_count,2\n"
            "latency_ms,12\n"
            "model_size_mb,5\n"
        ),
        encoding="utf-8",
    )
    return metrics_path


def _make_node_metrics(path: Path) -> Path:
    metrics_path = path / "node_metrics.csv"
    metrics_path.write_text(
        "\n".join(
            [
                "candidate_id,node_id,dataset_version,split,metric_name,value,source",
                "baseline,node_baseline,dataset-v1,val,map50,0.6,benchmark",
                "baseline,node_baseline,dataset-v1,val,recall,0.7,benchmark",
                "baseline,node_baseline,dataset-v1,val,latency_ms,12,benchmark",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return metrics_path


def _make_reordered_loop_policy(path: Path) -> Path:
    policy_path = path / "loop_policy.yaml"
    data = yaml.safe_load(Path("configs/loop_policy.yaml").read_text(encoding="utf-8"))
    stages = data["stages"]
    profile_index = next(index for index, stage in enumerate(stages) if stage["id"] == "profile_data")
    advise_index = next(index for index, stage in enumerate(stages) if stage["id"] == "advise_labels")
    stages[profile_index], stages[advise_index] = stages[advise_index], stages[profile_index]
    policy_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return policy_path


def test_loop_orchestrator_blocks_when_detection_errors_are_missing(tmp_path: Path) -> None:
    """Auto loop should stop at diagnose_errors when required error evidence is absent."""
    task_path = _make_task(tmp_path)
    data_yaml = _make_dataset(tmp_path / "dataset")
    orchestrator = LoopOrchestrator.initialize(
        run_id="missing-errors",
        task_path=task_path,
        data_yaml=data_yaml,
        run_root=tmp_path / "runs",
    )

    results = orchestrator.run_until_blocked()

    assert [result.stage for result in results] == ["profile_data", "advise_labels", "diagnose_errors"]
    assert results[-1].status == "blocked"
    assert (orchestrator.context.artifact_path("dataset_report.json")).exists()
    state = LoopState.from_yaml(orchestrator.context.run_dir / "loop_state.yaml")
    assert state.stages["diagnose_errors"].status == "blocked"
    assert state.dataset_version == "unversioned"
    assert state.task_spec == task_path
    assert "profile_data" in state.completed
    assert "advise_labels" in state.completed
    assert "missing_detection_errors" in state.blocked
    assert "dataset_report" in state.artifacts
    manifest = ArtifactManifest(orchestrator.context.artifact_path("artifact_manifest.jsonl")).read()
    dataset_entry = next(record for record in manifest if record.name == "dataset_report")
    assert dataset_entry.producer_stage == "profile_data"
    assert dataset_entry.sha256 == sha256_file(orchestrator.context.artifact_path("dataset_report.json"))
    assert dataset_entry.verify() is True
    events = EventLog(orchestrator.context.run_dir / "events.jsonl").read()
    assert [event.event_type for event in events if event.stage == "diagnose_errors"][-1] == "contract_blocked"
    assert events[-1].details["missing_required"] == ["detection_errors"]


def test_loop_orchestrator_rejects_unmanifested_artifact_dependency(tmp_path: Path) -> None:
    """Stage contracts should not trust same-named artifacts without manifest evidence."""
    task_path = _make_task(tmp_path)
    data_yaml = _make_dataset(tmp_path / "dataset")
    errors_path = _make_errors(tmp_path)
    orchestrator = LoopOrchestrator.initialize(
        run_id="stale-artifact",
        task_path=task_path,
        data_yaml=data_yaml,
        run_root=tmp_path / "runs",
        detection_errors_path=errors_path,
    )
    dataset_report_path = orchestrator.context.artifact_path("dataset_report.json")
    dataset_report_path.write_text(
        yaml.safe_dump(
            {
                "data_yaml": str(data_yaml),
                "dataset_root": str(data_yaml.parent),
                "scene": "generic",
                "image_count": 1,
                "label_count": 0,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    result = orchestrator.run_stage("diagnose_errors")

    assert result.status == "blocked"
    events = EventLog(orchestrator.context.run_dir / "events.jsonl").read()
    assert events[-1].details["invalid_artifacts"] == ["dataset_report: missing manifest entry"]


def test_loop_init_records_dataset_manifest_in_run_context(tmp_path: Path) -> None:
    """Loop init should bind the run to a hashed dataset manifest."""
    task_path = _make_task(tmp_path)
    data_yaml = _make_dataset(tmp_path / "dataset")
    orchestrator = LoopOrchestrator.initialize(
        run_id="manifest-run",
        task_path=task_path,
        data_yaml=data_yaml,
        run_root=tmp_path / "runs",
        dataset_version="dataset-v1",
    )

    context = RunContext.from_run_dir(orchestrator.context.run_dir)

    assert context.dataset_root == data_yaml.parent
    assert context.dataset_version_store_path == orchestrator.context.run_dir / "dataset_versions"
    assert context.dataset_manifest_path == orchestrator.context.run_dir / "dataset_versions" / "dataset-v1" / "manifest.json"
    assert context.dataset_manifest_path.is_file()
    assert context.dataset_manifest_sha256 == sha256_file(context.dataset_manifest_path)

    manifest = DatasetVersionManifest.from_json(context.dataset_manifest_path)
    manifest_files = {record.path for record in manifest.files}
    assert manifest.version == "dataset-v1"
    assert manifest.source_root == data_yaml.parent
    assert {"data.yaml", "images/train/img1.jpg", "labels/train/img1.txt"}.issubset(manifest_files)
    assert all(not path.startswith("dataset_versions/") for path in manifest_files)

    artifact_records = ArtifactManifest(context.artifact_path("artifact_manifest.jsonl")).read()
    dataset_entry = next(record for record in artifact_records if record.name == "dataset_manifest")
    assert dataset_entry.sha256 == context.dataset_manifest_sha256
    assert dataset_entry.verify() is True


def test_loop_orchestrator_uses_policy_stage_order(tmp_path: Path) -> None:
    """Auto loop should use stage order from loop_policy.yaml, not a code constant."""
    task_path = _make_task(tmp_path)
    data_yaml = _make_dataset(tmp_path / "dataset")
    policy_path = _make_reordered_loop_policy(tmp_path)
    orchestrator = LoopOrchestrator.initialize(
        run_id="policy-order",
        task_path=task_path,
        data_yaml=data_yaml,
        run_root=tmp_path / "runs",
        loop_policy_path=policy_path,
    )

    results = orchestrator.run_until_blocked()

    assert orchestrator.state.stage_order[:3] == ["init", "advise_labels", "profile_data"]
    assert [result.stage for result in results[:2]] == ["advise_labels", "profile_data"]
    assert results[-1].stage == "diagnose_errors"
    assert results[-1].status == "blocked"


def test_loop_orchestrator_runs_harness_until_metrics_import_block(tmp_path: Path) -> None:
    """With errors available, the loop should produce plans and stop before missing metrics."""
    task_path = _make_task(tmp_path)
    data_yaml = _make_dataset(tmp_path / "dataset")
    errors_path = _make_errors(tmp_path)
    orchestrator = LoopOrchestrator.initialize(
        run_id="loop-run",
        task_path=task_path,
        data_yaml=data_yaml,
        run_root=tmp_path / "runs",
        detection_errors_path=errors_path,
    )

    results = orchestrator.run_until_blocked()

    assert results[-1].stage == "import_metrics"
    assert results[-1].status == "blocked"
    assert (orchestrator.context.artifact_path("loop_diagnosis.json")).exists()
    assert (orchestrator.context.artifact_path("loop_plan.yaml")).exists()
    assert (orchestrator.context.artifact_path("policy_evaluation.yaml")).exists()
    assert (orchestrator.context.artifact_path("decision_ledger.jsonl")).exists()
    assert (orchestrator.context.run_dir / "plan.yaml").exists()
    assert (orchestrator.context.run_dir / "ablation_plan.yaml").exists()
    assert (orchestrator.context.artifact_path("smoke_result.json")).exists()
    assert (orchestrator.context.artifact_path("evidence_status.json")).exists()
    events = EventLog(orchestrator.context.run_dir / "events.jsonl").read()
    assert any(event.event_type == "stage_completed" and event.stage == "smoke" for event in events)
    assert events[-1].event_type == "contract_blocked"
    assert events[-1].stage == "import_metrics"
    ledger = DecisionLedger(orchestrator.context.artifact_path("decision_ledger.jsonl")).read()
    assert ledger
    assert all(record.proposal.get("policy_id") == record.policy_id for record in ledger)
    assert all(record.decision for record in ledger)


def test_loop_decision_ledger_records_policy_outcomes(tmp_path: Path) -> None:
    """evaluate_policies should write accepted, rejected, and needs-evidence decisions."""
    task_path = _make_task(tmp_path)
    data_yaml = _make_dataset(tmp_path / "dataset")
    orchestrator = LoopOrchestrator.initialize(
        run_id="ledger-run",
        task_path=task_path,
        data_yaml=data_yaml,
        run_root=tmp_path / "runs",
    )
    loop_plan_path = orchestrator.context.artifact_path("loop_plan.yaml")
    loop_plan_path.write_text(
        yaml.safe_dump(
            {
                "candidate_policies": [
                    {
                        "policy_id": "accepted_nwd",
                        "base_model": "yolo11n",
                        "scale": "n",
                        "framework": "ultralytics",
                        "components": ["loss.bbox.nwd"],
                    },
                    {
                        "policy_id": "rejected_latency",
                        "base_model": "yolo11n",
                        "scale": "n",
                        "framework": "ultralytics",
                        "constraints": [{"name": "estimated_latency_ms", "value": 45}],
                    },
                    {
                        "policy_id": "needs_recall",
                        "base_model": "yolo11n",
                        "scale": "n",
                        "framework": "ultralytics",
                        "components": ["assigner.stal"],
                        "evidence_required": ["recall"],
                    },
                ]
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    orchestrator.evidence_store.log_artifact_manifest(
        run_id="ledger-run",
        name="loop_plan",
        artifact_path=loop_plan_path,
        producer_stage="generate_loop_plan",
    )

    result = orchestrator.run_stage("evaluate_policies")

    assert result.status == "completed"
    ledger_path = orchestrator.context.artifact_path("decision_ledger.jsonl")
    records = {record.policy_id: record for record in DecisionLedger(ledger_path).read()}
    assert records["accepted_nwd"].decision == "accepted"
    assert records["accepted_nwd"].created_candidate_id == "accepted_nwd"
    assert records["accepted_nwd"].created_node_id == "node_accepted_nwd"
    assert records["accepted_nwd"].candidate_config is not None
    assert records["accepted_nwd"].experiment_node is not None
    assert records["rejected_latency"].decision == "rejected"
    assert records["rejected_latency"].deployment_constraints == [{"name": "estimated_latency_ms", "value": 45, "hard": True}]
    assert records["rejected_latency"].blocked_by
    assert records["needs_recall"].decision == "needs_evidence"
    assert records["needs_recall"].missing_evidence == ["recall"]
    assert "recall" in records["needs_recall"].blocked_by


def test_loop_cli_init_and_run_stage(tmp_path: Path) -> None:
    """Loop CLI should initialize a run and execute one state-machine stage."""
    task_path = _make_task(tmp_path)
    data_yaml = _make_dataset(tmp_path / "dataset")
    run_root = tmp_path / "runs"

    assert main(
        [
            "loop",
            "init",
            "--run-id",
            "cli-run",
            "--task",
            str(task_path),
            "--data",
            str(data_yaml),
            "--run-root",
            str(run_root),
        ]
    ) == 0
    assert main(["loop", "run-stage", "--run", str(run_root / "cli-run"), "--stage", "profile_data"]) == 0
    assert (run_root / "cli-run" / "artifacts" / "dataset_report.json").exists()


def test_loop_cli_resume_retries_blocked_stage(tmp_path: Path) -> None:
    """Loop resume should reset the first blocked stage and continue when evidence appears."""
    task_path = _make_task(tmp_path)
    data_yaml = _make_dataset(tmp_path / "dataset")
    errors_path = _make_errors(tmp_path)
    run_root = tmp_path / "runs"
    orchestrator = LoopOrchestrator.initialize(
        run_id="resume-run",
        task_path=task_path,
        data_yaml=data_yaml,
        run_root=run_root,
    )
    results = orchestrator.run_until_blocked()
    assert results[-1].stage == "diagnose_errors"
    assert results[-1].status == "blocked"

    orchestrator.context.detection_errors_path = errors_path
    orchestrator.context.to_yaml()

    assert main(["loop", "--run", str(run_root / "resume-run"), "--resume"]) == 0

    state = LoopState.from_yaml(run_root / "resume-run" / "loop_state.yaml")
    assert state.stages["diagnose_errors"].status == "completed"
    assert state.stages["import_metrics"].status == "blocked"
    assert "missing_metrics" in state.blocked
    assert "loop_diagnosis" in state.artifacts


def test_loop_cli_workflow_commands_run_without_training(tmp_path: Path) -> None:
    """Dedicated loop CLI commands should drive the harness in explicit phases."""
    task_path = _make_task(tmp_path)
    data_yaml = _make_dataset(tmp_path / "dataset")
    errors_path = _make_errors(tmp_path)
    metrics_path = _make_metrics(tmp_path)
    run_root = tmp_path / "runs"
    run_dir = run_root / "phase-run"

    assert main(
        [
            "loop",
            "init",
            "--run-id",
            "phase-run",
            "--task",
            str(task_path),
            "--data",
            str(data_yaml),
            "--run-root",
            str(run_root),
        ]
    ) == 0
    assert main(["loop", "diagnose", "--run", str(run_dir), "--errors", str(errors_path)]) == 0
    assert main(["loop", "plan", "--run", str(run_dir)]) == 0
    assert main(["loop", "enqueue", "--run", str(run_dir)]) == 0
    assert main(["loop", "execute", "--run", str(run_dir), "--executor", "dry-run"]) == 0
    assert main(["loop", "smoke", "--run", str(run_dir)]) == 0
    assert main(["loop", "ingest-metrics", "--run", str(run_dir), "--metrics", str(metrics_path)]) == 0
    assert main(["loop", "next", "--run", str(run_dir)]) == 0

    assert (run_dir / "artifacts" / "loop_diagnosis.json").exists()
    assert (run_dir / "artifacts" / "policy_evaluation.yaml").exists()
    assert (run_dir / "execution_queue.yaml").exists()
    assert (run_dir / "artifacts" / "execution_results").exists()
    assert (run_dir / "artifacts" / "smoke_result.json").exists()
    assert (run_dir / "artifacts" / "metrics_import.json").exists()
    assert (run_dir / "report.md").exists()
    queue = ExecutionQueue.from_yaml(run_dir / "execution_queue.yaml")
    assert queue.items
    assert queue.counts()["completed"] == len(queue.items)
    assert all(item.last_result is not None and item.last_result.status == "dry_run" for item in queue.items)
    next_round = yaml.safe_load((run_dir / "artifacts" / "next_round.yaml").read_text(encoding="utf-8"))
    assert next_round["parent_run_id"] == "phase-run"
    assert next_round["parent_best_candidate"]["candidate_id"] == "phase-run"
    assert next_round["parent_best_candidate"]["metric_name"] == "map50"
    assert next_round["recommended_stage"] == "generate_loop_plan"
    assert next_round["stop_reason"] == "unresolved_diagnoses"
    assert "present_now" in next_round["evidence_delta"]
    state = LoopState.from_yaml(run_dir / "loop_state.yaml")
    assert state.stages["next_round"].status == "completed"


def test_loop_cli_fork_next_materializes_child_run(tmp_path: Path) -> None:
    """fork-next should turn next_round.yaml into a fresh child run."""
    task_path = _make_task(tmp_path)
    data_yaml = _make_dataset(tmp_path / "dataset")
    errors_path = _make_errors(tmp_path)
    run_root = tmp_path / "runs"
    parent_dir = run_root / "parent-run"
    child_dir = run_root / "child-run"

    assert main(
        [
            "loop",
            "auto",
            "--run-id",
            "parent-run",
            "--task",
            str(task_path),
            "--data",
            str(data_yaml),
            "--run-root",
            str(run_root),
            "--errors",
            str(errors_path),
            "--dataset-version",
            "dataset-v1",
        ]
    ) == 0
    assert main(["loop", "next", "--run", str(parent_dir)]) == 0

    parent_context = RunContext.from_run_dir(parent_dir)

    assert main(["loop", "fork-next", "--run", str(parent_dir), "--new-run-id", "child-run"]) == 0

    child_context = RunContext.from_run_dir(child_dir)
    child_state = LoopState.from_yaml(child_dir / "loop_state.yaml")

    assert child_context.run_id == "child-run"
    assert child_context.task_path == task_path
    assert child_context.data_yaml == data_yaml
    assert child_context.dataset_version == "dataset-v1"
    assert child_context.dataset_manifest_sha256 == parent_context.dataset_manifest_sha256
    assert child_context.metadata["parent_run_id"] == "parent-run"
    assert child_context.metadata["parent_run_dir"] == parent_dir.as_posix()
    assert "latency_ms" in child_context.metadata["inherited_missing_evidence"]
    assert "map50" in child_context.metadata["inherited_missing_evidence"]
    assert child_context.metadata["recommended_stage"] == "import_metrics"
    assert child_context.metadata["parent_stop_reason"] == "missing_evidence"
    assert isinstance(child_context.metadata["inherited_unresolved_diagnoses"], list)
    assert child_context.metadata["parent_evidence_delta"]["current_missing"]
    assert (child_dir / "artifacts" / "parent_next_round.yaml").exists()
    assert (child_dir / "artifacts" / "fork_context.yaml").exists()
    assert child_state.stages["init"].status == "completed"
    assert child_state.stages["profile_data"].status == "pending"

    graph = RunLineageStore(run_root).graph()
    assert graph.parent_of("child-run") == "parent-run"
    assert graph.inherited_dataset_manifest_sha("child-run") == parent_context.dataset_manifest_sha256
    assert "child-run" in graph.children_of("parent-run")
    initial_delta = graph.evidence_delta("child-run")
    assert "map50" in initial_delta["current_missing"]
    assert "map50" not in initial_delta["resolved"]

    metrics_path = _make_metrics(tmp_path)
    assert main(["loop", "diagnose", "--run", str(child_dir), "--errors", str(errors_path)]) == 0
    assert main(["loop", "plan", "--run", str(child_dir)]) == 0
    assert main(["loop", "smoke", "--run", str(child_dir)]) == 0
    assert main(["loop", "ingest-metrics", "--run", str(child_dir), "--metrics", str(metrics_path)]) == 0

    updated_graph = RunLineageStore(run_root).graph()
    delta = updated_graph.evidence_delta("child-run")
    assert "map50" in delta["resolved"]
    assert "recall" in delta["resolved"]
    assert "latency_ms" in delta["resolved"]
    best = updated_graph.best_trusted_run()
    assert best is not None
    assert best.run_id == "child-run"
    assert best.best_metric_name == "map50"
    assert main(["loop", "lineage", "--run-root", str(run_root), "--run", "child-run"]) == 0
    assert main(["loop", "lineage", "--run-root", str(run_root), "--best"]) == 0


def test_loop_ingest_metrics_persists_candidate_records(tmp_path: Path) -> None:
    """Loop metrics ingest should persist candidate/node-level metric evidence."""
    task_path = _make_task(tmp_path)
    data_yaml = _make_dataset(tmp_path / "dataset")
    errors_path = _make_errors(tmp_path)
    metrics_path = _make_node_metrics(tmp_path)
    run_root = tmp_path / "runs"
    run_dir = run_root / "node-metrics-run"

    assert main(
        [
            "loop",
            "init",
            "--run-id",
            "node-metrics-run",
            "--task",
            str(task_path),
            "--data",
            str(data_yaml),
            "--run-root",
            str(run_root),
            "--errors",
            str(errors_path),
        ]
    ) == 0
    assert main(["loop", "diagnose", "--run", str(run_dir)]) == 0
    assert main(["loop", "plan", "--run", str(run_dir)]) == 0
    assert main(["loop", "smoke", "--run", str(run_dir)]) == 0
    assert main(["loop", "ingest-metrics", "--run", str(run_dir), "--metrics", str(metrics_path)]) == 0

    evidence = LoopOrchestrator.from_run_dir(run_dir).evidence_store.load_run("node-metrics-run")
    assert (run_dir / "metrics_by_node.jsonl").exists()
    metric_values = {record.metric_name: record.value for record in evidence.metric_records}
    assert {
        name: metric_values[name]
        for name in ["map50", "recall", "latency_ms"]
    } == {
        "map50": 0.6,
        "recall": 0.7,
        "latency_ms": 12,
    }
    assert {
        name: metric_values[name]
        for name in ["smoke_passed", "yaml_generated", "ultralytics_imported", "forward_checked"]
    } == {
        "smoke_passed": False,
        "yaml_generated": True,
        "ultralytics_imported": False,
        "forward_checked": False,
    }
    map_record = next(record for record in evidence.metric_records if record.metric_name == "map50")
    smoke_record = next(record for record in evidence.metric_records if record.metric_name == "smoke_passed")
    assert map_record.candidate_id == "baseline"
    assert map_record.node_id == "node_baseline"
    assert smoke_record.split == "guard"
    assert smoke_record.validator == "SmokeRunner"


def test_loop_auto_can_initialize_from_task_and_data(tmp_path: Path) -> None:
    """loop auto should initialize a run when task/data are provided."""
    task_path = _make_task(tmp_path)
    data_yaml = _make_dataset(tmp_path / "dataset")
    run_root = tmp_path / "runs"

    assert main(
        [
            "loop",
            "auto",
            "--run-id",
            "auto-run",
            "--task",
            str(task_path),
            "--data",
            str(data_yaml),
            "--run-root",
            str(run_root),
        ]
    ) == 0

    run_dir = run_root / "auto-run"
    assert (run_dir / "run_context.yaml").exists()
    state = LoopState.from_yaml(run_dir / "loop_state.yaml")
    assert state.stages["diagnose_errors"].status == "blocked"
    assert "missing_detection_errors" in state.blocked
