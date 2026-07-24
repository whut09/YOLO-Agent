"""Executor abstraction tests."""

from __future__ import annotations

import sys
import types
from pathlib import Path

from yolo_agent.agents.candidate_generator import CandidateConfig
from yolo_agent.core.evidence_store import EvidenceStore
from yolo_agent.core.executor import (
    BenchmarkImporter,
    CommandSpec,
    DryRunExecutor,
    ShellExecutor,
    UltralyticsExecutor,
    _fast_baseline_gate_applies,
)
from yolo_agent.core.experiment_graph import ExperimentNode


def _node() -> ExperimentNode:
    return ExperimentNode(
        node_id="node-baseline",
        candidate_config=CandidateConfig(
            candidate_id="baseline",
            base_model="yolo11n",
            scale="n",
            framework="ultralytics",
        ),
        data_version="dataset-v1",
        command="yolo train model=baseline.yaml",
    )


def test_dry_run_executor_does_not_execute_command() -> None:
    """DryRunExecutor should return a result without starting a process."""
    result = DryRunExecutor().execute(_node(), run_id="dry-run")

    assert result.status == "dry_run"
    assert result.return_code is None
    assert result.node_id == "node-baseline"
    assert result.candidate_id == "baseline"
    assert result.command.metadata["candidate_id"] == "baseline"


def test_fast_baseline_gate_does_not_block_matched_control() -> None:
    """ASHA controls must run even when their profile is named pilot."""
    node = _node()
    node.command_spec = CommandSpec(
        command_type="train",
        command="yolo",
        metadata={
            "training_budget_profile": "pilot",
            "matched_baseline_control": True,
        },
    )

    assert _fast_baseline_gate_applies("pilot", node) is False


def test_shell_executor_runs_only_when_explicitly_used() -> None:
    """ShellExecutor should execute explicit subprocess commands."""
    command = CommandSpec(command=sys.executable, args=["-c", "print('executor-ok')"])

    result = ShellExecutor().execute(_node(), run_id="shell-run", command=command)

    assert result.status == "completed"
    assert result.return_code == 0
    assert "executor-ok" in result.stdout
    assert result.duration_seconds is not None


def test_ultralytics_executor_skips_when_framework_unavailable(monkeypatch) -> None:
    """UltralyticsExecutor should skip when the integration is not available."""
    from yolo_agent.adapters.ultralytics.adapter import UltralyticsAdapter

    monkeypatch.setattr(UltralyticsAdapter, "is_available", lambda self: False)
    result = UltralyticsExecutor().execute(_node(), run_id="ultralytics-run")

    assert result.status == "skipped"
    assert "unverified" in result.message or "not installed" in result.message


def test_ultralytics_executor_prepares_dry_run_when_available(monkeypatch) -> None:
    """UltralyticsExecutor should generate model YAML artifacts without running."""
    from yolo_agent.adapters.ultralytics.adapter import UltralyticsAdapter

    fake_yaml = types.SimpleNamespace(
        output_path=Path("generated_models/baseline.yaml"),
        changes=["copy baseline template"],
        warnings=[],
    )

    adapter = UltralyticsAdapter()
    monkeypatch.setattr(adapter, "is_available", lambda: True)
    monkeypatch.setattr(adapter, "generate_model_yaml", lambda candidate, **kwargs: fake_yaml)
    monkeypatch.setattr(adapter, "build_train_command", lambda node, **kwargs: "yolo train model=generated_models/baseline.yaml")

    result = UltralyticsExecutor(adapter=adapter, try_forward=False).execute(_node(), run_id="ultra-dry")

    assert result.status == "dry_run"
    assert result.artifacts.get("model_yaml") == Path("generated_models/baseline.yaml")
    assert "prepared training artifacts" in result.message


def test_ultralytics_executor_runs_training_when_try_forward(monkeypatch) -> None:
    """UltralyticsExecutor should run the training command when try_forward is True."""
    import yolo_agent.core.executor as executor_mod
    from yolo_agent.adapters.ultralytics.adapter import UltralyticsAdapter

    fake_yaml = types.SimpleNamespace(
        output_path=Path("generated_models/baseline.yaml"),
        changes=[],
        warnings=[],
    )

    class FakeCompletedProcess:
        returncode = 0
        stdout = "train ok"
        stderr = ""

    adapter = UltralyticsAdapter()
    monkeypatch.setattr(adapter, "is_available", lambda: True)
    monkeypatch.setattr(adapter, "generate_model_yaml", lambda candidate, **kwargs: fake_yaml)
    monkeypatch.setattr(adapter, "build_train_command", lambda node, **kwargs: "yolo train model=generated_models/baseline.yaml")
    monkeypatch.setattr(executor_mod.subprocess, "run", lambda *args, **kwargs: FakeCompletedProcess())

    result = UltralyticsExecutor(adapter=adapter, try_forward=True).execute(_node(), run_id="ultra-forward")

    assert result.status == "completed"
    assert result.stdout == "train ok"


def test_execution_result_logs_to_evidence_store(tmp_path: Path) -> None:
    """ExecutionResult should persist command evidence and metrics."""
    store = EvidenceStore(tmp_path / "runs")
    result = DryRunExecutor().execute(_node(), run_id="dry-run")

    config_path = result.log_to_evidence_store(store)
    evidence = store.load_run("dry-run")

    assert config_path == tmp_path / "runs" / "dry-run" / "config.yaml"
    assert evidence.config["execution_result"]["status"] == "dry_run"
    assert evidence.metrics["execution_duration_seconds"] == 0.0


def test_benchmark_importer_writes_node_metric_evidence(tmp_path: Path) -> None:
    """BenchmarkImporter should persist run and node metric evidence."""
    metrics_path = tmp_path / "metrics.csv"
    metrics_path.write_text(
        "\n".join(
            [
                "candidate_id,node_id,dataset_version,split,metric_name,value,source,verified,validator,source_artifact,metric_schema_version,higher_is_better,confidence",
                "baseline,node-baseline,dataset-v1,val,map50,0.6,benchmark,true,official_eval,results.csv,1.0,true,0.99",
                "baseline,node-baseline,dataset-v1,val,recall,0.7,benchmark,true,official_eval,results.csv,1.0,true,0.98",
                "baseline,node-baseline,dataset-v1,val,precision,0.8,benchmark,false,draft_eval,results.csv,1.0,true,0.5",
                "",
            ]
        ),
        encoding="utf-8",
    )
    store = EvidenceStore(tmp_path / "runs")

    result = BenchmarkImporter(store).import_metrics("bench-run", metrics_path)
    evidence = store.load_run("bench-run")

    assert result.run_metrics == {"map50": 0.6, "recall": 0.7}
    assert result.metric_records_output_path == tmp_path / "runs" / "bench-run" / "metrics_by_node.jsonl"
    assert {record.metric_name: record.value for record in evidence.metric_records if record.verified} == {
        "map50": 0.6,
        "recall": 0.7,
    }
    assert len(evidence.metric_records) == 3
    assert evidence.metric_records[0].validator == "official_eval"
    assert evidence.metric_records[0].source_artifact == Path("results.csv")
    assert evidence.metric_records[0].metric_schema_version == "1.0"
    assert evidence.metric_records[0].higher_is_better is True
    assert evidence.metric_records[0].confidence == 0.99
    assert next(record for record in evidence.metric_records if record.metric_name == "precision").verified is False
