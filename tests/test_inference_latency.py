"""Offline coverage for fixed-protocol inference latency evidence."""

from __future__ import annotations

from pathlib import Path

import pytest

from yolo_agent.adapters.ultralytics.inference_latency import (
    InferenceLatencyConfig,
    benchmark_checkpoint,
    requires_fixed_inference_latency,
    should_run_inference_latency_benchmark,
)
from yolo_agent.agents.candidate_generator import CandidateConfig
from yolo_agent.core.command_spec import CommandSpec
from yolo_agent.core.evidence_store import EvidenceStore
from yolo_agent.core.executor import _ensure_inference_latency_evidence
from yolo_agent.core.experiment_graph import ExperimentNode
from yolo_agent.core.paired_experiment import build_paired_experiment_result


def _node(candidate_id: str, node_id: str) -> ExperimentNode:
    return ExperimentNode(
        node_id=node_id,
        candidate_config=CandidateConfig(
            candidate_id=candidate_id,
            base_model="yolo26n.pt",
            scale="n",
            framework="ultralytics",
        ),
        data_version="coco2017",
        seed=42,
    )


def _spec() -> CommandSpec:
    return CommandSpec(
        command="yolo",
        argv=["yolo", "detect", "train", "device=0", "imgsz=640", "epochs=3"],
        metadata={
            "run_protocol_hash": "protocol-pilot-3",
            "dataset_manifest_sha256": "dataset-sha",
            "subset_manifest_sha256": "subset-sha",
            "eval_protocol_hash": "eval-sha",
            "batch_policy_hash": "batch-sha",
            "ultralytics_version": "test-version",
            "round_stage": "pilot_3",
            "epochs": 3,
        },
    )


def test_fixed_latency_benchmark_uses_mock_model(monkeypatch, tmp_path: Path) -> None:
    import yolo_agent.adapters.ultralytics.inference_latency as latency

    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"weights")
    calls: list[dict[str, object]] = []

    class FakeModel:
        def predict(self, **kwargs: object) -> None:
            calls.append(kwargs)

    monkeypatch.setattr(latency, "_load_yolo", lambda _: FakeModel())
    monkeypatch.setattr(latency, "_synchronize_device", lambda _: None)

    result = benchmark_checkpoint(
        checkpoint,
        device="0",
        config=InferenceLatencyConfig(enabled=True, warmup_runs=2, timed_runs=4),
    )

    assert result.status == "completed"
    assert result.latency_ms is not None and result.latency_ms >= 0
    assert result.throughput is not None and result.throughput > 0
    assert len(calls) == 6
    assert {call["imgsz"] for call in calls} == {640}
    assert {call["device"] for call in calls} == {"0"}


def test_latency_evidence_forms_verified_matched_pair(monkeypatch, tmp_path: Path) -> None:
    from yolo_agent.adapters.ultralytics.inference_latency import InferenceLatencyResult

    store = EvidenceStore(tmp_path / "runs")
    config = InferenceLatencyConfig(enabled=True, warmup_runs=0, timed_runs=2)
    spec = _spec()
    baseline_spec = spec.model_copy(update={"metadata": {**spec.metadata, "matched_baseline_control": True}})
    baseline = _node("baseline", "node_baseline")
    candidate = _node("candidate", "node_candidate")
    baseline_dir = tmp_path / "baseline"
    candidate_dir = tmp_path / "candidate"
    for directory in (baseline_dir, candidate_dir):
        (directory / "weights").mkdir(parents=True)
        (directory / "weights" / "best.pt").write_bytes(b"weights")

    def fake_benchmark(checkpoint: Path, *, device: str, config: InferenceLatencyConfig) -> InferenceLatencyResult:
        latency_ms = 10.0 if checkpoint.parent.parent == baseline_dir else 10.4
        return InferenceLatencyResult(
            status="completed",
            checkpoint=checkpoint,
            device=device,
            imgsz=config.imgsz,
            warmup_runs=config.warmup_runs,
            timed_runs=config.timed_runs,
            latency_ms=latency_ms,
            throughput=1000.0 / latency_ms,
        )

    monkeypatch.setattr("yolo_agent.core.executor._coco_evidence_identity", lambda spec, node: {
        "protocol_hash": "protocol-pilot-3",
        "dataset_manifest_sha256": "dataset-sha",
        "subset_manifest_sha256": "subset-sha",
        "eval_protocol_hash": "eval-sha",
        "seed": node.seed,
        "fidelity": "pilot_3",
        "epochs": 3,
        "batch_policy_hash": "batch-sha",
        "ultralytics_version": "test-version",
        "imgsz": 640,
    })
    _ensure_inference_latency_evidence(
        evidence_store=store, node=baseline, run_id="run-1", spec=baseline_spec,
        actual_run_dir=baseline_dir, config=config, device="0", benchmark=fake_benchmark,
    )
    _ensure_inference_latency_evidence(
        evidence_store=store, node=candidate, run_id="run-1", spec=spec,
        actual_run_dir=candidate_dir, config=config, device="0", benchmark=fake_benchmark,
    )
    identity = {
        "protocol_hash": "protocol-pilot-3",
        "dataset_manifest_sha256": "dataset-sha",
        "subset_manifest_sha256": "subset-sha",
        "eval_protocol_hash": "eval-sha",
        "seed": 42,
        "fidelity": "pilot_3",
        "epochs": 3,
        "batch_policy_hash": "batch-sha",
        "ultralytics_version": "test-version",
        "imgsz": 640,
    }
    for node, role, map_value, size in [
        (baseline, "baseline_reference", 0.30, 5.0),
        (candidate, "current_observation", 0.31, 5.1),
    ]:
        store.upsert_candidate_metrics(
            run_id="run-1",
            candidate_id=node.candidate_config.candidate_id,
            node_id=node.node_id,
            metrics={"map50_95": map_value, "model_size_mb": size},
            dataset_version=node.data_version,
            split="val",
            source="test",
            verified=True,
            validator="test",
            evidence_role=role,
            **identity,
        )

    evidence = store.load_run("run-1")
    paired = build_paired_experiment_result(
        run_id="run-1",
        candidate_id="candidate",
        candidate_node_id="node_candidate",
        metric_records=evidence.metric_records,
        error_facts=[],
    )
    assert paired.verified is True
    assert paired.latency_delta is not None
    assert paired.latency_delta.paired_delta == pytest.approx(0.4)


def test_latency_protocol_only_applies_to_staged_candidates() -> None:
    config = InferenceLatencyConfig(enabled=True)
    assert should_run_inference_latency_benchmark("pilot", config) is True
    assert should_run_inference_latency_benchmark("debug", config) is False
    assert requires_fixed_inference_latency("pilot", "pilot_3") is True
    assert requires_fixed_inference_latency("debug", None) is False
