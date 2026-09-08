from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from yolo_agent.agents.auto_optimization_loop import (
    _activate_hard_negative_bootstrap_candidates,
    _ensure_hard_negative_bootstrap,
    _execute_hard_negative_bootstrap_queue,
    _hard_negative_replay_needs_bootstrap,
    _recover_hard_negative_bootstrap_resource_waits,
)
from yolo_agent.agents.candidate_generator import CandidateConfig
from yolo_agent.agents.loop_policy_evaluator import (
    LoopPolicyEvaluation,
    LoopPolicyEvaluationReport,
)
from yolo_agent.agents.orchestrator import LoopOrchestrator
from yolo_agent.components.adapters.data_pipeline.hard_negative import (
    HardNegativeEvidenceBootstrap,
    HardNegativeManifest,
)
from yolo_agent.components.adapters.data_pipeline.hard_negative_evidence import (
    TrainHardNegativePredictionBatch,
    TrainSampleIndex,
    train_sample_index_from_yolo_data,
)
from yolo_agent.core.command_spec import CommandSpec
from yolo_agent.core.execution_queue import (
    ExecutionQueue,
    ExecutionQueueItem,
    ExecutionQueueStore,
)
from yolo_agent.core.experiment_graph import ExperimentNode
from yolo_agent.core.run_context import RunContext
from yolo_agent.core.round_execution_plan import (
    RoundExecutionPlan,
    build_hard_negative_bootstrap_nodes,
)
from yolo_agent.agents.loop_io import read_yaml, write_yaml
from yolo_agent.tools.hard_negative_bootstrap import (
    execute_hard_negative_bootstrap_stage,
)


def _node(
    tmp_path: Path,
    candidate_id: str,
    *,
    components: list[str] | None = None,
    matched_control: bool = False,
) -> ExperimentNode:
    metadata = {
        "matched_baseline_control": matched_control,
        "matched_pilot_required": not matched_control,
        "dataset_manifest_hash": "dataset-train-v1",
        "dataset_manifest_sha256": "dataset-train-v1",
        "split": "val2017",
        "fidelity": "pilot_3",
        "seed_policy": "42",
        "protocol_hash": "protocol-train-v1",
        "run_protocol_hash": "protocol-train-v1",
        "paper_readiness_state": "asha_eligible",
        "paper_readiness_blockers": "[]",
    }
    command = CommandSpec.ultralytics_train(
        model="yolo26n.pt",
        data=tmp_path / "data.yaml",
        project=tmp_path / "ultralytics",
        name=candidate_id,
        epochs=3,
        imgsz=640,
        batch=2,
        metadata=metadata,
    )
    return ExperimentNode(
        node_id=f"node_{candidate_id}",
        candidate_config=CandidateConfig(
            candidate_id=candidate_id,
            base_model="yolo26n.pt",
            scale="n",
            framework="ultralytics",
            components=list(components or []),
            action_domain="data",
            action_id=candidate_id,
            train_overrides={},
            target_error_facts=[
                {"fact_type": "background_false_positive_class", "subject": "person"}
            ],
        ),
        data_version="dataset-train-v1",
        command_spec=command,
    )


def _context(tmp_path: Path) -> RunContext:
    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text(
        yaml.safe_dump(
            {
                "path": str(tmp_path),
                "train": "images/train",
                "val": "images/val",
                "names": ["person"],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return RunContext(
        run_id="hard-negative-bootstrap-r1",
        run_root=tmp_path / "runs",
        task_path=tmp_path / "task.yaml",
        data_yaml=data_yaml,
        dataset_manifest_sha256="dataset-train-v1",
    )


def _assets(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    (tmp_path / "images" / "train").mkdir(parents=True)
    (tmp_path / "images" / "val").mkdir(parents=True)
    (tmp_path / "images" / "train" / "train-a.jpg").write_bytes(b"train-a")
    (tmp_path / "images" / "train" / "train-b.jpg").write_bytes(b"train-b")
    (tmp_path / "images" / "val" / "val-a.jpg").write_bytes(b"val-a")
    index_path = tmp_path / "train-index.json"
    train_sample_index_from_yolo_data(
        tmp_path / "data.yaml",
        dataset_manifest_hash="dataset-train-v1",
        output_path=index_path,
    )
    checkpoint = tmp_path / "baseline.pt"
    checkpoint.write_bytes(b"frozen-baseline-checkpoint")
    prediction_path = tmp_path / "train-predictions.json"
    batch = TrainHardNegativePredictionBatch(
        dataset_manifest_hash="dataset-train-v1",
        source_run_id="hard-negative-bootstrap-r1",
        baseline_protocol_hash="protocol-train-v1",
        baseline_checkpoint_hash=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        predictions=[
            {
                "image_id": "train-a",
                "predicted_class": 0,
                "score": 0.91,
                "bbox": [1.0, 2.0, 10.0, 12.0],
            }
        ],
    )
    batch.write(prediction_path)
    return index_path, checkpoint, prediction_path, tmp_path / "hard-negative-manifest.json"


def _bind_checkpoint(node: ExperimentNode, checkpoint: Path) -> ExperimentNode:
    assert node.command_spec is not None
    return node.model_copy(
        update={
            "command_spec": node.command_spec.model_copy(
                update={
                    "metadata": {
                        **node.command_spec.metadata,
                        "baseline_checkpoint_path": checkpoint.as_posix(),
                    }
                }
            )
        }
    )


def test_train_sample_index_reads_only_configured_train_split(tmp_path: Path) -> None:
    context = _context(tmp_path)
    _assets(tmp_path)

    index = TrainSampleIndex.from_path(
        train_sample_index_from_yolo_data(
            context.data_yaml,
            dataset_manifest_hash="dataset-train-v1",
        ).write(tmp_path / "index-copy.json")
    )

    assert [item.image_id for item in index.samples] == ["train-a", "train-b"]
    assert all("val-a" not in (item.image_path or "") for item in index.samples)


def test_bootstrap_stages_produce_strict_train_manifest(tmp_path: Path) -> None:
    context = _context(tmp_path)
    index_path, checkpoint, prediction_path, manifest_path = _assets(tmp_path)
    source = _node(
        tmp_path,
        "paper_hard_negative_replay",
        components=["sampling.hard_negative_replay"],
    )
    source = _bind_checkpoint(source, checkpoint)
    control = _node(tmp_path, "matched-baseline", matched_control=True)
    state_path = tmp_path / "bootstrap.json"
    state = HardNegativeEvidenceBootstrap.create(
        candidate_id=source.candidate_config.candidate_id,
        source_run_id=context.run_id,
        dataset_manifest_hash="dataset-train-v1",
        baseline_protocol_hash="protocol-train-v1",
    )
    state.write(state_path)
    nodes = build_hard_negative_bootstrap_nodes(
        source_node=source,
        baseline_control_node=control,
        run_id=context.run_id,
        artifact_dir=tmp_path / "bootstrap-artifacts",
        dataset_manifest_hash="dataset-train-v1",
        baseline_protocol_hash="protocol-train-v1",
        data_yaml=context.data_yaml,
        train_index_path=index_path,
        baseline_checkpoint_path=checkpoint,
        bootstrap_path=state_path,
    )
    inference, manifest = nodes
    inference_prediction_path = Path(
        inference.command_spec.metadata["prediction_artifact_path"]
    )
    inference_prediction_path.parent.mkdir(parents=True, exist_ok=True)
    inference_prediction_path.write_text(
        prediction_path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    inference_result = execute_hard_negative_bootstrap_stage(
        inference,
        context.run_id,
        inference.command_spec,
    )
    assert inference_result.status == "completed"
    manifest_result = execute_hard_negative_bootstrap_stage(
        manifest,
        context.run_id,
        manifest.command_spec,
    )
    assert manifest_result.status == "completed"
    produced = HardNegativeManifest.from_path(
        Path(manifest.command_spec.metadata["manifest_artifact_path"])
    )
    assert produced.provenance_complete
    assert produced.source_split == "train"
    assert produced.train_index_hash == TrainSampleIndex.from_path(index_path).index_hash
    assert produced.baseline_checkpoint_hash == hashlib.sha256(
        checkpoint.read_bytes()
    ).hexdigest()
    assert Path(manifest.command_spec.metadata["manifest_artifact_path"]).is_file()
    assert produced.manifest_hash
    assert manifest_path != Path(manifest.command_spec.metadata["manifest_artifact_path"])


def test_bootstrap_state_enforces_order_before_candidate_activation() -> None:
    state = HardNegativeEvidenceBootstrap.create(
        candidate_id="replay",
        source_run_id="run-1",
        dataset_manifest_hash="dataset",
        baseline_protocol_hash="protocol",
    )

    with pytest.raises(ValueError, match="before baseline_train"):
        state.advance("train_split_inference")

    state.advance("baseline_train", artifact_path="baseline.pt")
    assert state.next_stage == "train_split_inference"
    assert not state.candidate_activation_allowed


def test_production_manifest_requires_complete_hash_bound_provenance() -> None:
    manifest = HardNegativeManifest.from_records(
        dataset_manifest_hash="dataset",
        source_run_id="run",
        baseline_protocol_hash="protocol",
        baseline_checkpoint_hash="a" * 64,
        train_index_hash="b" * 64,
        records=[
            {
                "image_id": "train-a",
                "sample_index": 0,
                "predicted_class": 0,
                "score": 0.9,
                "bbox": [0.0, 0.0, 1.0, 1.0],
                "error_type": "background_false_positive",
            }
        ],
    )

    with pytest.raises(ValueError, match="provenance is incomplete"):
        manifest.validate_runtime(
            dataset_manifest_hash="dataset",
            protocol_hash="protocol",
            dataset_length=1,
            require_provenance=True,
        )


def test_production_manifest_rejects_non_sha256_provenance() -> None:
    manifest = HardNegativeManifest.from_records(
        dataset_manifest_hash="dataset",
        source_run_id="run",
        baseline_protocol_hash="protocol",
        baseline_checkpoint_hash="checkpoint",
        train_index_hash="index",
        prediction_artifact_sha256="prediction",
        dataset_sample_count=1,
        records=[
            {
                "image_id": "train-a",
                "sample_index": 0,
                "predicted_class": 0,
                "score": 0.9,
                "bbox": [0.0, 0.0, 1.0, 1.0],
                "error_type": "background_false_positive",
            }
        ],
    )

    with pytest.raises(ValueError, match="invalid SHA-256"):
        manifest.validate_runtime(
            dataset_manifest_hash="dataset",
            protocol_hash="protocol",
            dataset_length=1,
            require_provenance=True,
        )


def test_bootstrap_rejects_validation_prediction_artifact(tmp_path: Path) -> None:
    context = _context(tmp_path)
    index_path, checkpoint, _, _ = _assets(tmp_path)
    source = _bind_checkpoint(
        _node(
            tmp_path,
            "paper_hard_negative_replay",
            components=["sampling.hard_negative_replay"],
        ),
        checkpoint,
    )
    control = _node(tmp_path, "matched-baseline", matched_control=True)
    state_path = tmp_path / "bootstrap.json"
    state = HardNegativeEvidenceBootstrap.create(
        candidate_id=source.candidate_config.candidate_id,
        source_run_id=context.run_id,
        dataset_manifest_hash="dataset-train-v1",
        baseline_protocol_hash="protocol-train-v1",
    )
    index = TrainSampleIndex.from_path(index_path)
    checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    state.baseline_checkpoint_hash = checkpoint_hash
    state.train_index_hash = index.index_hash
    state.advance("baseline_train", artifact_path=checkpoint)
    state.advance("train_split_inference", artifact_path=tmp_path / "val-predictions.json")
    state.write(state_path)
    nodes = build_hard_negative_bootstrap_nodes(
        source_node=source,
        baseline_control_node=control,
        run_id=context.run_id,
        artifact_dir=tmp_path / "bootstrap-artifacts",
        dataset_manifest_hash="dataset-train-v1",
        baseline_protocol_hash="protocol-train-v1",
        data_yaml=context.data_yaml,
        train_index_path=index_path,
        baseline_checkpoint_path=checkpoint,
        bootstrap_path=state_path,
    )
    manifest_node = nodes[1]
    prediction_path = Path(manifest_node.command_spec.metadata["prediction_artifact_path"])
    prediction_path.write_text(
        json.dumps(
            {
                "source_split": "val",
                "dataset_manifest_hash": "dataset-train-v1",
                "source_run_id": context.run_id,
                "baseline_protocol_hash": "protocol-train-v1",
                "baseline_checkpoint_hash": checkpoint_hash,
                "train_index_hash": index.index_hash,
                "predictions": [
                    {
                        "image_id": "train-a",
                        "category_id": 0,
                        "score": 0.95,
                        "bbox": [1, 2, 3, 4],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = execute_hard_negative_bootstrap_stage(
        manifest_node,
        context.run_id,
        manifest_node.command_spec,
    )

    assert result.status == "failed"
    assert "source_split" in result.message
    assert HardNegativeEvidenceBootstrap.from_path(
        state_path
    ).stage("hard_negative_manifest").status == "failed"


def test_replay_bootstrap_does_not_bind_other_component_families(tmp_path: Path) -> None:
    cases = [
        "loss.hard_negative_classification",
        "distillation.yolo26_teacher_student",
        "domain_adaptation.feature_alignment",
    ]
    for index, component in enumerate(cases):
        node = _node(tmp_path, f"candidate-{index}", components=[component])
        assert not _hard_negative_replay_needs_bootstrap(node.candidate_config, node)


def test_external_gpu_wait_is_requeued_after_contention_clears(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    node = _node(
        tmp_path,
        "paper_hard_negative_replay",
        components=["sampling.hard_negative_replay"],
    )
    item = ExecutionQueueItem.from_node("run-1", node)
    item.status = "needs_resume"
    item.resource_blockers = ["external_gpu_process"]
    queue = ExecutionQueue(run_id="run-1", items=[item])
    monkeypatch.setattr(
        "yolo_agent.agents.auto_optimization_loop.inspect_gpu_runtime",
        lambda command: type("Snapshot", (), {"has_external_training_conflict": False})(),
    )

    recovered = _recover_hard_negative_bootstrap_resource_waits(queue)

    assert recovered == [node.node_id]
    assert queue.items[0].status == "queued"
    assert queue.items[0].resource_blockers == []
    assert "queued for retry" in queue.items[0].message


def test_failed_replay_bootstrap_does_not_suppress_another_candidate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context(tmp_path)
    context.ensure_dirs()
    child = LoopOrchestrator(context)
    index_path, checkpoint, prediction_path, _ = _assets(tmp_path)
    control = _node(tmp_path, "matched-baseline", matched_control=True)
    sources = [
        _bind_checkpoint(
            _node(
                tmp_path,
                candidate_id,
                components=["sampling.hard_negative_replay"],
            ),
            checkpoint,
        )
        for candidate_id in ("replay-failed", "replay-ready")
    ]
    plan = RoundExecutionPlan(
        run_id=context.run_id,
        round_id="round-1",
        deferred_nodes=[control, *sources],
    )
    plan.to_yaml(context.artifact_path("round_execution_plan.yaml"))

    states = {}
    nodes_by_candidate = {}
    for source in sources:
        states[source.candidate_config.candidate_id] = _ensure_hard_negative_bootstrap(
            child,
            plan,
            source,
            control,
            protocol_hash="protocol-train-v1",
        )
        nodes_by_candidate[source.candidate_config.candidate_id] = {
            _bootstrap_stage: node
            for _bootstrap_stage, node in (
                (
                    item.command_spec.metadata["hard_negative_bootstrap_stage"],
                    item,
                )
                for item in plan.evidence_bootstrap_nodes
                if item.command_spec.metadata["source_candidate_id"]
                == source.candidate_config.candidate_id
            )
        }

    ready_id = "replay-ready"
    ready_state = states[ready_id]
    ready_nodes = nodes_by_candidate[ready_id]
    train_index = TrainSampleIndex.from_path(index_path)
    checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    ready_state.baseline_checkpoint_hash = checkpoint_hash
    ready_state.train_index_hash = train_index.index_hash
    ready_state.advance("baseline_train", artifact_path=checkpoint)
    ready_prediction_path = Path(
        ready_nodes["train_split_inference"].command_spec.metadata[
            "prediction_artifact_path"
        ]
    )
    ready_prediction_path.parent.mkdir(parents=True, exist_ok=True)
    ready_prediction_path.write_bytes(prediction_path.read_bytes())
    ready_index_path = Path(
        ready_nodes["train_split_inference"].command_spec.metadata["train_index_path"]
    )
    ready_index_path.write_bytes(index_path.read_bytes())
    ready_state.advance(
        "train_split_inference",
        artifact_path=ready_prediction_path,
    )
    ready_state.write(
        Path(
            child.context.metadata["hard_negative_bootstrap_states"][ready_id]
        )
    )

    failed_id = "replay-failed"
    failed_state = states[failed_id]
    failed_state.advance("baseline_train", artifact_path=checkpoint)
    failed_state.advance(
        "train_split_inference",
        status="failed",
        reason_codes=["mock_train_split_inference_failed"],
    )
    failed_state.write(
        Path(
            child.context.metadata["hard_negative_bootstrap_states"][failed_id]
        )
    )

    queue_items = []
    for candidate_id in (failed_id, ready_id):
        item = ExecutionQueueItem.from_node(
            context.run_id,
            nodes_by_candidate[candidate_id]["train_split_inference"],
        )
        item.status = "failed" if candidate_id == failed_id else "completed"
        queue_items.append(item)
    queue = ExecutionQueue(
        run_id=context.run_id,
        items=queue_items,
        metadata={"hard_negative_bootstrap_only": True},
    )
    ExecutionQueueStore(context.run_dir).save(queue)

    def fake_execute(_executor: str) -> ExecutionQueue:
        store = ExecutionQueueStore(context.run_dir)
        current = store.load()
        for item in list(current.items):
            if item.status != "queued":
                continue
            result = execute_hard_negative_bootstrap_stage(
                item.experiment_node,
                context.run_id,
                item.command,
            )
            item.mark_result(result)
            current = store.update_item(item)
        return current

    monkeypatch.setattr(child, "execute_queue", fake_execute)
    result = _execute_hard_negative_bootstrap_queue(
        child,
        executor="mock-ultralytics",
    )

    assert result is not None
    assert result.counts()["failed"] == 1
    assert result.counts()["completed"] == 2
    assert any(
        item.node_id == nodes_by_candidate[ready_id]["hard_negative_manifest"].node_id
        and item.status == "completed"
        for item in result.items
    )
    assert not any(
        item.node_id == nodes_by_candidate[failed_id]["hard_negative_manifest"].node_id
        for item in result.items
    )


def test_candidate_failure_is_not_reclassified_as_external_wait(tmp_path: Path) -> None:
    node = _node(
        tmp_path,
        "paper_hard_negative_replay",
        components=["sampling.hard_negative_replay"],
    )
    item = ExecutionQueueItem.from_node("run-1", node)
    item.status = "failed"
    item.message = "hard-negative manifest malformed"
    queue = ExecutionQueue(run_id="run-1", items=[item])

    assert _recover_hard_negative_bootstrap_resource_waits(queue) == []
    assert queue.items[0].status == "failed"


def test_bootstrap_rejects_missing_checkpoint_without_traceback(tmp_path: Path) -> None:
    context = _context(tmp_path)
    index_path, checkpoint, prediction_path, _ = _assets(tmp_path)
    source = _node(
        tmp_path,
        "paper_hard_negative_replay",
        components=["sampling.hard_negative_replay"],
    )
    source = _bind_checkpoint(source, checkpoint)
    control = _node(tmp_path, "matched-baseline", matched_control=True)
    state_path = tmp_path / "bootstrap.json"
    state = HardNegativeEvidenceBootstrap.create(
        candidate_id=source.candidate_config.candidate_id,
        source_run_id=context.run_id,
        dataset_manifest_hash="dataset-train-v1",
        baseline_protocol_hash="protocol-train-v1",
    )
    state.write(state_path)
    nodes = build_hard_negative_bootstrap_nodes(
        source_node=source,
        baseline_control_node=control,
        run_id=context.run_id,
        artifact_dir=tmp_path / "bootstrap-artifacts",
        dataset_manifest_hash="dataset-train-v1",
        baseline_protocol_hash="protocol-train-v1",
        data_yaml=context.data_yaml,
        train_index_path=index_path,
        baseline_checkpoint_path=tmp_path / "missing.pt",
        bootstrap_path=state_path,
    )
    result = execute_hard_negative_bootstrap_stage(
        nodes[0],
        context.run_id,
        nodes[0].command_spec,
    )

    assert result.status == "failed"
    assert "baseline checkpoint" in result.message
    persisted = HardNegativeEvidenceBootstrap.from_path(state_path)
    assert persisted.stage("train_split_inference").status == "failed"
    assert checkpoint.is_file()
    assert prediction_path.is_file()


def test_auto_bootstrap_queue_appends_manifest_after_inference(
    tmp_path: Path,
    monkeypatch,
) -> None:
    context = _context(tmp_path)
    context.ensure_dirs()
    child = LoopOrchestrator(context)
    index_path, checkpoint, prediction_path, _ = _assets(tmp_path)
    source = _node(
        tmp_path,
        "paper_hard_negative_replay",
        components=["sampling.hard_negative_replay"],
    )
    source = _bind_checkpoint(source, checkpoint)
    control = _node(tmp_path, "matched-baseline", matched_control=True)
    plan = RoundExecutionPlan(
        run_id=context.run_id,
        round_id="round-1",
        deferred_nodes=[control, source],
    )
    plan.to_yaml(context.artifact_path("round_execution_plan.yaml"))
    child.context.run_protocol_hash = "protocol-train-v1"
    child.context.to_yaml()
    state = _ensure_hard_negative_bootstrap(
        child,
        plan,
        source,
        control,
        protocol_hash="protocol-train-v1",
    )
    inference_node = next(
        item
        for item in plan.evidence_bootstrap_nodes
        if item.command_spec.metadata["hard_negative_bootstrap_stage"]
        == "train_split_inference"
    )
    inference_prediction_path = Path(
        inference_node.command_spec.metadata["prediction_artifact_path"]
    )
    inference_prediction_path.parent.mkdir(parents=True, exist_ok=True)
    inference_prediction_path.write_text(
        prediction_path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    inference_index_path = Path(inference_node.command_spec.metadata["train_index_path"])
    inference_index_path.write_bytes(index_path.read_bytes())
    assert state.next_stage == "baseline_train"
    assert _hard_negative_replay_needs_bootstrap(source.candidate_config, source)

    def fake_execute(_executor: str):
        store = ExecutionQueueStore(context.run_dir)
        queue = store.load()
        for item in list(queue.items):
            if item.status != "queued":
                continue
            result = execute_hard_negative_bootstrap_stage(
                item.experiment_node,
                context.run_id,
                item.command,
            )
            item.mark_result(result)
            store.update_item(item)
        return store.load()

    monkeypatch.setattr(child, "execute_queue", fake_execute)
    queue = _execute_hard_negative_bootstrap_queue(
        child,
        executor="mock-ultralytics",
    )

    assert queue is not None
    assert queue.counts()["completed"] == 2
    assert {item.command.command_type for item in queue.items} == {
        "hard_negative_inference",
        "hard_negative_manifest",
    }
    persisted = HardNegativeEvidenceBootstrap.from_path(
        Path(
            child.context.metadata["hard_negative_bootstrap_states"][
                source.candidate_config.candidate_id
            ]
        )
    )
    assert persisted.candidate_activation_allowed


def test_activation_binds_manifest_payload_back_to_candidate(tmp_path: Path) -> None:
    context = _context(tmp_path)
    context.ensure_dirs()
    child = LoopOrchestrator(context)
    index_path, checkpoint, prediction_path, _ = _assets(tmp_path)
    source = _node(
        tmp_path,
        "paper_hard_negative_replay",
        components=["sampling.hard_negative_replay"],
    )
    source = _bind_checkpoint(source, checkpoint)
    control = _node(tmp_path, "matched-baseline", matched_control=True)
    plan = RoundExecutionPlan(
        run_id=context.run_id,
        round_id="round-1",
        deferred_nodes=[control, source],
    )
    plan.to_yaml(context.artifact_path("round_execution_plan.yaml"))
    _ensure_hard_negative_bootstrap(
        child,
        plan,
        source,
        control,
        protocol_hash="protocol-train-v1",
    )
    inference, manifest = plan.evidence_bootstrap_nodes
    inference_prediction_path = Path(
        inference.command_spec.metadata["prediction_artifact_path"]
    )
    inference_prediction_path.parent.mkdir(parents=True, exist_ok=True)
    inference_prediction_path.write_text(
        prediction_path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    Path(inference.command_spec.metadata["train_index_path"]).write_bytes(
        index_path.read_bytes()
    )
    execute_hard_negative_bootstrap_stage(inference, context.run_id, inference.command_spec)
    execute_hard_negative_bootstrap_stage(manifest, context.run_id, manifest.command_spec)
    report = LoopPolicyEvaluationReport(
        evaluations=[
            LoopPolicyEvaluation(
                policy_id="policy-replay",
                decision="accepted",
                candidate_config=source.candidate_config,
                experiment_node=source,
            )
        ]
    )
    write_yaml(
        context.artifact_path("policy_evaluation.yaml"),
        report.model_dump(mode="json"),
    )

    activated = _activate_hard_negative_bootstrap_candidates(child)

    assert activated == [source.candidate_config.candidate_id]
    updated = LoopPolicyEvaluationReport.model_validate(
        read_yaml(context.artifact_path("policy_evaluation.yaml"))
    ).evaluations[0].candidate_config
    assert updated is not None
    assert updated.train_overrides["manifest_path"]
    assert updated.train_overrides["require_provenance"] is True
    assert not _hard_negative_replay_needs_bootstrap(updated, source)
    updated_plan = RoundExecutionPlan.from_yaml(
        context.artifact_path("round_execution_plan.yaml")
    )
    assert not updated_plan.evidence_bootstrap_nodes
    projected = RoundExecutionPlan.from_yaml(
        context.artifact_path("round_execution_plan.yaml")
    ).experiment_projection()
    assert not any(
        item.command_spec
        and item.command_spec.command_type
        in {"hard_negative_inference", "hard_negative_manifest"}
        for item in projected.nodes
    )
