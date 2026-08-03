"""Offline multi-mechanism paper auto-optimization state-machine tests."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

import yolo_agent.certification.paper_auto_optimization_multi as multi_module
from yolo_agent.adapters.ultralytics.plugin_context import PluginRuntimeEvidence
from yolo_agent.certification.paper_auto_optimization import (
    PaperAutoOptimizationAcceptanceSuite,
)
from yolo_agent.certification.paper_auto_optimization_maturity import (
    PaperPilotReproductionEvidence,
)
from yolo_agent.certification.paper_auto_optimization_protocol import (
    build_paper_protocol_identity,
    hash_payload,
)
from yolo_agent.certification.paper_auto_optimization_research import (
    PaperAcceptanceResearchContext,
    PaperAcceptanceTrackContext,
)
from yolo_agent.certification.paper_auto_optimization_tracks import (
    PAPER_ACCEPTANCE_RECIPES,
)
from yolo_agent.certification.runner import BackendEvaluation, BackendRun
from yolo_agent.components.adapters.base import ExpectedArtifact, RollbackPlan
from yolo_agent.components.adapters.runtime import (
    AdapterRuntimePayload,
    RuntimePluginReference,
)


PLUGIN_BY_COMPONENT = {
    "sampling.small_object": (
        "dataloader_plugin",
        "yolo_agent.components.adapters.sampling.small_object_sampling:"
        "SmallObjectSamplingRuntimePlugin",
        "build_train_dataloader",
        ["sampler_manifest"],
    ),
    "loss.quality.correlation": (
        "loss_plugin",
        "yolo_agent.components.adapters.losses.quality_alignment:"
        "QualityAlignmentRuntimePlugin",
        "compute_loss",
        ["auxiliary_loss_correlation_evidence"],
    ),
    "distillation.yolo26_teacher_student": (
        "loss_plugin",
        "yolo_agent.components.adapters.distillation.yolo26_distillation:"
        "YOLO26DistillationRuntimePlugin",
        "compute_loss",
        ["distillation_evidence"],
    ),
    "head.p2_small_object": (
        "model_graph_plugin",
        "yolo_agent.components.adapters.head.p2_head:P2HeadRuntimePlugin",
        "build_model",
        ["p2_head_manifest", "p2_model_yaml"],
    ),
}


class MockGpuBackend:
    def __init__(
        self,
        *,
        omit_eval_component: str | None = None,
        pilot_10_gain: float = 0.05,
        pilot_10_target_improved: bool = True,
    ) -> None:
        self.train_calls: list[dict[str, Any]] = []
        self.omit_eval_component = omit_eval_component
        self.pilot_10_gain = pilot_10_gain
        self.pilot_10_target_improved = pilot_10_target_improved

    @staticmethod
    def environment() -> dict[str, object]:
        return {
            "cuda_available": True,
            "gpu_name": "mock-gpu",
            "ultralytics_version": "8.4.mock",
        }

    def train(self, **kwargs: object) -> BackendRun:
        candidate_id = str(kwargs["candidate_id"])
        node_id = str(kwargs["node_id"])
        epochs = int(kwargs["epochs"])  # type: ignore[arg-type]
        seed = int(kwargs["seed"])  # type: ignore[arg-type]
        protocol_hash = str(kwargs["protocol_hash"])
        data_yaml = Path(kwargs["data_yaml"])  # type: ignore[arg-type]
        workdir = Path(kwargs["workdir"])  # type: ignore[arg-type]
        self.train_calls.append(
            {
                "candidate_id": candidate_id,
                "node_id": node_id,
                "epochs": epochs,
                "seed": seed,
                "protocol_hash": protocol_hash,
            }
        )
        run_dir = workdir / "mock_runs" / node_id
        checkpoint = run_dir / "weights" / "best.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(b"mock checkpoint")
        artifacts = (
            _mock_runtime_artifacts(
                workdir / "mock_runtime" / node_id,
                component_id=candidate_id,
                protocol_hash=protocol_hash,
                node_id=node_id,
            )
            if candidate_id in PLUGIN_BY_COMPONENT
            else {}
        )
        return BackendRun(
            candidate_id=candidate_id,
            node_id=node_id,
            run_dir=run_dir,
            checkpoint=checkpoint,
            command=["mock-train", node_id],
            runtime_artifacts=artifacts,
            protocol_identity=build_paper_protocol_identity(
                data_yaml=data_yaml,
                protocol_hash=protocol_hash,
                objective_hash=str(kwargs.get("objective_hash"))
                if kwargs.get("objective_hash")
                else hash_payload({"mock": "objective-bound-by-protocol"}),
                epochs=epochs,
                seed=seed,
                ultralytics_version="8.4.mock",
            ),
        )

    def evaluate(self, **kwargs: object) -> BackendEvaluation:
        run = kwargs["run"]
        assert isinstance(run, BackendRun)
        component_id = _component_for_run(run)
        baseline = run.candidate_id.startswith("baseline_")
        if not baseline and component_id == self.omit_eval_component:
            raise RuntimeError("mock candidate predictions missing")
        workdir = Path(kwargs["workdir"])  # type: ignore[arg-type]
        data_yaml = Path(kwargs["data_yaml"])  # type: ignore[arg-type]
        pilot_10 = run.node_id.endswith("pilot_10")
        gains = {
            "sampling.small_object": 0.04,
            "loss.quality.correlation": 0.03,
            "distillation.yolo26_teacher_student": 0.02,
            "head.p2_small_object": 0.01,
        }
        gain = 0.0 if baseline else gains[component_id]
        if pilot_10 and not baseline:
            gain = self.pilot_10_gain
        output = workdir / "mock_eval" / run.node_id
        output.mkdir(parents=True, exist_ok=True)
        evaluation = output / "coco_eval.json"
        evaluation.write_text(
            json.dumps(
                {
                    "AP": 0.30 + gain,
                    "AP50": 0.50 + gain,
                    "AP75": 0.28 + gain,
                    "AP_small": 0.20 + gain,
                    "AP_medium": 0.32 + gain,
                    "AP_large": 0.40 + gain,
                    "AR_small": 0.25 + gain,
                    "per_class_ap": {"object": 0.30 + gain},
                    "per_class_ar": {"object": 0.40 + gain},
                }
            ),
            encoding="utf-8",
        )
        annotations = json.loads(
            (
                data_yaml.parent / "annotations" / "instances_val2017.json"
            ).read_text(encoding="utf-8")
        )
        predictions = output / "predictions.json"
        predictions.write_text(
            json.dumps(
                [
                    {
                        "image_id": item["image_id"],
                        "category_id": item["category_id"],
                        "bbox": item["bbox"],
                        "score": 0.9,
                    }
                    for item in annotations["annotations"]
                    if int(item["image_id"]) <= (2 if baseline else 4)
                ]
            ),
            encoding="utf-8",
        )
        false_negatives = 5
        localization_errors = 5
        if not baseline:
            if component_id in {
                "sampling.small_object",
                "distillation.yolo26_teacher_student",
                "head.p2_small_object",
            }:
                false_negatives = 3
            if component_id == "loss.quality.correlation":
                localization_errors = 3
            if pilot_10 and not self.pilot_10_target_improved:
                false_negatives = 5
                localization_errors = 5
        errors = output / "coco_error_report.json"
        errors.write_text(
            json.dumps(
                {
                    "false_negative_top_classes": [
                        {
                            "category_id": 1,
                            "name": "object",
                            "false_negative": false_negatives,
                            "recall": 0.5 + gain,
                        }
                    ],
                    "localization_error_top_classes": [
                        {
                            "category_id": 1,
                            "name": "object",
                            "localization_error": localization_errors,
                        }
                    ],
                    "background_false_positive_top_classes": [],
                    "class_confusion_pairs": {},
                }
            ),
            encoding="utf-8",
        )
        return BackendEvaluation(
            eval_path=evaluation,
            predictions_path=predictions,
            error_report_path=errors,
            latency_ms=10.0,
            model_size_mb=5.0,
        )


def _mock_runtime_artifacts(
    root: Path,
    *,
    component_id: str,
    protocol_hash: str,
    node_id: str,
) -> dict[str, Path]:
    root.mkdir(parents=True, exist_ok=True)
    plugin_kind, reference, hook, artifact_names = PLUGIN_BY_COMPONENT[component_id]
    plugin = RuntimePluginReference(
        reference=reference,
        options={"imgsz": 640},
        required_hooks=[hook],
    )
    plugin_fields = {
        "dataloader_plugin": [],
        "loss_plugin": [],
        "model_graph_plugin": [],
    }
    plugin_fields[plugin_kind] = [plugin]
    expected = [
        ExpectedArtifact(name=name, relative_path=Path(f"{name}.json"))
        for name in artifact_names
    ]
    payload = AdapterRuntimePayload(
        component_ids=[component_id],
        adapter_classes=["MockCertifiedAdapter"],
        adapter_versions={component_id: "mock-v1"},
        source_commits={component_id: "mock-commit"},
        **plugin_fields,
        changed_variables={
            next(
                item.changed_variable
                for item in PAPER_ACCEPTANCE_RECIPES
                if item.component_id == component_id
            ): "mock-active"
        },
        expected_artifacts=expected,
        rollback_plan=RollbackPlan(actions=["discard mock runtime"]),
        protocol_hash=protocol_hash,
        base_command=["mock-train", node_id],
        supports_amp=True,
        supports_ddp=True,
        supports_resume=True,
    )
    payload_path = payload.write(root / "adapter_runtime_payload.yaml")
    artifacts: dict[str, Path] = {
        "runtime_payload": payload_path,
        "plugin_runtime_evidence": root / "plugin_runtime_evidence.json",
    }
    for item in expected:
        path = root / item.relative_path
        path.write_text("{}", encoding="utf-8")
        artifacts[item.name] = path
    artifacts["plugin_runtime_evidence"].write_text(
        PluginRuntimeEvidence(
            payload_hash=payload.payload_hash,
            protocol_hash=protocol_hash,
            component_ids=[component_id],
            changed_variables=payload.changed_variables,
            ultralytics_version="8.4.mock",
            signature_hash="mock-signature",
            compatible=True,
            hook_call_counts={reference: {hook: 1}},
        ).model_dump_json(indent=2),
        encoding="utf-8",
    )
    return artifacts


class MockResearchPreparer:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.calls = 0

    def prepare(self, output_path: Path | str) -> PaperAcceptanceResearchContext:
        self.calls += 1
        snapshot = self.root / "snapshot"
        snapshot.mkdir(parents=True, exist_ok=True)
        (snapshot / "snapshot.yaml").write_text(
            "snapshot_hash: mock-snapshot\n",
            encoding="utf-8",
        )
        tracks = [
            PaperAcceptanceTrackContext(
                track_id=recipe.track_id,
                component_id=recipe.component_id,
                component_family=recipe.component_family,
                paper_ids=[f"paper-{recipe.track_id}"],
                method_profile_ids=[f"profile-{recipe.track_id}"],
                implementation_decision_hashes=[f"decision-{recipe.track_id}"],
                adapter_hash=(str(index) * 64)[:64],
                maturity="gpu_certified",
                maturity_protocol_hash=f"protocol-{recipe.track_id}",
                ultralytics_version="8.4.mock",
            )
            for index, recipe in enumerate(PAPER_ACCEPTANCE_RECIPES, start=1)
        ]
        sampling = tracks[0]
        context = PaperAcceptanceResearchContext(
            snapshot_hash="mock-snapshot",
            snapshot_path=snapshot,
            source_commit="mock-commit",
            paper_ids=sampling.paper_ids,
            method_profile_ids=sampling.method_profile_ids,
            implementation_decision_hashes=sampling.implementation_decision_hashes,
            adapter_hash=sampling.adapter_hash,
            maturity=sampling.maturity,
            maturity_protocol_hash=sampling.maturity_protocol_hash,
            ultralytics_version="8.4.mock",
            tracks=tracks,
        )
        context.to_yaml(output_path, exclude_none=True, sort_keys=False)
        return context


def _fake_promote(**kwargs: object) -> PaperPilotReproductionEvidence:
    track = kwargs["track"]
    recipe = kwargs["recipe"]
    research = kwargs["research"]
    assert isinstance(track, PaperAcceptanceTrackContext)
    evidence = PaperPilotReproductionEvidence(
        component_id=track.component_id,
        recipe_id=recipe.recipe_id,  # type: ignore[union-attr]
        paper_ids=track.paper_ids,
        adapter_hash=track.adapter_hash,
        snapshot_hash=research.snapshot_hash,  # type: ignore[union-attr]
        acceptance_protocol_hash=str(kwargs["acceptance_protocol_hash"]),
        maturity_protocol_hash=track.maturity_protocol_hash,
        pilot_3=kwargs["pilot_3"],  # type: ignore[arg-type]
        pilot_10=kwargs["pilot_10"],  # type: ignore[arg-type]
    )
    evidence.to_yaml(
        kwargs["output_path"],  # type: ignore[arg-type]
        exclude_none=True,
        sort_keys=False,
    )
    return evidence


def _component_for_run(run: BackendRun) -> str:
    if run.candidate_id in PLUGIN_BY_COMPONENT:
        return run.candidate_id
    return next(
        component_id
        for component_id in PLUGIN_BY_COMPONENT
        if component_id.replace(".", "_") in run.node_id
    )


def _memory_records(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_full_offline_multi_mechanism_state_machine(
    tmp_path: Path,
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    backend = MockGpuBackend()
    research = MockResearchPreparer(tmp_path)
    monkeypatch.setattr(
        multi_module,
        "promote_component_pilot_reproduced",
        _fake_promote,
    )

    report = PaperAutoOptimizationAcceptanceSuite(backend, research).run(
        workdir=tmp_path / "acceptance",
        maturity_registry=tmp_path / "maturity.yaml",
        policy_memory_root=tmp_path / "memory",
        execute_real_gpu=True,
    )

    assert report.status == "passed", report.failures
    assert research.calls == 1
    assert set(report.component_families) == {
        "sampling",
        "auxiliary_loss",
        "distillation",
        "model_graph",
    }
    assert len([item for item in report.paired_deltas if item.stage_id == "pilot_3"]) == 4
    assert report.asha_survivors == ["sampling.small_object"]
    assert report.pilot_reproduced_component_ids == ["sampling.small_object"]
    assert report.objective_hash
    assert all(
        identity.objective_hash == report.objective_hash
        for identity in report.protocol_identities.values()
    )
    assert len(backend.train_calls) == 10
    assert sum(call["epochs"] == 3 for call in backend.train_calls) == 8
    assert sum(call["epochs"] == 10 for call in backend.train_calls) == 2
    assert all(call["epochs"] <= 10 and call["seed"] == 1 for call in backend.train_calls)
    assert not any("full" in call["node_id"] for call in backend.train_calls)
    records = _memory_records(tmp_path / "memory" / "policy_memory.jsonl")
    assert len(records) == 4
    assert sum(bool(item["failure_reason"]) for item in records) == 3
    assert any(item["action_fingerprint"]["fidelity"] == "pilot_10" for item in records)


def test_suite_is_gpu_opt_in_and_does_not_prepare_snapshot(tmp_path: Path) -> None:
    backend = MockGpuBackend()
    research = MockResearchPreparer(tmp_path)

    report = PaperAutoOptimizationAcceptanceSuite(backend, research).run(
        workdir=tmp_path / "acceptance"
    )

    assert report.status == "skipped"
    assert research.calls == 0
    assert backend.train_calls == []


def test_suite_rejects_concurrent_use_of_same_workdir(tmp_path: Path) -> None:
    lock_path = tmp_path / ".paper_auto_optimization.lock"
    lock_path.write_text(
        json.dumps({"pid": os.getpid(), "token": "active"}),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="workdir is already active"):
        PaperAutoOptimizationAcceptanceSuite(MockGpuBackend()).run(workdir=tmp_path)


def test_missing_evidence_stops_before_next_component(tmp_path: Path) -> None:
    backend = MockGpuBackend(omit_eval_component="sampling.small_object")

    report = PaperAutoOptimizationAcceptanceSuite(
        backend,
        MockResearchPreparer(tmp_path),
    ).run(
        workdir=tmp_path / "acceptance",
        policy_memory_root=tmp_path / "memory",
        execute_real_gpu=True,
    )

    assert report.status == "recovery"
    assert report.evidence_recovery_actions == [
        "recover_control_coco_post_eval",
        "recover_candidate_coco_post_eval",
    ]
    assert len(backend.train_calls) == 2
    assert not (tmp_path / "memory" / "policy_memory.jsonl").exists()


def test_pilot_10_elimination_is_recorded_without_maturity_promotion(
    tmp_path: Path,
) -> None:
    backend = MockGpuBackend(pilot_10_gain=0.0, pilot_10_target_improved=False)

    report = PaperAutoOptimizationAcceptanceSuite(
        backend,
        MockResearchPreparer(tmp_path),
    ).run(
        workdir=tmp_path / "acceptance",
        policy_memory_root=tmp_path / "memory",
        execute_real_gpu=True,
    )

    assert report.status == "failed"
    assert report.pilot_reproduced_component_ids == []
    records = _memory_records(tmp_path / "memory" / "policy_memory.jsonl")
    sampling = next(
        item
        for item in records
        if item["action_fingerprint"]["component_ids"] == ["sampling.small_object"]
    )
    assert sampling["evidence_status"] == "failed"
    assert sampling["action_fingerprint"]["fidelity"] == "pilot_10"
    assert sampling["pilot_10_delta"] == pytest.approx(0.0)


def test_protocol_mismatch_blocks_before_post_eval(tmp_path: Path) -> None:
    class ProtocolMismatchBackend(MockGpuBackend):
        def train(self, **kwargs: object) -> BackendRun:
            run = super().train(**kwargs)
            if run.candidate_id == "sampling.small_object":
                assert run.protocol_identity is not None
                run.protocol_identity = run.protocol_identity.model_copy(
                    update={"batch_policy_hash": "polluted-batch-policy"}
                )
            return run

    backend = ProtocolMismatchBackend()
    report = PaperAutoOptimizationAcceptanceSuite(
        backend,
        MockResearchPreparer(tmp_path),
    ).run(
        workdir=tmp_path / "acceptance",
        execute_real_gpu=True,
    )

    assert report.status == "failed"
    assert "batch_policy_hash" in report.failures[0]
    assert len(backend.train_calls) == 2
