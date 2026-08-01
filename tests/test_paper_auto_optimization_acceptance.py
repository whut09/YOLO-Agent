"""Offline state-machine tests for paper-driven optimization acceptance."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

import yolo_agent.certification.paper_auto_optimization as acceptance_module

from yolo_agent.adapters.ultralytics.plugin_context import PluginRuntimeEvidence
from yolo_agent.certification.paper_auto_optimization import (
    PaperAutoOptimizationAcceptanceSuite,
)
from yolo_agent.certification.paper_auto_optimization_maturity import (
    PaperPilotReproductionEvidence,
)
from yolo_agent.certification.paper_auto_optimization_research import (
    PaperAcceptanceResearchContext,
)
from yolo_agent.certification.paper_auto_optimization_protocol import (
    build_paper_protocol_identity,
    hash_payload,
)
from yolo_agent.certification.runner import BackendEvaluation, BackendRun
from yolo_agent.components.adapters.base import RollbackPlan
from yolo_agent.components.adapters.runtime import (
    AdapterRuntimePayload,
    RuntimePluginReference,
)
from yolo_agent.components.adapters.sampling.small_object_sampling import (
    SmallObjectSamplingManifest,
)


class MockGpuBackend:
    def __init__(
        self,
        *,
        omit_candidate_eval: bool = False,
        gain: float = 0.04,
        candidate_false_negatives: int = 1,
    ) -> None:
        self.train_calls: list[tuple[str, int]] = []
        self.omit_candidate_eval = omit_candidate_eval
        self.gain = gain
        self.candidate_false_negatives = candidate_false_negatives

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
        self.train_calls.append((node_id, epochs))
        run_dir = workdir / "mock_runs" / node_id
        checkpoint = run_dir / "weights" / "best.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(b"mock checkpoint")
        artifacts: dict[str, Path] = {}
        if candidate_id == "sampling.small_object":
            artifacts = _mock_runtime_artifacts(
                workdir / "mock_runtime" / node_id,
                protocol_hash=protocol_hash,
                seed=seed,
                node_id=node_id,
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
                objective_hash=hash_payload(
                    {
                        "primary_metric": "ap_small",
                        "target_metrics": ["per_class_ar/object"],
                        "target_error_facts": ["false_negative/object"],
                    }
                ),
                epochs=epochs,
                seed=seed,
                ultralytics_version="8.4.mock",
            ),
        )

    def evaluate(self, **kwargs: object) -> BackendEvaluation:
        run = kwargs["run"]
        assert isinstance(run, BackendRun)
        if self.omit_candidate_eval and run.candidate_id == "sampling.small_object":
            raise RuntimeError("mock candidate predictions missing")
        workdir = Path(kwargs["workdir"])  # type: ignore[arg-type]
        data_yaml = Path(kwargs["data_yaml"])  # type: ignore[arg-type]
        baseline = run.candidate_id.startswith("baseline")
        gain = 0.0 if baseline else self.gain
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
                data_yaml.parent
                / "annotations"
                / "instances_val2017.json"
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
        errors = output / "coco_error_report.json"
        errors.write_text(
            json.dumps(
                {
                    "false_negative_top_classes": [
                        {
                            "category_id": 1,
                            "name": "object",
                            "false_negative": (
                                2 if baseline else self.candidate_false_negatives
                            ),
                            "recall": 0.5 + gain,
                        }
                    ],
                    "background_false_positive_top_classes": [],
                    "localization_error_top_classes": [],
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
    protocol_hash: str,
    seed: int,
    node_id: str,
) -> dict[str, Path]:
    root.mkdir(parents=True, exist_ok=True)
    reference = (
        "yolo_agent.components.adapters.sampling.small_object_sampling:"
        "SmallObjectSamplingRuntimePlugin"
    )
    payload = AdapterRuntimePayload(
        component_ids=["sampling.small_object"],
        adapter_classes=["SmallObjectSamplingAdapter"],
        adapter_versions={"sampling.small_object": "mock-v1"},
        source_commits={"sampling.small_object": "mock-commit"},
        dataloader_plugin=[
            RuntimePluginReference(
                reference=reference,
                options={"imgsz": 640},
                required_hooks=["build_train_dataloader"],
            )
        ],
        changed_variables={"data.sampling_policy": {"imgsz": 640}},
        rollback_plan=RollbackPlan(actions=["discard mock runtime"]),
        protocol_hash=protocol_hash,
        base_command=["mock-train", node_id],
        supports_amp=True,
        supports_ddp=True,
        supports_resume=True,
    )
    payload_path = payload.write(root / "adapter_runtime_payload.yaml")
    manifest = root / "sampler_manifest.json"
    manifest.write_text(
        SmallObjectSamplingManifest(
            dataset_manifest="mock-mini-coco",
            protocol_hash=protocol_hash,
            runtime_payload_hash=payload.payload_hash,
            split="train",
            seed=seed,
            area_thresholds={"small": 0.01},
            image_count=4,
            small_image_count=2,
            raw_weights=[2.0, 1.0, 2.0, 1.0],
            final_weights=[2.0, 1.0, 2.0, 1.0],
            image_paths=["a.jpg", "b.jpg", "c.jpg", "d.jpg"],
            sample_count=4,
            adapter_hash="mock-adapter",
            val_unchanged=True,
        ).model_dump_json(indent=2),
        encoding="utf-8",
    )
    runtime_evidence = root / "plugin_runtime_evidence.json"
    runtime_evidence.write_text(
        PluginRuntimeEvidence(
            payload_hash=payload.payload_hash,
            protocol_hash=protocol_hash,
            ultralytics_version="8.4.mock",
            signature_hash="mock-signature",
            compatible=True,
            hook_call_counts={reference: {"build_train_dataloader": 1}},
        ).model_dump_json(indent=2),
        encoding="utf-8",
    )
    return {
        "runtime_payload": payload_path,
        "sampler_manifest": manifest,
        "plugin_runtime_evidence": runtime_evidence,
    }


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
        context = PaperAcceptanceResearchContext(
            snapshot_hash="mock-snapshot",
            snapshot_path=snapshot,
            source_commit="mock-commit",
            paper_ids=["paper-small-object"],
            method_profile_ids=["profile-small-object"],
            implementation_decision_hashes=["decision-small-object"],
            adapter_hash="a" * 64,
            maturity="gpu_certified",
            maturity_protocol_hash="component-gpu-protocol",
            ultralytics_version="8.4.mock",
        )
        context.to_yaml(output_path, exclude_none=True, sort_keys=False)
        return context


def _fake_promote(**kwargs: object) -> PaperPilotReproductionEvidence:
    evidence = PaperPilotReproductionEvidence(
        paper_ids=kwargs["research"].paper_ids,  # type: ignore[union-attr]
        adapter_hash=kwargs["research"].adapter_hash,  # type: ignore[union-attr]
        snapshot_hash=kwargs["research"].snapshot_hash,  # type: ignore[union-attr]
        acceptance_protocol_hash=str(kwargs["acceptance_protocol_hash"]),
        maturity_protocol_hash=kwargs["research"].maturity_protocol_hash,  # type: ignore[union-attr]
        pilot_3=kwargs["pilot_3"],  # type: ignore[arg-type]
        pilot_10=kwargs["pilot_10"],  # type: ignore[arg-type]
    )
    evidence.to_yaml(kwargs["output_path"], exclude_none=True, sort_keys=False)  # type: ignore[arg-type]
    return evidence


def test_full_offline_paper_auto_optimization_state_machine(
    tmp_path: Path,
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    backend = MockGpuBackend()
    research = MockResearchPreparer(tmp_path)
    monkeypatch.setattr(
        acceptance_module,
        "promote_sampling_pilot_reproduced",
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
    assert report.recipe_id == "sampling.small_object"
    assert report.asha_survivor == "sampling.small_object"
    assert report.pilot_reproduced is True
    assert report.scalar_hpo_enabled is False
    assert [item.stage_id for item in report.paired_deltas] == [
        "pilot_3",
        "pilot_10",
    ]
    assert all(item.ap_small_delta and item.ap_small_delta > 0 for item in report.paired_deltas)
    assert backend.train_calls == [
        ("baseline_pilot_3", 3),
        ("sampling_small_object_pilot_3", 3),
        ("baseline_pilot_10", 10),
        ("sampling_small_object_pilot_10", 10),
    ]
    assert (tmp_path / "memory" / "policy_memory.jsonl").is_file()
    assert (
        tmp_path / "acceptance" / "paper_auto_optimization_report.yaml"
    ).is_file()


def test_suite_is_gpu_opt_in_and_does_not_prepare_snapshot(tmp_path: Path) -> None:
    backend = MockGpuBackend()
    research = MockResearchPreparer(tmp_path)

    report = PaperAutoOptimizationAcceptanceSuite(backend, research).run(
        workdir=tmp_path / "acceptance"
    )

    assert report.status == "skipped"
    assert research.calls == 0
    assert backend.train_calls == []


def test_suite_rejects_concurrent_use_of_the_same_workdir(tmp_path: Path) -> None:
    lock_path = tmp_path / ".paper_auto_optimization.lock"
    lock_path.write_text(
        json.dumps({"pid": os.getpid(), "token": "active"}), encoding="utf-8"
    )

    with pytest.raises(RuntimeError, match="workdir is already active"):
        PaperAutoOptimizationAcceptanceSuite(MockGpuBackend()).run(workdir=tmp_path)


def test_missing_post_eval_enters_recovery_without_pilot_10(tmp_path: Path) -> None:
    backend = MockGpuBackend(omit_candidate_eval=True)

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
    assert backend.train_calls == [
        ("baseline_pilot_3", 3),
        ("sampling_small_object_pilot_3", 3),
    ]
    assert not (tmp_path / "memory" / "policy_memory.jsonl").exists()


def test_non_improving_pilot_is_eliminated_before_pilot_10(tmp_path: Path) -> None:
    backend = MockGpuBackend(gain=0.0, candidate_false_negatives=2)

    report = PaperAutoOptimizationAcceptanceSuite(
        backend,
        MockResearchPreparer(tmp_path),
    ).run(
        workdir=tmp_path / "acceptance",
        policy_memory_root=tmp_path / "memory",
        execute_real_gpu=True,
    )

    assert report.status == "failed"
    assert "ASHA eliminated sampling.small_object" in report.failures[0]
    assert "ap_small_improved" in report.failures[0]
    assert "false_negative_reduced" in report.failures[0]
    assert backend.train_calls == [
        ("baseline_pilot_3", 3),
        ("sampling_small_object_pilot_3", 3),
    ]


def test_protocol_mismatch_blocks_candidate_before_post_eval(tmp_path: Path) -> None:
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
    assert report.failures == [
        "candidate/control protocol mismatch: batch_policy_hash"
    ]
    assert backend.train_calls == [
        ("baseline_pilot_3", 3),
        ("sampling_small_object_pilot_3", 3),
    ]
