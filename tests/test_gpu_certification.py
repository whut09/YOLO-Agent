"""Offline tests for the opt-in real GPU certification orchestrator."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

import yolo_agent.certification.runner as certification_runner
from yolo_agent.adapters.ultralytics.plugin_context import PluginRuntimeEvidence
from yolo_agent.certification.fixture import (
    create_mini_coco_fixture,
    load_mini_coco_fixture_manifest,
)
from yolo_agent.certification.runner import (
    BackendEvaluation,
    BackendRun,
    CERTIFICATION_INSTALL_COMMAND,
    RealGpuAcceptanceSuite,
    UltralyticsGpuBackend,
)
from yolo_agent.certification.schemas import (
    CertificationObjectiveResult,
    CertificationPromotionResult,
    CertificationReport,
    CertificationStage,
)
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
        small_object_gain: float = 0.04,
        small_object_latency_delta: float = 0.1,
        small_object_fn: int = 1,
    ) -> None:
        self.train_calls: list[tuple[str, int]] = []
        self.protocol_calls: list[dict[str, object]] = []
        self.small_object_gain = small_object_gain
        self.small_object_latency_delta = small_object_latency_delta
        self.small_object_fn = small_object_fn

    def environment(self) -> dict[str, object]:
        return {
            "cuda_available": True,
            "gpu_name": "mock-gpu",
            "ultralytics_version": "8.4.mock",
        }

    def certify_component(
        self,
        *,
        component_id: str,
        workdir: Path,
        device: str,
    ) -> CertificationStage:
        report = workdir / "mock_component_certification.yaml"
        report.write_text(
            f"component_id: {component_id}\ndevice: {device}\nstatus: passed\n",
            encoding="utf-8",
        )
        return CertificationStage(
            stage_id="component_runtime_certification",
            status="passed",
            artifacts={"mock_report": report.as_posix()},
            metrics={
                "component_id": component_id,
                "cpu_final_maturity": "smoke_passed",
                "gpu_final_maturity": "gpu_certified",
            },
        )

    def train_entrypoint(
        self, *, data_yaml: Path, model: str, workdir: Path, device: str
    ) -> list[str]:
        return [
            "yolo-agent",
            "train",
            "--data",
            str(data_yaml),
            "--model",
            model,
            "--dry-run",
        ]

    def train(
        self,
        *,
        candidate_id: str,
        node_id: str,
        data_yaml: Path,
        model: str,
        workdir: Path,
        device: str,
        epochs: int,
        seed: int,
        protocol_hash: str,
        overrides: dict[str, object],
    ) -> BackendRun:
        del data_yaml, model, device
        self.train_calls.append((node_id, epochs))
        self.protocol_calls.append(
            {
                "candidate_id": candidate_id,
                "node_id": node_id,
                "epochs": epochs,
                "seed": seed,
                "protocol_hash": protocol_hash,
                "overrides": overrides,
            }
        )
        run_dir = workdir / "mock_runs" / node_id
        checkpoint = run_dir / "weights" / "best.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(b"mock checkpoint")
        runtime_artifacts: dict[str, Path] = {}
        if candidate_id == "small_object_sampling":
            runtime_dir = workdir / "mock_runtime" / node_id
            runtime_dir.mkdir(parents=True, exist_ok=True)
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
            payload_path = payload.write(runtime_dir / "adapter_runtime_payload.yaml")
            (runtime_dir / "sampler_manifest.json").write_text(
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
            (runtime_dir / "plugin_runtime_evidence.json").write_text(
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
            runtime_artifacts = {
                "runtime_payload": payload_path,
                "sampler_manifest": runtime_dir / "sampler_manifest.json",
                "plugin_runtime_evidence": runtime_dir / "plugin_runtime_evidence.json",
            }
        return BackendRun(
            candidate_id=candidate_id,
            node_id=node_id,
            run_dir=run_dir,
            checkpoint=checkpoint,
            command=["mock-train", node_id, str(epochs)],
            runtime_artifacts=runtime_artifacts,
        )


class MockPaperBackend:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def prepare(self, root: Path):
        self.calls.extend(["mock_catalog", "mock_llm", "mock_adapter"])
        stages = [
            CertificationStage(
                stage_id="catalog_import",
                status="passed",
                metrics={"backend": "mock_catalog"},
            ),
            CertificationStage(
                stage_id="snapshot_creation",
                status="passed",
                metrics={"snapshot_hash": "mock-snapshot"},
            ),
            CertificationStage(
                stage_id="diagnosis_linked_paper_prior",
                status="passed",
                metrics={"backend": "mock_llm"},
            ),
            CertificationStage(
                stage_id="eligibility_gate", status="passed", metrics={"eligible": True}
            ),
            CertificationStage(
                stage_id="executable_recipe",
                status="passed",
                metrics={"backend": "mock_adapter", "maturity": "smoke_passed"},
            ),
        ]
        return stages, {"recipe_id": "mock-recipe", "snapshot_hash": "mock-snapshot"}

    def finalize(self, root: Path, *, recipe_id: str, paired_result):
        self.calls.append("policy_memory")
        return CertificationStage(
            stage_id="policy_memory_update",
            status="passed",
            metrics={
                "recipe_id": recipe_id,
                "paired_result_hash": paired_result.result_hash,
            },
        )

    def evaluate(
        self, *, run: BackendRun, data_yaml: Path, workdir: Path, device: str
    ) -> BackendEvaluation:
        output = workdir / "mock_eval" / run.node_id
        output.mkdir(parents=True, exist_ok=True)
        baseline = run.candidate_id.startswith("baseline")
        gain = (
            0.0
            if baseline
            else {
                "reduce_mosaic": 0.03,
                "increase_box_gain": 0.02,
                "reduce_cls_gain": 0.01,
                "small_object_sampling": self.small_object_gain,
            }.get(run.candidate_id, 0.0)
        )
        eval_path = output / "coco_eval.json"
        eval_path.write_text(
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
            (data_yaml.parent / "annotations" / "instances_val2017.json").read_text(
                encoding="utf-8"
            )
        )
        detected_images = 2 if baseline else 4
        predictions_payload = [
            {
                "image_id": item["image_id"],
                "category_id": item["category_id"],
                "bbox": item["bbox"],
                "score": 0.9,
            }
            for item in annotations["annotations"]
            if int(item["image_id"]) <= detected_images
        ]
        predictions = output / "predictions.json"
        predictions.write_text(json.dumps(predictions_payload), encoding="utf-8")
        error_path = output / "coco_error_report.json"
        error_path.write_text(
            json.dumps(
                {
                    "false_negative_top_classes": [
                        {
                            "category_id": 1,
                            "name": "object",
                            "false_negative": 2 if baseline else self.small_object_fn,
                            "recall": 0.5 + gain,
                        }
                    ],
                    "background_false_positive_top_classes": [
                        {
                            "category_id": 1,
                            "name": "object",
                            "background_false_positive": 1,
                            "precision": 0.7 + gain,
                        }
                    ],
                    "localization_error_top_classes": [
                        {
                            "category_id": 1,
                            "name": "object",
                            "localization_error": 1,
                            "ap50": 0.5 + gain,
                        }
                    ],
                    "class_confusion_pairs": {},
                }
            ),
            encoding="utf-8",
        )
        return BackendEvaluation(
            eval_path=eval_path,
            predictions_path=predictions,
            error_report_path=error_path,
            latency_ms=10.0
            + (
                0.0
                if baseline
                else (
                    self.small_object_latency_delta
                    if run.candidate_id == "small_object_sampling"
                    else 0.1
                )
            ),
            model_size_mb=5.0,
            command=["mock-eval", run.node_id],
        )


MockGpuBackend.evaluate = MockPaperBackend.evaluate  # type: ignore[attr-defined]


def test_mini_coco_fixture_is_valid_and_deterministic(tmp_path: Path) -> None:
    data_yaml = create_mini_coco_fixture(tmp_path / "mini")
    first_manifest = load_mini_coco_fixture_manifest(data_yaml.parent)
    second_manifest = load_mini_coco_fixture_manifest(data_yaml.parent)

    assert first_manifest.fixture_hash == second_manifest.fixture_hash
    assert len(first_manifest.file_hashes) == 22
    assert len(first_manifest.fixture_hash) == 64
    config = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    annotations = json.loads(
        (data_yaml.parent / "annotations" / "instances_val2017.json").read_text(
            encoding="utf-8"
        )
    )

    assert config["train"] == "images/train2017"
    assert config["val"] == "images/val2017"
    assert len(list((data_yaml.parent / "images" / "train2017").glob("*.png"))) == 6
    assert len(annotations["images"]) == 4
    assert len(annotations["annotations"]) == 4
    train_areas = []
    for label_path in sorted((data_yaml.parent / "labels" / "train2017").glob("*.txt")):
        values = [
            float(item) for item in label_path.read_text(encoding="utf-8").split()
        ]
        train_areas.append(values[3] * values[4])
    assert any(area <= 0.01 for area in train_areas)
    assert any(area > 0.01 for area in train_areas)
    assert all(item["area"] < 32**2 for item in annotations["annotations"])


def test_suite_is_safe_without_explicit_gpu_opt_in(tmp_path: Path) -> None:
    backend = MockGpuBackend()
    report = RealGpuAcceptanceSuite(backend).run(workdir=tmp_path)

    assert report.status == "skipped"
    assert backend.train_calls == []
    assert (tmp_path / "certification_report.yaml").is_file()


def test_suite_resolves_relative_workdir_before_building_backend_paths(
    tmp_path: Path,
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.chdir(tmp_path)

    report = RealGpuAcceptanceSuite(MockGpuBackend()).run(
        workdir=Path("relative-certification")
    )

    assert Path(report.data_yaml).is_absolute()
    assert (
        Path(report.data_yaml).parent
        == (tmp_path / "relative-certification" / "mini_coco").resolve()
    )


def test_real_backend_reports_all_missing_certification_dependencies(
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    available = {"torch"}
    monkeypatch.setattr(
        certification_runner.importlib.util,
        "find_spec",
        lambda package: object() if package in available else None,
    )

    try:
        UltralyticsGpuBackend().environment()
    except RuntimeError as exc:
        message = str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("missing dependencies should fail certification preflight")

    assert "ultralytics, pycocotools" in message
    assert CERTIFICATION_INSTALL_COMMAND in message


def test_mock_backend_certifies_complete_mini_pipeline(tmp_path: Path) -> None:
    backend = MockGpuBackend()
    report = RealGpuAcceptanceSuite(backend).run(
        workdir=tmp_path, execute_real_gpu=True
    )

    assert report.status == "passed", report.failures
    assert report.asha_survivor == "reduce_mosaic"
    assert report.executed_recipe_id == "reduce_mosaic"
    assert report.executed_changed_variable == "mosaic"
    assert {stage.stage_id for stage in report.stages} >= {
        "train_entrypoint",
        "debug",
        "pilot_3_control",
        "post_eval",
        "error_facts",
        "paired_delta",
        "asha_decision",
        "pilot_10",
        "catalog_import",
        "snapshot_creation",
        "diagnosis_linked_paper_prior",
        "eligibility_gate",
        "executable_recipe",
        "policy_memory_update",
    }
    assert any(epochs == 10 for _, epochs in backend.train_calls)
    assert len(report.paired_result_hashes) == 4
    assert report.report_hash
    assert all(
        claim.recipe_id and claim.snapshot_hash and claim.evidence_hash
        for claim in report.capability_claims
    )


def test_small_object_sampling_certifies_runtime_diagnostics_and_matched_protocol(
    tmp_path: Path,
) -> None:
    backend = MockGpuBackend()

    report = RealGpuAcceptanceSuite(backend).run(
        workdir=tmp_path,
        execute_real_gpu=True,
        recipe_id="small_object_sampling",
    )

    assert report.status == "passed", report.failures
    assert report.asha_survivor == "small_object_sampling"
    assert report.executed_changed_variable == "data.sampling_policy"
    assert report.objective is not None and report.objective.passed
    assert report.objective.primary_metric == "ap_small"
    assert report.objective.target_metric_deltas["ap_small"] > 0
    assert report.objective.target_metric_deltas["per_class_ar/object"] > 0
    assert report.objective.target_error_fact_deltas["false_negative/object"] > 0
    assert report.objective.latency_guard_passed
    assert report.objective.model_size_guard_passed
    assert len(report.promotion_results) == 2
    assert all(item.passed for item in report.promotion_results)
    assert {
        "component_runtime_certification",
        "runtime_adapter",
        "paired_bootstrap",
        "promotion_gate",
    }.issubset({stage.stage_id for stage in report.stages})
    runtime_stage = next(
        stage for stage in report.stages if stage.stage_id == "runtime_adapter"
    )
    assert runtime_stage.metrics["payload_protocol_matched"] is True
    assert runtime_stage.metrics["manifest_protocol_matched"] is True
    assert runtime_stage.metrics["manifest_payload_matched"] is True
    assert runtime_stage.metrics["train_dataloader_hook_called"] is True
    assert backend.train_calls == [
        ("debug", 1),
        ("baseline_pilot_3", 3),
        ("small_object_sampling_pilot_3", 3),
        ("baseline_pilot_10", 10),
        ("small_object_sampling_pilot_10", 10),
    ]
    assert any(
        claim.capability_id == "small_object_sampling_runtime"
        and claim.local_reproduction == "locally_pilot_reproduced"
        for claim in report.capability_claims
    )
    pilot_3 = [
        item
        for item in backend.protocol_calls
        if item["node_id"] in {"baseline_pilot_3", "small_object_sampling_pilot_3"}
    ]
    pilot_10 = [
        item
        for item in backend.protocol_calls
        if item["node_id"] in {"baseline_pilot_10", "small_object_sampling_pilot_10"}
    ]
    assert len({item["protocol_hash"] for item in pilot_3}) == 1
    assert len({item["epochs"] for item in pilot_3}) == 1
    assert len({item["seed"] for item in pilot_3}) == 1
    assert len({item["protocol_hash"] for item in pilot_10}) == 1
    assert len({item["epochs"] for item in pilot_10}) == 1
    assert len({item["seed"] for item in pilot_10}) == 1
    loaded = CertificationReport.load_verified(tmp_path / "certification_report.yaml")
    assert loaded.report_hash == report.report_hash


def test_failed_small_object_certification_cannot_claim_pilot_reproduction(
    tmp_path: Path,
) -> None:
    backend = MockGpuBackend(
        small_object_gain=0.0,
        small_object_latency_delta=2.0,
        small_object_fn=2,
    )

    report = RealGpuAcceptanceSuite(backend).run(
        workdir=tmp_path,
        execute_real_gpu=True,
        recipe_id="small_object_sampling",
    )

    assert report.status == "failed"
    assert report.capability_claims == []
    assert report.promotion_results
    assert not report.promotion_results[0].passed
    assert "ap_small_improved" in report.promotion_results[0].rejection_reasons
    assert "false_negative_reduced" in report.promotion_results[0].rejection_reasons
    assert "latency_guard" in report.promotion_results[0].rejection_reasons
    assert not any(epochs == 10 for _, epochs in backend.train_calls)


def test_sampling_runtime_protocol_mismatch_blocks_before_promotion(
    tmp_path: Path,
) -> None:
    class ProtocolMismatchBackend(MockGpuBackend):
        def train(self, **kwargs):  # type: ignore[no-untyped-def]
            run = super().train(**kwargs)
            if run.candidate_id == "small_object_sampling":
                path = run.runtime_artifacts["sampler_manifest"]
                manifest = SmallObjectSamplingManifest.model_validate_json(
                    path.read_text(encoding="utf-8")
                )
                path.write_text(
                    manifest.model_copy(
                        update={"protocol_hash": "stale-protocol"}
                    ).model_dump_json(indent=2),
                    encoding="utf-8",
                )
            return run

    backend = ProtocolMismatchBackend()
    report = RealGpuAcceptanceSuite(backend).run(
        workdir=tmp_path,
        execute_real_gpu=True,
        recipe_id="small_object_sampling",
    )

    assert report.status == "failed"
    assert report.capability_claims == []
    assert "manifest_protocol_matched" in report.failures[0]
    assert not any(epochs == 10 for _, epochs in backend.train_calls)


def test_small_object_passed_report_rejects_missing_runtime_certification_stages() -> (
    None
):
    required = {
        "environment",
        "train_entrypoint",
        "debug",
        "pilot_3_control",
        "pilot_3_candidates",
        "post_eval",
        "error_facts",
        "paired_delta",
        "asha_decision",
        "pilot_10",
        "catalog_import",
        "snapshot_creation",
        "diagnosis_linked_paper_prior",
        "eligibility_gate",
        "executable_recipe",
        "policy_memory_update",
    }
    promotions = [
        CertificationPromotionResult(
            stage_id=stage_id,
            passed=True,
            primary_metric="ap_small",
        )
        for stage_id in ("pilot_3", "pilot_10")
    ]

    with pytest.raises(ValueError, match="missing required stages"):
        CertificationReport(
            certification_id="invalid-small-object",
            level="mini_gpu_pilot",
            status="passed",
            model="yolo26n.pt",
            data_yaml="coco.yaml",
            device="mock",
            protocol_hash="protocol",
            executed_recipe_id="small_object_sampling",
            stages=[
                CertificationStage(stage_id=stage_id, status="passed")
                for stage_id in sorted(required)
            ],
            objective=CertificationObjectiveResult(passed=True),
            promotion_results=promotions,
        )


def test_real_backend_builds_small_object_runtime_entrypoint_without_training(
    tmp_path: Path,
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    captured: dict[str, object] = {}

    def fake_run(command: list[str], log_path: Path) -> None:
        captured["command"] = command
        captured["log_path"] = log_path
        payload_path = Path(command[command.index("--payload") + 1])
        payload = AdapterRuntimePayload.read(payload_path)
        payload_path.parent.mkdir(parents=True, exist_ok=True)
        (payload_path.parent / "sampler_manifest.json").write_text(
            "{}", encoding="utf-8"
        )
        (payload_path.parent / "plugin_runtime_evidence.json").write_text(
            "{}", encoding="utf-8"
        )
        node_id = next(
            item.split("=", 1)[1]
            for item in payload.base_command
            if item.startswith("name=")
        )
        checkpoint = tmp_path / "ultralytics" / node_id / "weights" / "best.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(b"checkpoint")

    monkeypatch.setattr(certification_runner, "_run_command", fake_run)
    monkeypatch.setattr(certification_runner.shutil, "which", lambda name: "yolo.exe")
    data_yaml = create_mini_coco_fixture(tmp_path / "mini")

    run = UltralyticsGpuBackend().train(
        candidate_id="small_object_sampling",
        node_id="small_object_sampling_pilot_3",
        data_yaml=data_yaml,
        model="yolo26n.pt",
        workdir=tmp_path,
        device="0",
        epochs=3,
        seed=7,
        protocol_hash="protocol-1",
        overrides={"data.sampling_policy": {"fn_heavy_class_ids": [0]}},
    )

    command = captured["command"]
    assert isinstance(command, list)
    assert "yolo_agent.adapters.ultralytics.runtime_entrypoint" in command
    assert all(path.is_file() for path in run.runtime_artifacts.values())
    payload = AdapterRuntimePayload.read(run.runtime_artifacts["runtime_payload"])
    assert payload.protocol_hash == "protocol-1"
    assert payload.dataloader_plugin[0].options["fn_heavy_class_ids"] == [0]


def test_full_offline_state_machine_uses_mock_catalog_llm_adapter_and_gpu(
    tmp_path: Path,
) -> None:
    gpu = MockGpuBackend()
    paper = MockPaperBackend()
    report = RealGpuAcceptanceSuite(gpu, paper).run(
        workdir=tmp_path, execute_real_gpu=True
    )
    assert report.status == "passed"
    assert paper.calls == ["mock_catalog", "mock_llm", "mock_adapter", "policy_memory"]
    assert gpu.train_calls
    assert report.capability_claims[0].recipe_id == "mock-recipe"


def test_full_offline_certification_requires_complete_matched_protocol() -> None:
    required = {
        "environment",
        "train_entrypoint",
        "debug",
        "pilot_3_control",
        "pilot_3_candidates",
        "post_eval",
        "error_facts",
        "paired_delta",
        "asha_decision",
        "pilot_10",
        "catalog_import",
        "snapshot_creation",
        "diagnosis_linked_paper_prior",
        "eligibility_gate",
        "executable_recipe",
        "policy_memory_update",
    }
    objective = CertificationObjectiveResult(
        objective_hash="objective",
        required_delta=0.02,
        observed_delta=0.025,
        baseline_seeds=[1, 2, 3],
        candidate_seeds=[1, 2, 3],
        passed=True,
        dataset_manifest_hash="dataset",
        subset_manifest_hash="subset",
        seed_policy_hash="same-seed-policy",
        batch_policy_hash="same-batch-policy",
        ultralytics_version="8.4.mock",
        eval_protocol_hash="coco-post-eval-v1",
        paired_bootstrap_ci=(0.01, 0.04),
        cross_seed_confidence_interval=(0.012, 0.035),
        latency_regression=0.01,
        model_size_regression=0.0,
        latency_guard_passed=True,
        model_size_guard_passed=True,
    )
    report = CertificationReport(
        certification_id="offline-full",
        level="full_coco_multi_seed",
        status="passed",
        model="yolo26n.pt",
        data_yaml="coco.yaml",
        device="mock",
        protocol_hash="protocol",
        stages=[
            CertificationStage(stage_id=item, status="passed")
            for item in sorted(required)
        ],
        objective=objective,
    )
    assert report.status == "passed"
    assert report.objective.fixed_imgsz == 640
