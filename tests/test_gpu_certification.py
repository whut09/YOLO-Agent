"""Offline tests for the opt-in real GPU certification orchestrator."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from yolo_agent.certification.fixture import create_mini_coco_fixture
from yolo_agent.certification.runner import BackendEvaluation, BackendRun, RealGpuAcceptanceSuite
from yolo_agent.certification.schemas import CertificationObjectiveResult, CertificationReport, CertificationStage


class MockGpuBackend:
    def __init__(self) -> None:
        self.train_calls: list[tuple[str, int]] = []

    def environment(self) -> dict[str, object]:
        return {"cuda_available": True, "gpu_name": "mock-gpu", "ultralytics_version": "8.4.mock"}

    def train_entrypoint(self, *, data_yaml: Path, model: str, workdir: Path, device: str) -> list[str]:
        return ["yolo-agent", "train", "--data", str(data_yaml), "--model", model, "--dry-run"]

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
        overrides: dict[str, object],
    ) -> BackendRun:
        self.train_calls.append((node_id, epochs))
        run_dir = workdir / "mock_runs" / node_id
        checkpoint = run_dir / "weights" / "best.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(b"mock checkpoint")
        return BackendRun(
            candidate_id=candidate_id,
            node_id=node_id,
            run_dir=run_dir,
            checkpoint=checkpoint,
            command=["mock-train", node_id, str(epochs)],
        )


class MockPaperBackend:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def prepare(self, root: Path):
        self.calls.extend(["mock_catalog", "mock_llm", "mock_adapter"])
        stages = [
            CertificationStage(stage_id="catalog_import", status="passed", metrics={"backend": "mock_catalog"}),
            CertificationStage(stage_id="snapshot_creation", status="passed", metrics={"snapshot_hash": "mock-snapshot"}),
            CertificationStage(stage_id="diagnosis_linked_paper_prior", status="passed", metrics={"backend": "mock_llm"}),
            CertificationStage(stage_id="eligibility_gate", status="passed", metrics={"eligible": True}),
            CertificationStage(stage_id="executable_recipe", status="passed", metrics={"backend": "mock_adapter", "maturity": "smoke_passed"}),
        ]
        return stages, {"recipe_id": "mock-recipe", "snapshot_hash": "mock-snapshot"}

    def finalize(self, root: Path, *, recipe_id: str, paired_result):
        self.calls.append("policy_memory")
        return CertificationStage(stage_id="policy_memory_update", status="passed", metrics={"recipe_id": recipe_id, "paired_result_hash": paired_result.result_hash})

    def evaluate(self, *, run: BackendRun, data_yaml: Path, workdir: Path, device: str) -> BackendEvaluation:
        output = workdir / "mock_eval" / run.node_id
        output.mkdir(parents=True, exist_ok=True)
        baseline = run.candidate_id.startswith("baseline")
        gain = 0.0 if baseline else {
            "reduce_mosaic": 0.03,
            "increase_box_gain": 0.02,
            "reduce_cls_gain": 0.01,
        }.get(run.candidate_id, 0.0)
        eval_path = output / "coco_eval.json"
        eval_path.write_text(json.dumps({
            "AP": 0.30 + gain,
            "AP50": 0.50 + gain,
            "AP75": 0.28 + gain,
            "AP_small": 0.20 + gain,
            "AP_medium": 0.32 + gain,
            "AP_large": 0.40 + gain,
            "AR_small": 0.25 + gain,
            "per_class_ap": {"object": 0.30 + gain},
            "per_class_ar": {"object": 0.40 + gain},
        }), encoding="utf-8")
        predictions = output / "predictions.json"
        predictions.write_text("[]", encoding="utf-8")
        error_path = output / "coco_error_report.json"
        error_path.write_text(json.dumps({
            "false_negative_top_classes": [
                {"category_id": 1, "name": "object", "false_negative": 2 if baseline else 1, "recall": 0.5 + gain}
            ],
            "background_false_positive_top_classes": [
                {"category_id": 1, "name": "object", "background_false_positive": 1, "precision": 0.7 + gain}
            ],
            "localization_error_top_classes": [
                {"category_id": 1, "name": "object", "localization_error": 1, "ap50": 0.5 + gain}
            ],
        }), encoding="utf-8")
        return BackendEvaluation(
            eval_path=eval_path,
            predictions_path=predictions,
            error_report_path=error_path,
            latency_ms=10.0 + (0.0 if baseline else 0.1),
            model_size_mb=5.0,
            command=["mock-eval", run.node_id],
        )


MockGpuBackend.evaluate = MockPaperBackend.evaluate  # type: ignore[attr-defined]


def test_mini_coco_fixture_is_valid_and_deterministic(tmp_path: Path) -> None:
    data_yaml = create_mini_coco_fixture(tmp_path / "mini")
    config = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    annotations = json.loads((data_yaml.parent / "annotations" / "instances_val2017.json").read_text(encoding="utf-8"))

    assert config["train"] == "images/train2017"
    assert config["val"] == "images/val2017"
    assert len(list((data_yaml.parent / "images" / "train2017").glob("*.png"))) == 6
    assert len(annotations["images"]) == 4
    assert len(annotations["annotations"]) == 4


def test_suite_is_safe_without_explicit_gpu_opt_in(tmp_path: Path) -> None:
    backend = MockGpuBackend()
    report = RealGpuAcceptanceSuite(backend).run(workdir=tmp_path)

    assert report.status == "skipped"
    assert backend.train_calls == []
    assert (tmp_path / "certification_report.yaml").is_file()


def test_mock_backend_certifies_complete_mini_pipeline(tmp_path: Path) -> None:
    backend = MockGpuBackend()
    report = RealGpuAcceptanceSuite(backend).run(workdir=tmp_path, execute_real_gpu=True)

    assert report.status == "passed", report.failures
    assert report.asha_survivor == "reduce_mosaic"
    assert {stage.stage_id for stage in report.stages} >= {
        "train_entrypoint", "debug", "pilot_3_control", "post_eval", "error_facts",
        "paired_delta", "asha_decision", "pilot_10",
        "catalog_import", "snapshot_creation", "diagnosis_linked_paper_prior",
        "eligibility_gate", "executable_recipe", "policy_memory_update",
    }
    assert any(epochs == 10 for _, epochs in backend.train_calls)
    assert len(report.paired_result_hashes) == 4
    assert report.report_hash
    assert all(claim.recipe_id and claim.snapshot_hash and claim.evidence_hash for claim in report.capability_claims)


def test_full_offline_state_machine_uses_mock_catalog_llm_adapter_and_gpu(tmp_path: Path) -> None:
    gpu = MockGpuBackend()
    paper = MockPaperBackend()
    report = RealGpuAcceptanceSuite(gpu, paper).run(workdir=tmp_path, execute_real_gpu=True)
    assert report.status == "passed"
    assert paper.calls == ["mock_catalog", "mock_llm", "mock_adapter", "policy_memory"]
    assert gpu.train_calls
    assert report.capability_claims[0].recipe_id == "mock-recipe"


def test_full_offline_certification_requires_complete_matched_protocol() -> None:
    required = {
        "environment", "train_entrypoint", "debug", "pilot_3_control", "pilot_3_candidates",
        "post_eval", "error_facts", "paired_delta", "asha_decision", "pilot_10",
        "catalog_import", "snapshot_creation", "diagnosis_linked_paper_prior",
        "eligibility_gate", "executable_recipe", "policy_memory_update",
    }
    objective = CertificationObjectiveResult(
        objective_hash="objective", required_delta=0.02, observed_delta=0.025,
        baseline_seeds=[1, 2, 3], candidate_seeds=[1, 2, 3], passed=True,
        dataset_manifest_hash="dataset", subset_manifest_hash="subset",
        seed_policy_hash="same-seed-policy", batch_policy_hash="same-batch-policy",
        ultralytics_version="8.4.mock", eval_protocol_hash="coco-post-eval-v1",
        paired_bootstrap_ci=(0.01, 0.04), cross_seed_confidence_interval=(0.012, 0.035),
        latency_regression=0.01, model_size_regression=0.0,
        latency_guard_passed=True, model_size_guard_passed=True,
    )
    report = CertificationReport(
        certification_id="offline-full", level="full_coco_multi_seed", status="passed",
        model="yolo26n.pt", data_yaml="coco.yaml", device="mock", protocol_hash="protocol",
        stages=[CertificationStage(stage_id=item, status="passed") for item in sorted(required)],
        objective=objective,
    )
    assert report.status == "passed"
    assert report.objective.fixed_imgsz == 640
