import json
from pathlib import Path

from yolo_agent.certification.inference_policy_runner import (
    InferencePolicyCertificationRunner,
)
from yolo_agent.components.adapters.inference.backend import BackendResult
from yolo_agent.components.adapters.inference.policy import InferencePolicyConfig


class _Backend:
    def __init__(self) -> None:
        self.calls = 0

    def run(self, images, protocol, *, category_ids):  # type: ignore[no-untyped-def]
        self.calls += 1
        assert images[0].image_id == 9
        assert category_ids == [3]
        assert protocol.config.standard_imgsz == 640
        return BackendResult(
            predictions=[
                {"image_id": 9, "category_id": 3, "bbox": [0, 0, 4, 4], "score": 0.8}
            ],
            latency_ms=12.0,
            throughput=83.3,
            peak_vram_mb=128.0,
            merge_statistics={
                "merge_policy": "none",
                "input_count": 1,
                "output_count": 1,
            },
        )


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    images = tmp_path / "images"
    images.mkdir()
    (images / "one.jpg").write_bytes(b"fixture")
    annotations = tmp_path / "instances.json"
    annotations.write_text(
        json.dumps(
            {
                "images": [{"id": 9, "file_name": "one.jpg"}],
                "categories": [{"id": 3, "name": "item"}],
                "annotations": [],
            }
        ),
        encoding="utf-8",
    )
    return images, annotations


def test_mock_backend_writes_complete_isolated_certification(tmp_path: Path) -> None:
    images, annotations = _fixture(tmp_path)
    backend = _Backend()
    runner = InferencePolicyCertificationRunner()
    config = InferencePolicyConfig(
        policy_id="tta",
        kind="test_time_augmentation",
        scales=[1.0, 1.2],
    )

    report = runner.run(
        workdir=tmp_path / "run",
        model="model.pt",
        images=images,
        annotations=annotations,
        config=config,
        standard_metrics={"map50_95": 0.4},
        execute=True,
        backend=backend,
        evaluator=lambda *_: {"AP": 0.42, "AP_small": 0.25, "AR_100": 0.61},
    )

    assert report.status == "passed"
    assert backend.calls == 1
    assert report.policy_metrics is not None
    assert report.policy_metrics.resources.peak_vram_mb == 128.0
    assert report.standard_640_metrics == {"map50_95": 0.4}
    assert Path(report.artifacts["metrics"]).name == "tta_metrics.json"
    assert (tmp_path / "run" / "inference_policy_certification_report.yaml").is_file()


def test_execution_is_explicit_and_does_not_call_backend(tmp_path: Path) -> None:
    images, annotations = _fixture(tmp_path)
    backend = _Backend()

    report = InferencePolicyCertificationRunner().run(
        workdir=tmp_path / "run",
        model="model.pt",
        images=images,
        annotations=annotations,
        config=InferencePolicyConfig(
            policy_id="thresholds",
            kind="class_aware_thresholding",
            class_thresholds={3: 0.4},
        ),
        execute=False,
        backend=backend,
    )

    assert report.status == "skipped"
    assert backend.calls == 0
    assert report.training_attribution_allowed is False
