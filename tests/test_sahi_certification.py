from __future__ import annotations

import json
from pathlib import Path

from yolo_agent.certification.sahi_runner import SahiInferenceCertificationRunner
from yolo_agent.certification.sahi_schemas import SahiCertificationReport
from yolo_agent.components.adapters.inference.sliced_coco import SlicedCocoMetrics
from yolo_agent.components.adapters.inference.slicing import SlicingInferenceConfig


def _coco_fixture(tmp_path: Path) -> tuple[Path, Path]:
    images = tmp_path / "images"
    images.mkdir()
    (images / "one.jpg").write_bytes(b"image")
    annotations = tmp_path / "annotations.json"
    annotations.write_text(
        json.dumps(
            {
                "images": [{"id": 9, "file_name": "one.jpg", "width": 8, "height": 8}],
                "annotations": [],
                "categories": [{"id": 1, "name": "object"}],
            }
        ),
        encoding="utf-8",
    )
    return images, annotations


def test_certification_is_safe_and_skipped_by_default(tmp_path: Path) -> None:
    images, annotations = _coco_fixture(tmp_path)
    report = SahiInferenceCertificationRunner().run(
        workdir=tmp_path / "certification",
        model="yolo26n.pt",
        images=images,
        annotations=annotations,
        config=SlicingInferenceConfig(),
    )

    assert report.status == "skipped"
    assert report.training_attribution_allowed is False
    assert (tmp_path / "certification" / "sahi_certification_report.yaml").is_file()


def test_mock_certification_preserves_standard_and_sliced_namespaces(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    images, annotations = _coco_fixture(tmp_path)

    def backend(items, protocol):  # type: ignore[no-untyped-def]
        assert items[0].image_id == 9
        assert protocol.model_path == "yolo26n.pt"
        return [
            {"image_id": 9, "category_id": 1, "bbox": [0, 0, 2, 2], "score": 0.9}
        ], {"sliced_latency_ms": 31.0, "sliced_throughput": 32.25}

    monkeypatch.setattr(
        "yolo_agent.certification.sahi_runner.evaluate_sliced_coco",
        lambda **_: SlicedCocoMetrics(sliced_map50_95=0.43, sliced_ap_small=0.27),
    )
    standard = {"map50_95": 0.39, "ap_small": 0.20, "latency_ms": 11.5}
    workdir = tmp_path / "certification"
    report = SahiInferenceCertificationRunner().run(
        workdir=workdir,
        model="yolo26n.pt",
        images=images,
        annotations=annotations,
        config=SlicingInferenceConfig(slice_width=512, slice_height=512),
        standard_metrics=standard,
        execute=True,
        backend=backend,
    )

    assert report.status == "passed"
    assert report.standard_640_metrics == standard
    assert report.sliced_inference_metrics is not None
    assert report.sliced_inference_metrics.sliced_map50_95 == 0.43
    assert report.sliced_inference_metrics.sliced_ap_small == 0.27
    assert report.inference_policy_changed is True
    assert report.training_attribution_allowed is False
    restored = SahiCertificationReport.model_validate(
        report.model_dump(mode="json")
    )
    assert restored.report_hash == report.report_hash
    assert Path(report.artifacts["predictions"]).is_file()
    assert Path(report.artifacts["metrics"]).is_file()
