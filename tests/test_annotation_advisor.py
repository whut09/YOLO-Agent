"""Annotation advisor tests."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from yolo_agent.agents.annotation_advisor import AnnotationAdvisor
from yolo_agent.cli import COMMANDS, main
from yolo_agent.core.label_quality import analyze_label_quality


def _make_dataset(root: Path) -> tuple[Path, Path]:
    image_dir = root / "images" / "train"
    label_dir = root / "labels" / "train"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    for name in ["img1.jpg", "img2.jpg", "img3.jpg"]:
        (image_dir / name).write_bytes(b"")

    (label_dir / "img1.txt").write_text(
        "\n".join(
            [
                "0 0.5 0.5 0.2 0.2",
                "1 0.2 0.2 0.0005 0.4",
            ]
        ),
        encoding="utf-8",
    )
    (label_dir / "img2.txt").write_text("", encoding="utf-8")
    (label_dir / "img3.txt").write_text("1 0.7 0.7 0.2 0.2\n", encoding="utf-8")

    data_yaml = root / "data.yaml"
    data_yaml.write_text(
        "\n".join(
            [
                "path: .",
                "train: images/train",
                "names:",
                "  - defect",
                "  - scratch",
                "  - missing_class",
                "",
            ]
        ),
        encoding="utf-8",
    )
    predictions_path = root / "predictions.yaml"
    predictions_path.write_text(
        yaml.safe_dump(
            {
                "predictions": [
                    {
                        "image": "images/train/img2.jpg",
                        "boxes": [
                            {
                                "class_id": 0,
                                "confidence": 0.95,
                                "x_center": 0.4,
                                "y_center": 0.4,
                                "width": 0.2,
                                "height": 0.2,
                            }
                        ],
                    },
                    {
                        "image": "images/train/img3.jpg",
                        "boxes": [
                            {
                                "class_id": 0,
                                "confidence": 0.9,
                                "x_center": 0.7,
                                "y_center": 0.7,
                                "width": 0.2,
                                "height": 0.2,
                            }
                        ],
                    },
                ]
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return data_yaml, predictions_path


def test_label_quality_finds_missing_labels_bad_boxes_and_confusion(tmp_path: Path) -> None:
    """Quality analysis should turn predictions and labels into review signals."""
    data_yaml, predictions = _make_dataset(tmp_path / "dataset")

    report = analyze_label_quality(data_yaml, predictions)

    issue_types = {issue.issue_type for issue in report.issues}
    assert "suspected_missing_label" in issue_types
    assert "suspicious_box_geometry" in issue_types
    assert "class_confusion" in issue_types
    assert "low_class_coverage" in issue_types
    assert report.class_confusions == {"scratch->defect": 1}
    assert "missing_class" in report.low_coverage_classes
    assert report.suspicious_missing_labels[0].image == "images/train/img2.jpg"
    assert any("abnormal size" in item or "abnormal" in item for item in report.recommendations)


def test_annotation_advisor_outputs_worklists(tmp_path: Path) -> None:
    """Advisor should summarize quality signals into annotation worklists."""
    data_yaml, predictions = _make_dataset(tmp_path / "dataset")

    advice = AnnotationAdvisor().advise(data_yaml, predictions)

    assert "missing_class" in advice.classes_to_collect
    assert "images/train/img2.jpg" in advice.samples_for_review
    assert advice.boxes_to_redraw
    assert advice.labeling_tool_targets == advice.samples_for_review
    assert any("Label Studio or CVAT" in item for item in advice.recommendations)


def test_advise_labels_cli_writes_reports(tmp_path: Path) -> None:
    """The advise-labels CLI should write JSON and Markdown reports."""
    assert "advise-labels" in COMMANDS
    data_yaml, predictions = _make_dataset(tmp_path / "dataset")
    out_prefix = tmp_path / "runs" / "annotation_advice"

    exit_code = main(
        [
            "advise-labels",
            "--data",
            str(data_yaml),
            "--predictions",
            str(predictions),
            "--out",
            str(out_prefix),
        ]
    )

    assert exit_code == 0
    json_path = out_prefix.with_suffix(".json")
    markdown_path = out_prefix.with_suffix(".md")
    assert json_path.exists()
    assert markdown_path.exists()
    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert data["samples_for_review"]
    assert "Annotation Advice" in markdown_path.read_text(encoding="utf-8")
