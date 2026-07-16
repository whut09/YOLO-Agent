"""Paper-intelligence core schema tests."""

from datetime import datetime, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from yolo_agent.research.schemas import (
    RESEARCH_SCHEMA_VERSION,
    ComponentTaxonomy,
    PaperBenchmark,
    PaperComponentClaim,
    PaperRecord,
)
from yolo_agent.resources import ResourcePaths


ROOT = Path(__file__).resolve().parents[1]


def _benchmark() -> PaperBenchmark:
    return PaperBenchmark(
        dataset="COCO2017",
        split="val2017",
        model="research-detector-n",
        metric_name="map50_95",
        value=0.421,
        imgsz=640,
        latency_ms=3.2,
        hardware="paper-reported GPU",
        training_epochs=300,
        source_location="Table 2",
        evidence_level="paper_claim",
    )


def _claim() -> PaperComponentClaim:
    return PaperComponentClaim(
        component_id="matching.example",
        component_category="matching",
        claimed_effect="Improves positive matching quality.",
        evidence_level="paper_claim",
        target_metrics=["map50_95"],
        target_error_types=["false_negative"],
        reported_delta={"map50_95": 0.006},
        baseline="research-detector-n",
        experiment_conditions={"dataset": "COCO2017", "imgsz": 640},
        confidence=0.5,
        limitations=["Paper result has not been locally reproduced."],
    )


def _paper() -> PaperRecord:
    return PaperRecord(
        paper_id="arxiv:0000.00000",
        doi="10.0000/example.paper",
        title="Example Detection Paper",
        abstract="A fixture used to validate research metadata.",
        year=2025,
        published_at=datetime(2025, 1, 2, tzinfo=timezone.utc),
        authors=["Researcher A", "Researcher B"],
        task_families=["object_detection"],
        detector_family="end_to_end_detector",
        source_url="https://example.invalid/api/paper",
        paper_url="https://example.invalid/paper.pdf",
        official_code_url="https://example.invalid/code",
        code_license="Apache-2.0",
        framework="pytorch",
        datasets=["COCO2017"],
        benchmarks=[_benchmark()],
        training_budget={"epochs": 300, "gpus": 8},
        claimed_effects=[_claim()],
        component_ids=["matching.example"],
        applicability="recipe_idea_only",
        source="manual_test",
        ingestion_version="test.v1",
    )


def test_paper_record_roundtrips_json_and_yaml(tmp_path: Path) -> None:
    paper = _paper()
    json_path = paper.to_json(tmp_path / "paper.json")
    yaml_path = paper.to_yaml(tmp_path / "paper.yaml")

    from_json = PaperRecord.model_validate_json(json_path.read_text(encoding="utf-8-sig"))
    from_yaml = PaperRecord.from_yaml(yaml_path)

    assert from_json == paper
    assert from_yaml == paper
    assert from_yaml.schema_version == RESEARCH_SCHEMA_VERSION
    assert from_yaml.benchmarks[0].evidence_level == "paper_claim"
    assert from_yaml.claimed_effects[0].evidence_level == "paper_claim"


def test_benchmark_requires_explicit_evidence_level() -> None:
    with pytest.raises(ValidationError, match="evidence_level"):
        PaperBenchmark(
            dataset="COCO2017",
            model="detector",
            metric_name="map50_95",
            value=0.4,
        )


def test_claim_cannot_be_promoted_to_local_evidence() -> None:
    payload = _claim().model_dump(mode="json")
    payload["evidence_level"] = "locally_pilot_reproduced"

    with pytest.raises(ValidationError, match="paper_claim"):
        PaperComponentClaim.model_validate(payload)


def test_schema_validates_ranges_and_required_text() -> None:
    with pytest.raises(ValidationError):
        PaperRecord(paper_id="", title="Paper", year=2025)
    with pytest.raises(ValidationError):
        PaperBenchmark(
            dataset="COCO2017",
            model="detector",
            metric_name="map50_95",
            value=0.4,
            latency_ms=-1,
            evidence_level="paper_claim",
        )
    with pytest.raises(ValidationError):
        PaperComponentClaim(
            component_id="component",
            claimed_effect="effect",
            evidence_level="paper_claim",
            confidence=1.1,
        )


def test_bundled_component_taxonomy_is_complete() -> None:
    taxonomy = ComponentTaxonomy.from_yaml(ResourcePaths.COMPONENT_TAXONOMY)
    required = {
        "backbone",
        "neck",
        "detection_head",
        "feature_pyramid",
        "attention",
        "convolution_block",
        "optimizer",
        "lr_schedule",
        "loss_schedule",
        "assigner",
        "matching",
        "positive_sample_selection",
        "bbox_regression_loss",
        "classification_loss",
        "quality_estimation",
        "distillation",
        "augmentation",
        "sampling",
        "label_quality",
        "active_learning",
        "threshold",
        "slicing",
        "tta",
        "calibration",
        "ensemble",
        "nms",
        "pretraining",
        "domain_adaptation",
    }

    assert set(taxonomy.categories) == required


def test_taxonomy_rejects_unknown_category() -> None:
    with pytest.raises(ValidationError):
        ComponentTaxonomy(categories={"unknown_component": []})


def test_research_schemas_reject_unknown_fields() -> None:
    payload = _paper().model_dump(mode="json")
    payload["trusted_local_metric"] = True

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        PaperRecord.model_validate(payload)
