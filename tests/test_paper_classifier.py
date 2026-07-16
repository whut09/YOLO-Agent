"""Offline tests for rule-based paper classification."""

from yolo_agent.research.paper_classifier import PaperClassifier, ResearchPriorityConfig
from yolo_agent.research.schemas import PaperBenchmark, PaperComponentClaim, PaperRecord


def make_paper(**updates: object) -> PaperRecord:
    payload: dict[str, object] = {
        "paper_id": "paper-assigner",
        "title": "Real-time COCO Object Detection with Task Aligned Assignment",
        "abstract": "We introduce a lightweight assigner and bbox regression loss, and report ablation, latency, FPS, model size, and AP_small.",
        "year": 2024,
        "task_families": ["object_detection"],
        "detector_family": "yolo",
        "official_code_url": "https://example.invalid/code",
        "code_license": "Apache-2.0",
        "framework": "pytorch",
        "datasets": ["COCO2017"],
        "component_ids": ["assigner.example"],
        "claimed_effects": [PaperComponentClaim(
            component_id="assigner.example",
            component_category="assigner",
            claimed_effect="Improves positive assignment.",
            target_metrics=["map50_95", "ap_small"],
            target_error_types=["small_object_miss"],
            evidence_level="paper_claim",
        )],
        "benchmarks": [PaperBenchmark(
            dataset="COCO2017",
            model="yolo26n",
            metric_name="map50_95",
            value=0.43,
            latency_ms=5.0,
            evidence_level="paper_claim",
        )],
        "applicability": "direct_adapter_candidate",
    }
    payload.update(updates)
    return PaperRecord.model_validate(payload)


def test_classifier_detects_categories_targets_and_high_relevance() -> None:
    result = PaperClassifier().classify(make_paper())

    assert result.paper_id == "paper-assigner"
    assert "assigner" in result.component_categories
    assert "bbox_regression_loss" in result.component_categories
    assert "small_object_miss" in result.target_error_types
    assert "map50_95" in result.target_metrics
    assert result.detector_family == "yolo"
    assert result.likely_yolo26_relevance == "high"
    assert result.applicability == "direct_adapter_candidate"
    assert result.priority_score > 0
    assert result.classifier == "rules.v1"
    assert result.llm_status == "not_used"


def test_classifier_marks_detr_as_separate_detector_family() -> None:
    result = PaperClassifier().classify(make_paper(
        paper_id="paper-detr",
        title="DETR Decoder for Open Vocabulary Detection",
        abstract="A transformer decoder with bipartite matching for open-vocabulary detection.",
        detector_family="detr",
        official_code_url=None,
        applicability="insufficient_information",
        component_ids=["matching.detr"],
    ))

    assert result.detector_family == "detr"
    assert result.applicability == "separate_detector_family"
    assert result.likely_yolo26_relevance in {"medium", "low", "unknown"}
    assert any("separate detector" in reason.lower() for reason in result.reasons)


def test_classifier_marks_incompatible_assumptions() -> None:
    result = PaperClassifier().classify(make_paper(
        title="Anchor and DFL-only Detector",
        abstract="This method requires anchors and a DFL-only regression head.",
        detector_family="anchor_detector",
        official_code_url=None,
    ))

    assert result.applicability == "incompatible"
    assert result.likely_yolo26_relevance == "unknown"
    assert result.priority_score >= 0


def test_classifier_distinguishes_no_metadata_from_low_cost_recipe() -> None:
    empty = PaperClassifier().classify(PaperRecord(
        paper_id="empty",
        title="A Study",
        year=2024,
    ))
    assert empty.applicability == "insufficient_information"
    assert empty.component_categories == []
    assert empty.likely_yolo26_relevance == "unknown"


def test_classifier_many_is_sorted_by_priority() -> None:
    classifier = PaperClassifier()
    papers = [make_paper(paper_id="a"), make_paper(
        paper_id="b",
        title="A Generic Study",
        abstract="A generic analysis without detector or COCO details.",
        official_code_url=None,
        component_ids=[],
        claimed_effects=[],
        benchmarks=[],
    )]

    results = classifier.classify_many(papers)
    assert [result.paper_id for result in results] == ["a", "b"]


def test_classifier_accepts_custom_rules_without_llm() -> None:
    config = ResearchPriorityConfig(
        weights={"coco_relevance": 100.0},
        category_keywords={"assigner": ["custom matching"]},
        error_keywords={"custom_error": ["custom error"]},
        metric_keywords={"custom_metric": ["custom metric"]},
        detector_family_keywords={},
        direct_adapter_keywords=["matching"],
    )
    result = PaperClassifier(config).classify(PaperRecord(
        paper_id="custom",
        title="Custom Matching",
        abstract="custom matching custom error custom metric",
        year=2024,
        applicability="insufficient_information",
    ))
    assert result.component_categories == ["assigner"]
    assert result.target_error_types == ["custom_error"]
    assert result.target_metrics == ["custom_metric"]
