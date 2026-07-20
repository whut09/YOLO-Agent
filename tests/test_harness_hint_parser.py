"""Offline tests for non-executable harness hint parsing."""

from __future__ import annotations

from yolo_agent.research.harness_hint_parser import HarnessHintParser
from yolo_agent.research.schemas import ComponentTaxonomy, PaperProvenance, PaperRecord


def _paper(hints: list[str]) -> PaperRecord:
    return PaperRecord(
        paper_id="paper-hints",
        title="Mixed language paper",
        year=2025,
        component_ids=["small_object_sampling"],
        provenance=PaperProvenance(
            source_repository="awesome_object_detection",
            source_path="data/papers.json#paper-hints",
            source_record_hash="hash",
            importer_version="test.v1",
            original_harness_hints=hints,
        ),
    )


def test_harness_hints_extract_diagnostic_facts_without_actions() -> None:
    result = HarnessHintParser().parse(
        _paper(["当 AP_small low 时，check per-class AP and FN analysis because small object recall is weak; small_object_sampling is a candidate."]),
        ComponentTaxonomy(categories={}),
    )

    assert len(result.hints) == 1
    hint = result.hints[0]
    assert hint.symptom != "unknown"
    assert "ap_small" in hint.target_metrics
    assert "false_negative" in hint.target_error_facts
    assert hint.candidate_component_ids == ["small_object_sampling"]
    assert hint.evidence_level == "paper_claim"
    assert not hasattr(hint, "run_training")


def test_harness_hint_missing_fields_are_unknown_and_source_is_explicit() -> None:
    result = HarnessHintParser().parse(
        _paper(["A general observation without a metric."]),
        ComponentTaxonomy(categories={}),
    )

    hint = result.hints[0]
    assert hint.symptom == "unknown"
    assert hint.likely_cause == "unknown"
    assert hint.evidence_needed == ["unknown"]
    assert hint.source_location == "harness_hints[0]"


def test_harness_hint_parser_reports_replacement_characters_and_continues() -> None:
    result = HarnessHintParser().parse(
        _paper(["乱码 � AP_small low\x00"]),
        ComponentTaxonomy(categories={}),
    )

    assert result.hints
    assert any("encoding_replacement_character" in warning for warning in result.warnings)
    assert any("nul_character_removed" in warning for warning in result.warnings)


def test_harness_hint_parser_does_not_infer_unmentioned_component() -> None:
    result = HarnessHintParser().parse(
        _paper(["AP_small is low; collect AP_small evidence first."]),
        ComponentTaxonomy(categories={}),
    )

    assert result.hints[0].candidate_component_ids == []
