"""Offline tests for summary and Markdown note evidence extraction."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from yolo_agent.research.awesome_snapshot_builder import AwesomeSnapshotBuilder
from yolo_agent.research.harness_hint_parser import PaperDiagnosticHint
from yolo_agent.research.note_parser import PaperEvidenceSummary, PaperNoteParser
from yolo_agent.research.paper_registry import PaperRegistry
from yolo_agent.research.production_pipeline import ResearchProductionPipeline
from yolo_agent.research.schemas import ComponentTaxonomy, PaperProvenance, PaperRecord


def _paper(note_path: str | None = None) -> PaperRecord:
    return PaperRecord(
        paper_id="paper-note",
        title="Small object method",
        abstract="We propose the Small Object Sampler method on COCO. It improves AP_small by +1.2.",
        year=2025,
        datasets=["COCO"],
        detector_family="yolo26",
        component_ids=["small_object_sampling"],
        provenance=PaperProvenance(
            source_repository="awesome_object_detection",
            source_path="data/papers.json#paper-note",
            source_record_hash="hash",
            importer_version="test.v1",
            original_harness_hints=["When AP_small is low, check per-class AP."],
            original_note_path=note_path,
        ),
    )


def test_note_parser_extracts_mixed_language_metrics_formula_and_table(tmp_path: Path) -> None:
    note = tmp_path / "note.md"
    note.write_text(
        """## Method
提出 Small Object Sampler 方法，在 neck 中使用，baseline: YOLO26n.
Training cost: 2 GPU hours. Inference cost: latency +1ms. 公式 $L = L_cls + L_box$.

## Ablation
| Variant | AP_small | latency_ms |
| --- | ---: | ---: |
| baseline | 0.20 | 4.1 |
| sampler | 0.21 | 4.2 |

## Limitation
However, the method cannot evaluate domain shift.
""",
        encoding="utf-8",
    )

    result = PaperNoteParser().parse(
        paper=_paper(note.name),
        taxonomy=ComponentTaxonomy(categories={}),
        note_path=note,
    )

    assert result.status == "parsed"
    assert result.method_claims
    assert any("small_object_sampling" in item.component_ids for item in result.method_claims)
    assert any(item.dataset == "COCO" for item in result.method_claims)
    assert any(item.model_family == "yolo26" for item in result.method_claims)
    assert any(item.training_cost == "2 GPU hours" for item in result.method_claims)
    assert any(item.inference_cost == "latency +1ms" for item in result.method_claims)
    assert all(item.evidence_level == "paper_claim" for item in result.method_claims)
    assert result.explicit_claims
    assert any(item.claim_type == "metric" and item.evidence_level == "paper_claim" for item in result.explicit_claims)
    assert any(item.claim_type == "formula" and item.evidence_level == "paper_claim" for item in result.explicit_claims)
    assert len(result.ablation_hints) == 2
    assert all(item.source_location.startswith("note:table:1") for item in result.ablation_hints)
    assert result.limitations[0].evidence_level == "paper_claim"
    assert result.diagnostic_hints[0].evidence_level == "paper_claim"


def test_note_parser_writes_research_decision_ledger(tmp_path: Path) -> None:
    ledger = tmp_path / "research_decision_ledger.jsonl"
    result = PaperNoteParser(ledger_path=ledger).parse(
        paper=_paper(),
        taxonomy=ComponentTaxonomy(categories={}),
    )

    assert result.status == "parsed"
    record = json.loads(ledger.read_text(encoding="utf-8").splitlines()[0])
    assert record["paper_id"] == "paper-note"
    assert record["decision_type"] == "paper_evidence_extraction"
    assert record["output"]["evidence_level"] == "paper_claim"


def test_note_parser_missing_note_is_non_blocking_and_uses_summary() -> None:
    result = PaperNoteParser().parse(
        paper=_paper("missing-note.md"),
        taxonomy=ComponentTaxonomy(categories={}),
    )

    assert result.status == "partial"
    assert result.method_claims
    assert any("note_read_failed" in warning for warning in result.warnings) or any(
        "note_path_requires_catalog_root" in warning for warning in result.warnings
    )


def test_note_parser_handles_garbled_utf8_and_keeps_claims(tmp_path: Path) -> None:
    note = tmp_path / "garbled.md"
    note.write_bytes("乱码 � AP_small=0.21\x00".encode("utf-8") + b"\xff")

    result = PaperNoteParser().parse(
        paper=_paper(note.name),
        taxonomy=ComponentTaxonomy(categories={}),
        note_path=note,
    )

    assert result.status == "partial"
    assert result.explicit_claims
    assert any("encoding_replacement_character" in warning for warning in result.warnings)
    assert any("nul_character_removed" in warning for warning in result.warnings)


def test_optional_llm_enricher_failure_does_not_block_rules() -> None:
    def fail(_paper: PaperRecord, _summary: PaperEvidenceSummary) -> PaperEvidenceSummary:
        raise RuntimeError("mock LLM unavailable")

    result = PaperNoteParser(llm_enricher=fail).parse(
        paper=_paper(),
        taxonomy=ComponentTaxonomy(categories={}),
    )

    assert result.status == "partial"
    assert result.method_claims
    assert any("llm_enrichment_failed" in warning for warning in result.warnings)


def test_optional_llm_cannot_add_ungrounded_claims() -> None:
    def invent(_paper: PaperRecord, summary: PaperEvidenceSummary) -> PaperEvidenceSummary:
        fabricated = PaperDiagnosticHint(
            symptom="fabricated symptom",
            source_location="summary",
        )
        return summary.model_copy(update={"diagnostic_hints": [*summary.diagnostic_hints, fabricated]})

    result = PaperNoteParser(llm_enricher=invent).parse(
        paper=_paper(),
        taxonomy=ComponentTaxonomy(categories={}),
    )

    assert result.status == "partial"
    assert all(item.symptom != "fabricated symptom" for item in result.diagnostic_hints)
    assert any("ungrounded diagnostic_hints" in warning for warning in result.warnings)


def test_unexpected_parser_failure_does_not_block_snapshot(tmp_path: Path) -> None:
    class ExplodingParser:
        def parse(self, **kwargs):  # type: ignore[no-untyped-def]
            raise RuntimeError("broken parser")

    root = tmp_path / "research"
    PaperRegistry(root).add(_paper())

    result = ResearchProductionPipeline(root, note_parser=ExplodingParser()).run()  # type: ignore[arg-type]

    assert result.status == "completed", result.errors
    evidence = json.loads((root / "production" / "paper_evidence_summaries.jsonl").read_text(encoding="utf-8"))
    assert evidence["status"] == "failed"
    assert any("parser_failed_non_blocking" in warning for warning in evidence["warnings"])


def test_awesome_snapshot_build_parses_local_note_without_network(tmp_path: Path) -> None:
    source_root = tmp_path / "awesome"
    (source_root / "data").mkdir(parents=True)
    (source_root / "notes").mkdir(parents=True)
    (source_root / "notes" / "paper-note.md").write_text(
        "## Method\nWe propose Small Object Sampler on COCO; AP_small=0.21.\n",
        encoding="utf-8",
    )
    (source_root / "data" / "papers.json").write_text(
        json.dumps([{
            "paper_id": "paper-note",
            "title": "Small object method",
            "year": 2025,
            "summary": "Small object prior.",
            "component_ids": ["small_object_sampling"],
            "note_path": "notes/paper-note.md",
            "harness_hints": ["When AP_small is low, check per-class AP."],
        }]),
        encoding="utf-8",
    )

    result = AwesomeSnapshotBuilder(tmp_path / "research").build(
        source=source_root,
        source_commit="note-commit",
    )
    second = AwesomeSnapshotBuilder(tmp_path / "research-copy").build(
        source=source_root,
        source_commit="note-commit",
    )

    assert result.status == "completed", result.errors
    assert second.status == "completed", second.errors
    assert result.snapshot_hash == second.snapshot_hash
    evidence = [
        json.loads(line)
        for line in (tmp_path / "research" / "production" / "paper_evidence_summaries.jsonl").read_text(encoding="utf-8").splitlines()
    ][0]
    assert evidence["status"] == "parsed"
    assert evidence["diagnostic_hints"][0]["evidence_level"] == "paper_claim"
    ledger_line = (tmp_path / "research" / "production" / "research_decision_ledger.jsonl").read_text(encoding="utf-8").splitlines()[0]
    assert json.loads(ledger_line)["decision_type"] == "paper_evidence_extraction"
    snapshot = yaml.safe_load((tmp_path / "research" / "latest_snapshot.yaml").read_text(encoding="utf-8"))
    assert snapshot["snapshot_hash"] == result.snapshot_hash
