from __future__ import annotations

import json
from pathlib import Path

import pytest

from yolo_agent.research.component_aliases import ComponentAliasResolver
from yolo_agent.research.method_profiles import PaperMethodProfileBuilder
from yolo_agent.research.note_parser import PaperEvidenceSummary
from yolo_agent.research.paper_method_evidence_report import (
    build_method_evidence_coverage_report,
)
from yolo_agent.research.schemas import PaperRecord


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(
    not (ROOT / "research" / "papers.jsonl").is_file(),
    reason="local Awesome catalog production artifact is not committed",
)
def test_all_local_catalog_papers_receive_one_evidence_profile_and_audit() -> None:
    papers = [
        PaperRecord.model_validate(json.loads(line))
        for line in (ROOT / "research" / "papers.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]
    evidence_path = (
        ROOT / "research" / "production" / "paper_evidence_summaries.jsonl"
    )
    summaries: dict[str, PaperEvidenceSummary] = {}
    if evidence_path.is_file():
        for line in evidence_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                item = PaperEvidenceSummary.model_validate(json.loads(line))
                summaries[item.paper_id] = item

    coverage = PaperMethodProfileBuilder(
        ComponentAliasResolver.from_yaml()
    ).build(papers, evidence_summaries=summaries)
    audit = build_method_evidence_coverage_report(coverage)

    assert len(papers) == 728
    assert coverage.paper_count == 728
    assert coverage.profile_count == 728
    assert len(coverage.decisions) == 728
    assert audit.audited_paper_count == 728
    assert len({item.paper_id for item in audit.audits}) == 728
    assert all(profile.structured_method_evidence is not None for profile in coverage.profiles)
