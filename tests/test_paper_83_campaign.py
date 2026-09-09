from __future__ import annotations

from pathlib import Path

import pytest

from yolo_agent.research.paper_83_campaign import (
    Paper83CampaignError,
    assert_readme_campaign_shape,
    extract_frozen_paper_ids,
    find_acceptance_artifact,
    load_exact_acceptance_report,
    parse_readme_coverage,
)


ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
ACCEPTANCE = ROOT / "docs" / "paper-coverage-acceptance.yaml"


def _frozen_report():
    declaration = parse_readme_coverage(README)
    return load_exact_acceptance_report(
        find_acceptance_artifact(declaration.acceptance_hash, search_roots=[ROOT / "docs"]),
        declaration.acceptance_hash,
    )


def test_readme_declares_the_expected_historical_campaign() -> None:
    declaration = parse_readme_coverage(README)

    assert declaration.method_profile_numerator == 85
    assert declaration.method_profile_denominator == 85
    assert declaration.certified_adapter_numerator == 83
    assert declaration.certified_adapter_denominator == 85
    assert declaration.audit_snapshot_hash
    assert declaration.acceptance_hash
    assert_readme_campaign_shape(declaration)


def test_acceptance_artifact_is_found_by_report_hash_not_filename() -> None:
    declaration = parse_readme_coverage(README)

    found = find_acceptance_artifact(
        declaration.acceptance_hash,
        search_roots=[ROOT / "docs"],
    )

    assert found.resolve() == ACCEPTANCE.resolve()


def test_acceptance_report_hash_is_integrity_checked() -> None:
    declaration = parse_readme_coverage(README)
    report = load_exact_acceptance_report(ACCEPTANCE, declaration.acceptance_hash)

    assert report.report_hash == declaration.acceptance_hash
    assert report.calculate_hash() == report.report_hash


def test_frozen_numerator_is_exactly_83_ids_inside_85_ids() -> None:
    report = _frozen_report()
    metric = report.metrics["compatible_papers_certified_adapter"]
    paper_ids = extract_frozen_paper_ids(report)

    assert metric.metric_id == "compatible_papers_certified_adapter"
    assert len(paper_ids) == 83
    assert len(set(paper_ids)) == 83
    assert len(metric.denominator_ids) == 85
    assert set(paper_ids).issubset(metric.denominator_ids)


def test_readme_shape_change_fails_closed() -> None:
    declaration = parse_readme_coverage(README).model_copy(
        update={"certified_adapter_numerator": 82}
    )

    with pytest.raises(Paper83CampaignError, match="README_ACTUAL_CERTIFIED=82"):
        assert_readme_campaign_shape(declaration)
