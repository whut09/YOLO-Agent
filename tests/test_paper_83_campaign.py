from __future__ import annotations

from pathlib import Path

import pytest

from yolo_agent.research.paper_83_campaign import (
    Paper83CampaignError,
    assert_readme_campaign_shape,
    build_paper_83_manifest,
    calculate_membership_hash,
    extract_frozen_paper_ids,
    find_acceptance_artifact,
    load_current_method_coverage,
    load_exact_acceptance_report,
    load_paper_records_by_id,
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


@pytest.fixture(scope="module")
def production_manifest():
    return build_paper_83_manifest(repository_commit="test-paper-83")


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


def test_production_manifest_keeps_the_exact_historical_membership(
    production_manifest,
) -> None:
    expected = extract_frozen_paper_ids(_frozen_report())

    assert production_manifest.paper_count == 83
    assert [item.paper_id for item in production_manifest.papers] == expected
    assert len({item.paper_id for item in production_manifest.papers}) == 83


def test_every_frozen_id_has_exact_current_record_and_method_profile(
    production_manifest,
) -> None:
    records = load_paper_records_by_id(ROOT / "research")
    method = load_current_method_coverage(
        ROOT / "research" / "production" / "paper_method_coverage.yaml"
    )
    profile_ids = {item.paper_id for item in method.profiles}

    assert all(item.paper_id in records for item in production_manifest.papers)
    assert all(item.paper_id in profile_ids for item in production_manifest.papers)
    assert all(item.method_profile_id for item in production_manifest.papers)


def test_membership_hash_is_order_independent_and_metadata_independent() -> None:
    ids = extract_frozen_paper_ids(_frozen_report())

    assert calculate_membership_hash(ids) == calculate_membership_hash(list(reversed(ids)))
    assert calculate_membership_hash(ids) == calculate_membership_hash(tuple(ids))


def test_membership_hash_rejects_duplicates() -> None:
    with pytest.raises(ValueError, match="duplicate paper IDs"):
        calculate_membership_hash(["paper-a", "paper-a"])


def test_membership_hash_changes_when_membership_changes() -> None:
    ids = extract_frozen_paper_ids(_frozen_report())

    assert calculate_membership_hash(ids + ["paper-added"]) != calculate_membership_hash(ids)
    assert calculate_membership_hash(ids[:-1]) != calculate_membership_hash(ids)


def test_current_status_mutations_do_not_change_manifest_membership_hash(
    production_manifest,
) -> None:
    mutated = production_manifest.model_copy(
        update={
            "papers": [
                item.model_copy(
                    update={
                        "title": f"renamed:{item.title}",
                        "current_adapter_ids": [],
                        "current_disposition": "implementation_request",
                    }
                )
                for item in production_manifest.papers
            ]
        }
    )

    assert mutated.campaign.membership_hash == production_manifest.campaign.membership_hash
    assert calculate_membership_hash(
        [item.paper_id for item in mutated.papers]
    ) == production_manifest.campaign.membership_hash
