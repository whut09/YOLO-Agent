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
    load_current_inventory_entries,
    load_exact_acceptance_report,
    load_paper_records_by_id,
    parse_readme_coverage,
    resolve_current_paper_metadata,
)
from yolo_agent.research.paper_execution_schemas import PaperExecutionSpec
from yolo_agent.research.paper_83_campaign_schemas import Paper83Manifest


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


def test_manifest_lineage_matches_the_readme_referenced_acceptance_report(
    production_manifest,
) -> None:
    declaration = parse_readme_coverage(README)

    assert (
        production_manifest.campaign.membership_source.acceptance_hash
        == declaration.acceptance_hash
    )
    assert (
        production_manifest.campaign.membership_source.metric_id
        == "compatible_papers_certified_adapter"
    )
    assert production_manifest.campaign.membership_source.source_registry_hash


def test_manifest_current_mapping_uses_paper_specific_routes_when_available(
    production_manifest,
) -> None:
    by_id = {item.paper_id: item for item in production_manifest.papers}

    assert by_id["arxiv:2303.13853"].current_component_ids == [
        "domain_adaptation.2303_13853"
    ]
    assert by_id["arxiv:2303.13853"].current_adapter_ids == [
        "domain_adaptation.2303_13853"
    ]
    assert by_id["ecva:eccv2022:2285"].current_disposition == "evidence_recovery"


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


def test_missing_acceptance_hash_reports_other_report_artifacts(tmp_path: Path) -> None:
    other_hash = "a" * 64
    report_path = tmp_path / "other-report.yaml"
    report_path.write_text(f"report_hash: {other_hash}\n", encoding="utf-8")

    with pytest.raises(Paper83CampaignError) as error:
        find_acceptance_artifact("b" * 64, search_roots=[tmp_path])

    message = str(error.value)
    assert "MATCHING_ACCEPTANCE_ARTIFACT=NOT_FOUND" in message
    assert f"OTHER_ACCEPTANCE_ARTIFACT={report_path}" in message
    assert f"REPORT_HASH={other_hash}" in message


def test_ambiguous_matching_acceptance_reports_fail_closed(tmp_path: Path) -> None:
    report_hash = "c" * 64
    (tmp_path / "a.yaml").write_text(f"report_hash: {report_hash}\n", encoding="utf-8")
    (tmp_path / "b.yaml").write_text(f"report_hash: {report_hash}\n", encoding="utf-8")

    with pytest.raises(Paper83CampaignError, match="AMBIGUOUS"):
        find_acceptance_artifact(report_hash, search_roots=[tmp_path])


def test_metadata_resolution_is_by_paper_id_not_title(production_manifest) -> None:
    target_id = production_manifest.papers[0].paper_id
    records = load_paper_records_by_id(ROOT / "research")
    records[target_id] = records[target_id].model_copy(
        update={"title": "unrelated replacement title"}
    )

    resolved = resolve_current_paper_metadata(
        [target_id],
        acceptance_report=_frozen_report(),
        paper_records=records,
        method_coverage=load_current_method_coverage(
            ROOT / "research" / "production" / "paper_method_coverage.yaml"
        ),
    )

    assert resolved[0].paper_id == target_id
    assert resolved[0].title == "unrelated replacement title"


def test_missing_exact_paper_id_is_not_recovered_by_title(production_manifest) -> None:
    target_id = production_manifest.papers[0].paper_id
    records = load_paper_records_by_id(ROOT / "research")
    records.pop(target_id)

    with pytest.raises(Paper83CampaignError, match=target_id):
        resolve_current_paper_metadata(
            [target_id],
            acceptance_report=_frozen_report(),
            paper_records=records,
            method_coverage=load_current_method_coverage(
                ROOT / "research" / "production" / "paper_method_coverage.yaml"
            ),
        )


def test_refreshed_inventory_replaces_stale_current_aliases() -> None:
    """A current paper route must not be unioned with an older generic alias."""

    target_id = "arxiv:2303.13853"
    records = load_paper_records_by_id(ROOT / "research")
    coverage = load_current_method_coverage(
        ROOT / "research" / "production" / "paper_method_coverage.yaml"
    )
    frozen = _frozen_report()
    current = PaperExecutionSpec(
        paper_id=target_id,
        profile_id=next(
            item.profile_id for item in coverage.profiles if item.paper_id == target_id
        ),
        title=records[target_id].title,
        source_locations=["fixture:refreshed-inventory"],
        canonical_component_ids=["domain_adaptation.2303_13853"],
        paper_specific_mechanism_ids=["domain_adaptation.2303_13853"],
        required_dataset_protocol={"imgsz": 640},
        required_evidence=["paper_specific_route"],
        execution_fingerprint="a" * 64,
        current_disposition="implementation_request",
        disposition_reason="fixture current route",
    )

    resolved = resolve_current_paper_metadata(
        [target_id],
        acceptance_report=frozen,
        paper_records=records,
        method_coverage=coverage,
        inventory_entries={target_id: current},
    )

    assert resolved[0].current_component_ids == current.canonical_component_ids
    assert "domain_adaptation.general" not in resolved[0].current_component_ids
    assert resolved[0].current_adapter_ids == []


def test_current_inventory_loader_keeps_exact_rows() -> None:
    """The checked-in audit inventory remains addressable by exact paper ID."""

    entries = load_current_inventory_entries(
        ROOT / "runs" / "coverage-audit" / "paper_execution_inventory.yaml"
    )

    if not entries:
        pytest.skip("the optional current inventory is not present")
    assert len(entries) == len(set(entries))


def test_unresolved_exact_route_cannot_inherit_runtime_ready_status() -> None:
    """A generic artifact must not hide an explicit identity-recovery route."""

    manifest = build_paper_83_manifest(repository_commit="test-current-route")
    paper = next(
        item for item in manifest.papers if item.paper_id == "ecva:eccv2022:2285"
    )

    assert paper.current_disposition == "evidence_recovery"


def test_manifest_rejects_duplicate_paper_ids(production_manifest) -> None:
    payload = production_manifest.model_dump(mode="json")
    payload["papers"][1]["paper_id"] = payload["papers"][0]["paper_id"]

    with pytest.raises(ValueError, match="sorted and unique"):
        Paper83Manifest.model_validate(payload)


def test_manifest_rejects_membership_hash_drift(production_manifest) -> None:
    payload = production_manifest.model_dump(mode="json")
    payload["campaign"]["membership_hash"] = "a" * 64

    with pytest.raises(ValueError, match="membership_hash"):
        Paper83Manifest.model_validate(payload)


def test_frozen_adapter_mapping_is_retained_per_paper(production_manifest) -> None:
    assert all(
        item.frozen_certified_adapter_ids for item in production_manifest.papers
    )
