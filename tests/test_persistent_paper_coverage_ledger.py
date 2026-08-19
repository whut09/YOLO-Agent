"""CPU-only acceptance tests for persistent 83-paper routing coverage."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from yolo_agent.agents.paper_proposal_ledger import (
    PaperCandidateCoverageLedger,
    planned_recipe_disposition,
)
from yolo_agent.agents.paper_proposal_schemas import CoverageBoundary
from yolo_agent.research.paper_execution_schemas import (
    PaperExecutionInventory,
    PaperExecutionSpec,
)


REQUIRED_BOUNDARIES: tuple[CoverageBoundary, ...] = (
    "inventory",
    "planner",
    "critic",
    "materialization_input",
    "round_execution_plan",
    "runtime_readiness",
    "asha_registration",
    "candidate_terminal",
)


def _inventory_from_coverage_fixture() -> PaperExecutionInventory:
    payload = yaml.safe_load(
        Path("tests/fixtures/paper_recipe_coverage.yaml").read_text(encoding="utf-8")
    )
    records = []
    for item in payload["unresolved_bindings"]:
        mechanism = item["paper_specific_mechanism_id"]
        records.append(
            PaperExecutionSpec(
                paper_id=item["paper_id"],
                profile_id=f"profile:{item['paper_id']}",
                title=f"Fixture {item['paper_id']}",
                source_locations=[
                    f"tests/fixtures/paper_recipe_coverage.yaml#{item['paper_id']}"
                ],
                canonical_component_ids=[mechanism],
                paper_specific_mechanism_ids=[mechanism],
                required_evidence=["target_error_facts"],
                recipe_ids=[item["recipe_id"]],
                execution_fingerprint=item["execution_fingerprint"],
                current_disposition="evidence_recovery",
                disposition_reason="target_error_facts_missing",
            )
        )
    return PaperExecutionInventory(
        source_method_coverage_hash="a" * 64,
        all_paper_count=728,
        compatible_paper_count=83,
        exact_reproduction_candidates=0,
        records=sorted(records, key=lambda item: item.paper_id),
    ).with_hash()


def test_all_83_papers_cross_every_persistent_coverage_boundary(
    tmp_path: Path,
) -> None:
    inventory = _inventory_from_coverage_fixture()
    ledger = PaperCandidateCoverageLedger(
        tmp_path / "paper_candidate_coverage.yaml",
        run_id="coverage-83",
        protocol_hash="protocol-current",
        dataset_manifest_hash="dataset-current",
    )
    seeded = ledger.seed_inventory(inventory)
    failed_paper = inventory.records[0]
    candidate_id = "paper-terminal-candidate"
    ledger.upsert(
        planned_recipe_disposition(
            run_id="coverage-83",
            round_index=1,
            recipe_id=failed_paper.recipe_ids[0],
            recipe_version="v1",
            component_ids=failed_paper.canonical_component_ids,
            decision="selected",
            reasons=[],
            related_papers=[failed_paper.paper_id],
            method_profile_ids=[failed_paper.profile_id],
            execution_fingerprint=failed_paper.execution_fingerprint,
            candidate_id=candidate_id,
            protocol_hash="protocol-current",
            dataset_manifest_hash="dataset-current",
        )
    )
    ledger.seal_boundary("planner")

    for boundary, source_stage in (
        ("critic", "recipe_critic"),
        ("materialization_input", "materialization_input"),
        ("round_execution_plan", "round_execution_plan"),
        ("runtime_readiness", "runtime_readiness"),
    ):
        ledger.update_candidate_disposition(
            candidate_id=candidate_id,
            disposition="queued",
            reason_codes=[f"{boundary}_passed"],
            source_stage=source_stage,
            node_id="node-paper-terminal-candidate",
        )
        ledger.seal_boundary(boundary)

    ledger.update_candidate_disposition(
        candidate_id=candidate_id,
        disposition="queued",
        reason_codes=["asha_trial_registered"],
        source_stage="asha_registration",
        node_id="node-paper-terminal-candidate",
        asha_trial_id="coverage-83:paper-terminal-candidate",
    )
    ledger.seal_boundary("asha_registration")
    before_failure = ledger.read().current_by_paper

    ledger.update_candidate_disposition(
        candidate_id=candidate_id,
        disposition="blocked_runtime",
        reason_codes=["adapter_runtime_failed"],
        source_stage="candidate_failure",
        node_id="node-paper-terminal-candidate",
        asha_trial_id="coverage-83:paper-terminal-candidate",
    )
    completed = ledger.seal_boundary("candidate_terminal")

    assert seeded.expected_paper_count == 83
    assert len(completed.paper_coverage) == 83
    assert len(completed.current_by_paper) == 83
    assert completed.current_by_paper[failed_paper.paper_id].disposition == "blocked_runtime"
    for paper_id, paper in completed.current_by_paper.items():
        boundaries = {event.boundary for event in paper.stage_history}
        assert boundaries == set(REQUIRED_BOUNDARIES)
        if paper_id != failed_paper.paper_id:
            assert paper.disposition == before_failure[paper_id].disposition
            assert paper.execution_fingerprint == before_failure[paper_id].execution_fingerprint
    for boundary in REQUIRED_BOUNDARIES:
        ledger.assert_boundary_complete(boundary)


def test_old_protocol_coverage_cannot_be_reused_as_current_evidence(
    tmp_path: Path,
) -> None:
    path = tmp_path / "paper_candidate_coverage.yaml"
    old = PaperCandidateCoverageLedger(
        path,
        run_id="coverage-83",
        protocol_hash="protocol-old",
        dataset_manifest_hash="dataset-current",
    )
    old.seed_inventory(_inventory_from_coverage_fixture())

    with pytest.raises(RuntimeError, match="protocol mismatch"):
        PaperCandidateCoverageLedger(
            path,
            run_id="coverage-83",
            protocol_hash="protocol-current",
            dataset_manifest_hash="dataset-current",
        ).read()
