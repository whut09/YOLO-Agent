from __future__ import annotations

import hashlib
from pathlib import Path

from yolo_agent.core.paper_implementation_gate import (
    PaperImplementationCampaignGate,
)
from yolo_agent.research.paper_83_campaign_schemas import Paper83Manifest
from yolo_agent.research.paper_implementation_schemas import (
    PaperImplementationRegistry,
    PaperImplementationSpec,
)


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "configs" / "research" / "paper_83_manifest.yaml"


def _registry(manifest: Paper83Manifest, *, not_ready: str | None = None) -> PaperImplementationRegistry:
    records: list[PaperImplementationSpec] = []
    for paper in manifest.papers:
        ready = paper.paper_id != not_ready
        records.append(
            PaperImplementationSpec(
                paper_id=paper.paper_id,
                manifest_membership_hash=manifest.campaign.membership_hash,
                method_profile_id=paper.method_profile_id,
                title=paper.title,
                year=paper.year,
                implementation_domain="fixture",
                mechanism_summary="paper-specific fixture mechanism",
                paper_specific_mechanisms=[f"fixture:{paper.paper_id}"],
                component_ids=["fixture.component"],
                adapter_ids=["fixture.adapter"],
                runtime_insertion_points=["fixture.insertion"],
                runtime_hooks=["fixture.hook"],
                paper_specific_config={"paper_id": paper.paper_id},
                paper_evidence_refs=["fixture:evidence"],
                unit_test_refs=["tests/test_fixture.py"],
                smoke_test_refs=["tests/test_fixture.py"],
                compatibility_test_refs=["tests/test_fixture.py"],
                implementation_fingerprint=hashlib.sha256(
                    paper.paper_id.encode("utf-8")
                ).hexdigest(),
                readiness="implementation_ready" if ready else "profiled",
                implementation_evidence_class="paper_specific",
                runtime_implementation_verified=ready,
                unit_tests_passed=ready,
                non_mock_smoke_passed=ready,
                compatibility_validation_passed=ready,
            )
        )
    return PaperImplementationRegistry(
        manifest_path=str(MANIFEST_PATH),
        manifest_membership_hash=manifest.campaign.membership_hash,
        paper_count=len(records),
        records=records,
    ).with_hash()


def test_current_repository_is_blocked_at_12_of_83_without_training() -> None:
    decision = PaperImplementationCampaignGate().evaluate(
        repository_root=ROOT,
    )

    assert not decision.allowed
    assert decision.paper_count == 83
    assert decision.implementation_ready_count == 12
    assert any("paper_not_implementation_ready:" in item for item in decision.blockers)


def test_complete_paper_registry_is_the_only_authorization_input(tmp_path: Path) -> None:
    manifest = Paper83Manifest.from_yaml(MANIFEST_PATH)
    registry_path = tmp_path / "paper_implementation_registry.yaml"
    _registry(manifest).to_yaml(registry_path, exclude_none=True, sort_keys=False)

    decision = PaperImplementationCampaignGate().evaluate(
        repository_root=tmp_path,
        manifest_path=MANIFEST_PATH,
        registry_path=registry_path,
    )

    assert decision.allowed
    assert decision.paper_count == decision.implementation_ready_count == 83
    assert decision.blockers == []


def test_one_not_ready_paper_blocks_the_first_real_training_run(tmp_path: Path) -> None:
    manifest = Paper83Manifest.from_yaml(MANIFEST_PATH)
    registry_path = tmp_path / "paper_implementation_registry.yaml"
    paper_id = manifest.papers[0].paper_id
    _registry(manifest, not_ready=paper_id).to_yaml(
        registry_path,
        exclude_none=True,
        sort_keys=False,
    )

    decision = PaperImplementationCampaignGate().evaluate(
        repository_root=tmp_path,
        manifest_path=MANIFEST_PATH,
        registry_path=registry_path,
    )

    assert not decision.allowed
    assert decision.implementation_ready_count == 82
    assert f"paper_not_implementation_ready:{paper_id}:profiled" in decision.blockers


def test_missing_registry_is_a_blocker_and_not_an_implicit_ready_state(tmp_path: Path) -> None:
    decision = PaperImplementationCampaignGate().evaluate(
        repository_root=tmp_path,
        manifest_path=MANIFEST_PATH,
        registry_path=tmp_path / "missing.yaml",
    )

    assert not decision.allowed
    assert decision.implementation_ready_count == 0
    assert "paper_implementation_registry_missing" in decision.blockers
