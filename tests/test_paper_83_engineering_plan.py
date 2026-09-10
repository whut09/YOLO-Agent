from __future__ import annotations

import hashlib
from pathlib import Path

from yolo_agent.research.paper_83_campaign_schemas import Paper83Manifest
from yolo_agent.research.paper_83_engineering_plan import (
    Paper83EngineeringPlanBuilder,
    render_paper_83_engineering_plan_markdown,
)
from yolo_agent.research.paper_implementation_schemas import (
    PaperImplementationRegistry,
    PaperImplementationSpec,
)


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "configs" / "research" / "paper_83_manifest.yaml"


def _registry() -> PaperImplementationRegistry:
    manifest = Paper83Manifest.from_yaml(MANIFEST_PATH)
    records: list[PaperImplementationSpec] = []
    for index, paper in enumerate(manifest.papers):
        if index == 0:
            component = "domain_adaptation.general"
            specific: list[str] = []
            evidence = "generic_only"
            blockers = ["generic_only:paper_specific_mechanism_missing"]
        elif index == 1:
            component = "distillation.yolo26_teacher_student"
            specific = ["feature_distillation"]
            evidence = "paper_specific"
            blockers = ["missing_paper_config"]
        elif index == 2:
            component = "inference.sahi_slicing"
            specific = ["sahi_slicing"]
            evidence = "paper_specific"
            blockers = ["missing_test:compatibility:inference.sahi_slicing"]
        elif index == 3:
            component = "assigner.task_aligned"
            specific = ["task_aligned_assignment"]
            evidence = "paper_specific"
            blockers = ["component_smoke_evidence_missing:assigner.task_aligned"]
        else:
            component = "neck.rtmdet_large_kernel"
            specific = [f"paper_mechanism:{paper.paper_id}"]
            evidence = "paper_specific"
            blockers = []
        records.append(
            PaperImplementationSpec(
                paper_id=paper.paper_id,
                manifest_membership_hash=manifest.campaign.membership_hash,
                method_profile_id=paper.method_profile_id,
                title=paper.title,
                year=paper.year,
                implementation_domain="fixture",
                mechanism_summary="fixture paper mechanism",
                paper_specific_mechanisms=specific,
                shared_primitives=[component],
                component_ids=[component],
                adapter_ids=[component],
                runtime_insertion_points=["fixture_hook"],
                runtime_hooks=["fixture_runtime_hook"],
                paper_specific_config={"paper_id": paper.paper_id},
                required_assets=[],
                paper_evidence_refs=["fixture:paper"] if specific else [],
                unit_test_refs=["tests/test_fixture.py"],
                smoke_test_refs=["tests/test_fixture.py"],
                compatibility_test_refs=["tests/test_fixture.py"],
                implementation_fingerprint=hashlib.sha256(
                    paper.paper_id.encode("utf-8")
                ).hexdigest(),
                readiness="profiled",
                blockers=blockers,
                implementation_evidence_class=evidence,  # type: ignore[arg-type]
            )
        )
    return PaperImplementationRegistry(
        manifest_path=str(MANIFEST_PATH),
        manifest_membership_hash=manifest.campaign.membership_hash,
        paper_count=len(records),
        records=records,
    ).with_hash()


def _plan():  # type: ignore[no-untyped-def]
    return Paper83EngineeringPlanBuilder(
        manifest_path=MANIFEST_PATH,
        implementation_registry=_registry(),
    ).build()


def test_plan_keeps_exact_frozen_membership_and_batches_cover_every_paper() -> None:
    manifest = Paper83Manifest.from_yaml(MANIFEST_PATH)
    plan = _plan()

    assert plan.paper_count == 83
    assert plan.manifest_paper_ids == [item.paper_id for item in manifest.papers]
    assert [item.paper_id for item in plan.papers] == plan.manifest_paper_ids
    batched_ids = [paper_id for batch in plan.batches for paper_id in batch.paper_ids]
    assert sorted(batched_ids) == sorted(plan.manifest_paper_ids)
    assert len(batched_ids) == len(set(batched_ids)) == 83
    assert all(4 <= batch.paper_count <= 8 for batch in plan.batches)


def test_generic_mapping_creates_a_gap_not_a_paper_implementation() -> None:
    plan = _plan()
    first = plan.papers[0]

    assert first.paper_specific_mechanism_ids == []
    assert first.evidence_status == "blocked_missing_evidence"
    assert "paper_specific_mechanism_interpretation" in first.paper_specific_missing_parts
    assert "shared adapter mapping is not paper implementation" in first.known_facts


def test_plan_preserves_independent_mechanisms_and_current_status() -> None:
    plan = _plan()
    by_id = {item.paper_id: item for item in plan.papers}
    manifest = Paper83Manifest.from_yaml(MANIFEST_PATH)

    assert by_id[manifest.papers[1].paper_id].implementation_domain == "distillation"
    assert by_id[manifest.papers[2].paper_id].implementation_domain == "inference"
    assert by_id[manifest.papers[3].paper_id].implementation_domain == "assignment"
    assert (
        by_id[manifest.papers[1].paper_id].current_disposition
        == manifest.papers[1].current_disposition
    )


def test_plan_identity_is_stable_and_markdown_is_explicit_about_scope() -> None:
    first = _plan()
    second = _plan()

    assert first.plan_identity_hash == second.plan_identity_hash
    markdown = render_paper_83_engineering_plan_markdown(first)
    assert "not a reproduction or training result" in markdown
    assert "Shared primitives are dependencies only" in markdown
