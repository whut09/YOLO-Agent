"""83-paper recipe coverage tests. No GPU training."""

from __future__ import annotations

from yolo_agent.recipes.paper_recipe_bindings import (
    EXISTING_SPECIFIC_RECIPES,
    load_certified_paper_recipe_specs,
)
from yolo_agent.recipes.paper_recipe_coverage import (
    UNRESOLVED_DISPOSITIONS,
    build_paper_recipe_coverage,
)
from yolo_agent.research.paper_mechanism_resolver import GENERIC_MECHANISM_IDS
from yolo_agent.research.paper_protocol_ids import CERTIFIED_PAPER_MECHANISMS


def test_coverage_binds_all_83_papers_without_silent_drop() -> None:
    report = build_paper_recipe_coverage()
    assert report.papers_total == 83
    assert report.paper_recipe_bindings_total == 83
    assert report.silent_drops == []
    assert {row.paper_id for row in report.bindings} == set(CERTIFIED_PAPER_MECHANISMS)
    for row in report.unresolved_bindings:
        assert row.disposition in UNRESOLVED_DISPOSITIONS
        assert row.reason_codes


def test_no_generic_recipe_covers_the_family() -> None:
    specs = load_certified_paper_recipe_specs()
    da_ids = {
        spec.paper_specific_mechanism_id
        for spec in specs
        if spec.paper_specific_mechanism_id.startswith("domain_adaptation.")
    }
    kd_ids = {
        spec.paper_specific_mechanism_id
        for spec in specs
        if spec.paper_specific_mechanism_id.startswith("distillation.")
    }
    assert "domain_adaptation.general" not in da_ids
    assert "distillation.yolo26_teacher_student" not in kd_ids
    assert len(da_ids) >= 30
    assert len(kd_ids) >= 30
    assert not set(GENERIC_MECHANISM_IDS) & {spec.paper_specific_mechanism_id for spec in specs}


def test_shared_real_implementation_keeps_full_paper_ids() -> None:
    ota = next(
        spec
        for spec in load_certified_paper_recipe_specs()
        if spec.recipe_id == "yolo26_ota_assignment_shadow"
    )
    assert set(ota.paper_ids) >= {"arxiv:2103.14259", "arxiv:2107.08430"}
    assert EXISTING_SPECIFIC_RECIPES["arxiv:2103.14259"]["shared_with"] == ["arxiv:2107.08430"]
