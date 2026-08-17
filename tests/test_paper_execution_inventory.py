from __future__ import annotations

import pytest

from yolo_agent.research.method_profiles import (
    PaperImplementationDecision,
    PaperMethodCoverageReport,
    PaperMethodProfile,
)
from yolo_agent.research.paper_execution_inventory import (
    PaperExecutionInventoryBuilder,
)


def _profile(paper_id: str) -> PaperMethodProfile:
    return PaperMethodProfile(
        profile_id=f"profile:{paper_id}",
        paper_id=paper_id,
        source_locations=["paper_record"],
    )


def _decision(paper_id: str) -> PaperImplementationDecision:
    return PaperImplementationDecision(
        paper_id=paper_id,
        profile_id=f"profile:{paper_id}",
        decision="new_method_profile",
        reasons=["fixture"],
    )


def test_compatible_method_pairs_preserve_every_requested_paper() -> None:
    report = PaperMethodCoverageReport(
        paper_count=3,
        profile_count=3,
        profiles=[_profile("paper-c"), _profile("paper-a"), _profile("paper-b")],
        decisions=[_decision("paper-a"), _decision("paper-b"), _decision("paper-c")],
    )

    pairs = PaperExecutionInventoryBuilder.compatible_method_pairs(
        report,
        ["paper-c", "paper-a"],
    )

    assert [profile.paper_id for profile, _ in pairs] == ["paper-a", "paper-c"]


def test_compatible_method_pairs_reject_missing_decision() -> None:
    report = PaperMethodCoverageReport(
        paper_count=1,
        profile_count=1,
        profiles=[_profile("paper-a")],
        decisions=[],
    )

    with pytest.raises(ValueError, match="missing decisions: paper-a"):
        PaperExecutionInventoryBuilder.compatible_method_pairs(report, ["paper-a"])
