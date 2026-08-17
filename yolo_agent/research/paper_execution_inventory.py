"""Build a paper-level execution inventory from frozen method coverage."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from yolo_agent.research.method_profiles import (
    PaperImplementationDecision,
    PaperMethodCoverageReport,
    PaperMethodProfile,
)
from yolo_agent.research.schemas import PaperRecord


GENERIC_COMPONENT_IDS = frozenset(
    {
        "distillation.yolo26_teacher_student",
        "domain_adaptation.general",
    }
)


class PaperExecutionInventoryBuilder:
    """Create one execution record for every compatible paper ID."""

    def __init__(
        self,
        *,
        generic_component_ids: Iterable[str] = GENERIC_COMPONENT_IDS,
    ) -> None:
        self.generic_component_ids = frozenset(generic_component_ids)

    @staticmethod
    def compatible_method_pairs(
        method_coverage: PaperMethodCoverageReport,
        compatible_paper_ids: Iterable[str],
    ) -> list[tuple[PaperMethodProfile, PaperImplementationDecision]]:
        """Return sorted profile/decision pairs without dropping a paper ID."""
        profiles = {item.paper_id: item for item in method_coverage.profiles}
        decisions = {item.paper_id: item for item in method_coverage.decisions}
        requested = sorted(set(compatible_paper_ids))
        missing_profiles = sorted(set(requested) - set(profiles))
        missing_decisions = sorted(set(requested) - set(decisions))
        if missing_profiles or missing_decisions:
            failures = []
            if missing_profiles:
                failures.append("missing profiles: " + ", ".join(missing_profiles))
            if missing_decisions:
                failures.append("missing decisions: " + ", ".join(missing_decisions))
            raise ValueError("compatible paper coverage is incomplete; " + "; ".join(failures))
        return [(profiles[paper_id], decisions[paper_id]) for paper_id in requested]

    @staticmethod
    def paper_index(papers: Iterable[PaperRecord]) -> Mapping[str, PaperRecord]:
        """Index paper metadata and reject duplicate paper records."""
        indexed: dict[str, PaperRecord] = {}
        for paper in papers:
            if paper.paper_id in indexed:
                raise ValueError(f"duplicate paper metadata: {paper.paper_id}")
            indexed[paper.paper_id] = paper
        return indexed


__all__ = ["GENERIC_COMPONENT_IDS", "PaperExecutionInventoryBuilder"]
