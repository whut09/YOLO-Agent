"""In-memory query index for normalized research papers."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from yolo_agent.research.schemas import EvidenceLevel, PaperBenchmark, PaperRecord


_EVIDENCE_RANK: dict[EvidenceLevel, int] = {
    "paper_claim": 0,
    "official_code_available": 1,
    "externally_reproduced": 2,
    "locally_smoke_tested": 3,
    "locally_pilot_reproduced": 4,
    "locally_full_reproduced": 5,
    "confirmed_multi_seed": 6,
}


class PaperIndex:
    """Search papers and select preferred duplicate benchmark observations."""

    def __init__(self, papers: Iterable[PaperRecord] = ()) -> None:
        self.papers: list[PaperRecord] = list(papers)
        self._rebuild_maps()

    def replace(self, papers: Iterable[PaperRecord]) -> None:
        self.papers = list(papers)
        self._rebuild_maps()

    def list(
        self,
        *,
        year_from: int | None = None,
        year_to: int | None = None,
        task_family: str | None = None,
        detector_family: str | None = None,
        component: str | None = None,
        component_category: str | None = None,
        dataset: str | None = None,
        metric: str | None = None,
        framework: str | None = None,
        official_code: bool | None = None,
        license: str | None = None,
        evidence_level: EvidenceLevel | None = None,
        applicability: str | None = None,
    ) -> list[PaperRecord]:
        """Return papers matching all supplied filters."""
        values = self.papers
        if year_from is not None:
            values = [paper for paper in values if paper.year >= year_from]
        if year_to is not None:
            values = [paper for paper in values if paper.year <= year_to]
        if task_family:
            values = [paper for paper in values if _contains(paper.task_families, task_family)]
        if detector_family:
            values = [paper for paper in values if _equals(paper.detector_family, detector_family)]
        if component:
            values = [paper for paper in values if _contains(paper.component_ids, component)]
        if component_category:
            values = [
                paper
                for paper in values
                if any(claim.component_category == component_category for claim in paper.claimed_effects)
            ]
        if dataset:
            values = [paper for paper in values if _contains(paper.datasets, dataset)]
        if metric:
            values = [
                paper
                for paper in values
                if any(benchmark.metric_name.casefold() == metric.casefold() for benchmark in paper.benchmarks)
            ]
        if framework:
            values = [paper for paper in values if _equals(paper.framework, framework)]
        if official_code is not None:
            values = [paper for paper in values if bool(paper.official_code_url) is official_code]
        if license:
            values = [paper for paper in values if _equals(paper.code_license, license)]
        if evidence_level:
            values = [
                paper
                for paper in values
                if any(benchmark.evidence_level == evidence_level for benchmark in paper.benchmarks)
            ]
        if applicability:
            values = [paper for paper in values if paper.applicability == applicability]
        return sorted(values, key=lambda paper: (-paper.year, paper.title.casefold(), paper.paper_id))

    def get_preferred_benchmark(
        self,
        *,
        paper_id: str,
        dataset: str,
        model: str,
        metric_name: str,
        split: str = "val",
    ) -> PaperBenchmark | None:
        """Return the best benchmark for a paper/query key."""
        paper = next((item for item in self.papers if item.paper_id == paper_id), None)
        if paper is None:
            return None
        matches = [
            item
            for item in paper.benchmarks
            if item.dataset.casefold() == dataset.casefold()
            and item.model.casefold() == model.casefold()
            and item.metric_name.casefold() == metric_name.casefold()
            and item.split.casefold() == split.casefold()
        ]
        return max(matches, key=_benchmark_rank, default=None)

    def get_all_benchmarks(
        self,
        *,
        paper_id: str,
        dataset: str,
        model: str,
        metric_name: str,
        split: str = "val",
    ) -> list[PaperBenchmark]:
        """Return all duplicate observations in preference order."""
        preferred = self.get_preferred_benchmark(
            paper_id=paper_id,
            dataset=dataset,
            model=model,
            metric_name=metric_name,
            split=split,
        )
        paper = next((item for item in self.papers if item.paper_id == paper_id), None)
        if paper is None:
            return []
        matches = [
            item
            for item in paper.benchmarks
            if item.dataset.casefold() == dataset.casefold()
            and item.model.casefold() == model.casefold()
            and item.metric_name.casefold() == metric_name.casefold()
            and item.split.casefold() == split.casefold()
        ]
        return sorted(matches, key=_benchmark_rank, reverse=True) if preferred else []

    def _rebuild_maps(self) -> None:
        self.by_paper_id = {paper.paper_id: paper for paper in self.papers}
        self.by_year: dict[int, list[PaperRecord]] = defaultdict(list)
        for paper in self.papers:
            self.by_year[paper.year].append(paper)


def _benchmark_rank(benchmark: PaperBenchmark) -> tuple[int, int, str]:
    return (
        _EVIDENCE_RANK[benchmark.evidence_level],
        int(benchmark.verified),
        benchmark.schema_version,
    )


def _contains(values: list[str], expected: str) -> bool:
    normalized = expected.casefold()
    return any(value.casefold() == normalized for value in values)


def _equals(value: str | None, expected: str) -> bool:
    return value is not None and value.casefold() == expected.casefold()


__all__ = ["PaperIndex"]
