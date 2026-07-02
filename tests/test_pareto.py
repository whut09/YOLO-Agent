"""Pareto selector tests."""

from __future__ import annotations

from yolo_agent.agents.pareto import CandidateMetrics, ParetoSelector


def test_pareto_selector_keeps_accuracy_latency_tradeoffs() -> None:
    """Non-dominated front should keep distinct accuracy/latency/size tradeoffs."""
    candidates = [
        CandidateMetrics(candidate_id="fast", model="yolo26n + stal", accuracy=0.78, latency=8, model_size=5),
        CandidateMetrics(candidate_id="balanced", model="yolo11s + nwd + p2_head", accuracy=0.81, latency=12, model_size=12),
        CandidateMetrics(candidate_id="accurate", model="yolo12m + wiou", accuracy=0.85, latency=28, model_size=30),
        CandidateMetrics(candidate_id="dominated", model="slow_worse", accuracy=0.80, latency=20, model_size=20),
    ]

    front = ParetoSelector().select(candidates)

    assert [point.candidate_id for point in front.points] == ["accurate", "balanced", "fast"]
    assert front.dominated == ["dominated"]


def test_pareto_selector_handles_size_dominance() -> None:
    """A candidate worse on all comparable dimensions should be dominated."""
    front = ParetoSelector().select(
        [
            CandidateMetrics(candidate_id="small", model="small", accuracy=0.8, latency=10, model_size=5),
            CandidateMetrics(candidate_id="large", model="large", accuracy=0.8, latency=10, model_size=8),
        ]
    )

    assert [point.candidate_id for point in front.points] == ["small"]
    assert front.dominated == ["large"]

