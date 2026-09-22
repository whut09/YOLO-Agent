"""Delta-driven next-round portfolio (Prompt-18J §3).

The portfolio must consume the movement surface — a round that gained
AP_small but regressed background FP prioritises FP-addressing families
over piling on more small-object capacity.
"""

from __future__ import annotations

import pytest

from yolo_agent.agents.action_space import ActionCatalog
from yolo_agent.core.detection_error_delta import build_detection_error_delta
from yolo_agent.core.detection_error_profile import (
    DetectionErrorProfile,
    FalseNegativeFacts,
    FalsePositiveFacts,
    GlobalMetrics,
    ScaleMetrics,
)
from yolo_agent.core.error_next_round import build_next_round_portfolio
from yolo_agent.core.error_round_decision import decide_next_round
from yolo_agent.core.task_spec import MetricPriority, TaskSpec

_CATALOG = ActionCatalog.from_yaml("configs/actions/detection_action_catalog.yaml")


def _spec() -> TaskSpec:
    return TaskSpec(class_names=["person"], primary_metric=MetricPriority(name="map50_95"))


def _profile(
    profile_id: str,
    candidate_id: str,
    *,
    map50_95: float,
    ap_small: float,
    fn_total: int,
    fp_total: int,
    background_fp: int,
) -> DetectionErrorProfile:
    return DetectionErrorProfile(
        profile_id=profile_id,
        run_id="run-18j",
        candidate_id=candidate_id,
        gt_artifact="gt.json",
        dataset_manifest_hash="manifest-1",
        global_=GlobalMetrics(
            map50=map50_95 + 0.15,
            map50_95=map50_95,
            precision=0.7,
            recall=0.6,
        ),
        scale=ScaleMetrics(ap_small=ap_small, ap_medium=0.5, ap_large=0.6),
        false_negative=FalseNegativeFacts(total=fn_total),
        false_positive=FalsePositiveFacts(total=fp_total, background_fp=background_fp),
    )


def _small_fp_regression_delta():
    """AP_small strongly up, background FP up: the §3 example."""
    parent = _profile(
        "prof-parent", "baseline",
        map50_95=0.300, ap_small=0.250, fn_total=40, fp_total=50, background_fp=30,
    )
    candidate = _profile(
        "prof-cand", "cand",
        map50_95=0.310, ap_small=0.310, fn_total=32, fp_total=58, background_fp=40,
    )
    delta = build_detection_error_delta(candidate, parent)
    return candidate, delta


def test_fp_regression_round_ranks_fp_families_first() -> None:
    candidate, delta = _small_fp_regression_delta()
    decision = decide_next_round(delta, _spec())
    assert decision.decision == "refine"

    portfolio = build_next_round_portfolio(candidate, delta, decision, _CATALOG)

    assert portfolio.entries, "the catalog must yield eligible specs"
    top = portfolio.entries[0]
    # Hard-negative mining and friends address the regressed FP surface.
    assert top.family in {"sampling", "classification_loss", "threshold", "postprocess"}
    assert top.rationale.startswith("addresses regressed surface")
    ranked_zero = [e for e in portfolio.entries if e.rank < len(portfolio.entries)]
    assert ranked_zero[0].rationale.startswith("addresses regressed surface")
    # Deterministic order
    again = build_next_round_portfolio(candidate, delta, decision, _CATALOG)
    assert [e.action_id for e in again.entries] == [
        e.action_id for e in portfolio.entries
    ]


def test_portfolio_records_families_without_specs() -> None:
    candidate, delta = _small_fp_regression_delta()
    decision = decide_next_round(delta, _spec())

    portfolio = build_next_round_portfolio(candidate, delta, decision, _CATALOG)

    # The gap is recorded as actionable signal, not hidden.
    assert isinstance(portfolio.families_without_specs, list)


def test_portfolio_requires_budget_consuming_decision() -> None:

    parent = _profile(
        "prof-parent", "baseline",
        map50_95=0.30, ap_small=0.25, fn_total=10, fp_total=8, background_fp=5,
    )
    candidate = _profile(
        "prof-cand", "cand",
        map50_95=0.31, ap_small=0.30, fn_total=8, fp_total=7, background_fp=4,
    )
    delta = build_detection_error_delta(candidate, parent)
    decision = decide_next_round(delta, _spec())
    assert decision.decision == "promote"

    unmatched = delta.model_copy(update={"matched_evaluation": False})
    blocked = decide_next_round(unmatched, _spec())
    assert blocked.decision == "collect_evidence"

    with pytest.raises(ValueError, match="budget-consuming"):
        build_next_round_portfolio(candidate, delta, blocked, _CATALOG)


def test_portfolio_hypotheses_carry_real_evidence() -> None:
    candidate, delta = _small_fp_regression_delta()
    decision = decide_next_round(delta, _spec())

    portfolio = build_next_round_portfolio(candidate, delta, decision, _CATALOG)

    assert portfolio.hypotheses
    for hypothesis in portfolio.hypotheses:
        for link in hypothesis.evidence:
            assert link.fact_ids, "every evidence link cites real fact ids"
