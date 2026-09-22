"""Confirmation report: one model, two faithful renderings (§18K §7)."""

from __future__ import annotations

from pathlib import Path

import pytest

from yolo_agent.agents.confirmation_report import (
    ConfirmationReport,
    build_confirmation_report,
    write_confirmation_report,
)
from yolo_agent.agents.promotion_rule import (
    PromotionDecision,
    PromotionRule,
    apply_promotion_rule,
)
from yolo_agent.agents.seed_confirmation import SeedConfirmationState


def _confirmed_state() -> SeedConfirmationState:
    state = SeedConfirmationState.create(
        confirmation_id="conf-report", pilot_winner_id="trial-abc"
    )
    table = {
        "baseline": {
            "map50_95": [0.30, 0.30, 0.30],
            "precision": [0.70, 0.70, 0.70],
            "recall": [0.60, 0.60, 0.60],
            "latency_ms": [10.0, 10.0, 10.0],
        },
        "candidate": {
            "map50_95": [0.34, 0.35, 0.33],
            "precision": [0.72, 0.72, 0.72],
            "recall": [0.63, 0.63, 0.63],
            "latency_ms": [10.4, 10.4, 10.4],
        },
    }
    for role, metrics in table.items():
        for index, seed in enumerate(state.baseline_seeds):
            run = state.run_for(role, seed)  # type: ignore[arg-type]
            run.status = "completed"
            for name, series in metrics.items():
                run.metrics[name] = series[index]
    return state


def _decision(state: SeedConfirmationState) -> PromotionDecision:
    rule = PromotionRule.model_validate(
        {
            "target_delta": 0.01,
            "max_latency_regression": 0.05,
            "max_model_size_regression": None,
        }
    )
    return apply_promotion_rule(state, rule)


def test_report_yaml_roundtrips_and_pins_the_decision(tmp_path: Path) -> None:
    state = _confirmed_state()
    report = build_confirmation_report(state, _decision(state))

    path = tmp_path / "confirmation_report.yaml"
    report.to_yaml(path)

    loaded = ConfirmationReport.from_yaml(path)

    assert loaded.confirmation_id == "conf-report"
    assert loaded.decision.verdict == "CONFIRMED"
    assert loaded.primary.baseline_mean == 0.30
    assert loaded.primary.candidate_mean == 0.34
    assert loaded.primary.mean_delta is not None
    assert loaded.primary.ci95_low is not None
    assert loaded.primary.ci95_low < loaded.primary.mean_delta
    assert loaded.primary.ci95_high is not None


def test_markdown_contains_every_required_section() -> None:
    state = _confirmed_state()
    report = build_confirmation_report(state, _decision(state))

    markdown = report.to_markdown()

    # baseline mean / candidate mean / paired delta / CI (§7 list)
    assert "| mean | 0.3000 | 0.3400 | +0.0400 |" in markdown
    assert "95% CI" in markdown
    assert "Latency: 10.000 ms -> 10.400 ms" in markdown
    # decision
    assert "## Decision: CONFIRMED" in markdown
    # secondary metrics table
    assert "| precision |" in markdown
    assert "| recall |" in markdown


def test_missing_latency_is_reported_not_fabricated() -> None:
    state = _confirmed_state()
    for run in state.runs:
        run.metrics.pop("latency_ms", None)
    rule = PromotionRule.model_validate(
        {
            "target_delta": 0.01,
            "max_latency_regression": None,
            "max_model_size_regression": None,
        }
    )
    decision = apply_promotion_rule(state, rule)
    report = build_confirmation_report(state, decision)

    assert decision.verdict == "CONFIRMED"
    assert report.secondary["latency_ms"].recorded is False
    assert "Latency: not recorded" in report.to_markdown()
    assert "latency_ms" in report.missing_metrics


def test_write_emits_both_artifacts(tmp_path: Path) -> None:
    state = _confirmed_state()
    report = build_confirmation_report(state, _decision(state))

    yaml_path = tmp_path / "confirmation_report.yaml"
    md_path = tmp_path / "confirmation_report.md"
    write_confirmation_report(report, yaml_path, md_path)

    assert yaml_path.read_text(encoding="utf-8").strip()
    markdown = md_path.read_text(encoding="utf-8")
    assert markdown == report.to_markdown()
    assert ConfirmationReport.from_yaml(yaml_path).decision.verdict == "CONFIRMED"


def test_report_requires_decision_statistics() -> None:
    state = _confirmed_state()
    bare = PromotionDecision(
        verdict="INCONCLUSIVE", primary_metric="map50_95", target_delta=0.01
    )

    with pytest.raises(ValueError, match="carry its statistics"):
        build_confirmation_report(state, bare)
