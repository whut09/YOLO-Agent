"""Three-seed confirmation report: machine YAML + human Markdown (Prompt-18K §7).

Both renderings come from one ``ConfirmationReport`` built out of the paired
statistics and the promotion decision, so the numbers in the Markdown a human
approves can never drift from the YAML a machine pins:

- baseline mean / candidate mean / paired delta / 95% CI on the primary,
- the secondary metrics recorded alongside it (precision, recall, AP_small),
- latency movement,
- the four-way decision with its reasons, violations and acknowledged gains.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from yolo_agent.agents.promotion_rule import PromotionDecision
from yolo_agent.agents.seed_confirmation import SeedConfirmationState
from yolo_agent.core.yaml_io import YAMLModelMixin

REPORT_SCHEMA_VERSION = "confirmation-report-v1"


class MetricReportRow(BaseModel):
    """One metric's baseline/candidate means and paired movement."""

    model_config = ConfigDict(extra="forbid")

    metric: str
    baseline_mean: float | None = None
    candidate_mean: float | None = None
    mean_delta: float | None = None
    ci95_low: float | None = None
    ci95_high: float | None = None
    #: False when the metric was not recorded on every run.
    recorded: bool = True


class ConfirmationReport(BaseModel, YAMLModelMixin):
    """The full confirmation outcome, renderable to YAML and Markdown."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = REPORT_SCHEMA_VERSION
    confirmation_id: str
    pilot_winner_id: str
    seeds: list[int] = Field(default_factory=list)
    primary_metric: str
    target_delta: float
    generated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    primary: MetricReportRow
    secondary: dict[str, MetricReportRow] = Field(default_factory=dict)
    missing_metrics: list[str] = Field(default_factory=list)

    decision: PromotionDecision

    # --- renderings -------------------------------------------------------

    @staticmethod
    def _fmt(value: float | None, digits: int = 4) -> str:
        return "n/a" if value is None else f"{value:+.{digits}f}"

    @staticmethod
    def _fmt_mean(value: float | None, digits: int = 4) -> str:
        return "n/a" if value is None else f"{value:.{digits}f}"

    def to_markdown(self) -> str:
        """Human-readable rendering of the same numbers pinned in YAML."""
        primary = self.primary
        lines = [
            f"# Three-Seed Confirmation Report — {self.confirmation_id}",
            "",
            f"- Pilot winner: `{self.pilot_winner_id}`",
            f"- Matched seeds: {', '.join(str(s) for s in self.seeds)}",
            f"- Primary metric: `{self.primary_metric}` "
            f"(target delta >= {self.target_delta:+g})",
            "",
            "| | baseline | candidate | paired delta |",
            "|---|---:|---:|---:|",
            "| mean | "
            f"{self._fmt_mean(primary.baseline_mean)} | "
            f"{self._fmt_mean(primary.candidate_mean)} | "
            f"{self._fmt(primary.mean_delta)} |",
            "| 95% CI | | | "
            f"[{self._fmt(primary.ci95_low)}, {self._fmt(primary.ci95_high)}] |",
            "",
        ]

        # Latency gets its own explicit line (§7 requires it in the report).
        latency = self.secondary.get("latency_ms")
        if latency is None or not latency.recorded:
            lines.append("Latency: not recorded")
        else:
            lines.append(
                "Latency: "
                f"{self._fmt_mean(latency.baseline_mean, 3)} ms -> "
                f"{self._fmt_mean(latency.candidate_mean, 3)} ms "
                f"(delta {self._fmt(latency.mean_delta, 3)} ms, "
                f"CI [{self._fmt(latency.ci95_low, 3)}, "
                f"{self._fmt(latency.ci95_high, 3)}])"
            )
        lines.append("")

        recorded_secondary = [
            row
            for name, row in sorted(self.secondary.items())
            if name != "latency_ms" and row.recorded
        ]
        if recorded_secondary:
            lines += [
                "## Secondary metrics",
                "",
                "| metric | baseline | candidate | delta | 95% CI |",
                "|---|---:|---:|---:|---:|",
            ]
            for row in recorded_secondary:
                lines.append(
                    f"| {row.metric} | {self._fmt_mean(row.baseline_mean)} | "
                    f"{self._fmt_mean(row.candidate_mean)} | "
                    f"{self._fmt(row.mean_delta)} | "
                    f"[{self._fmt(row.ci95_low)}, {self._fmt(row.ci95_high)}] |"
                )
            lines.append("")
        if self.missing_metrics:
            lines.append(
                "Missing on some runs: " + ", ".join(self.missing_metrics)
            )
            lines.append("")

        decision = self.decision
        lines += [f"## Decision: {decision.verdict}", ""]
        for reason in decision.reasons:
            lines.append(f"- {reason}")
        if decision.hard_constraint_violations:
            lines.append("")
            lines.append("Hard-constraint violations:")
            for violation in decision.hard_constraint_violations:
                lines.append(f"- {violation}")
        if decision.acknowledged_benefits:
            lines.append("")
            lines.append("Acknowledged benefits:")
            for benefit in decision.acknowledged_benefits:
                lines.append(f"- {benefit}")
        lines.append("")
        return "\n".join(lines)


def _row_from_stats(stats: object, metric: str, *, recorded: bool) -> MetricReportRow:
    """Build a report row from a ``MetricStatistics``-shaped object."""
    row = MetricReportRow(metric=metric, recorded=recorded)
    if not recorded or stats is None:
        return row
    row.baseline_mean = getattr(stats, "baseline_mean", None)
    row.candidate_mean = getattr(stats, "candidate_mean", None)
    row.mean_delta = getattr(stats, "mean_delta", None)
    row.ci95_low = getattr(stats, "ci95_low", None)
    row.ci95_high = getattr(stats, "ci95_high", None)
    return row


def build_confirmation_report(
    state: SeedConfirmationState,
    decision: PromotionDecision,
) -> ConfirmationReport:
    """Assemble the report from the completed matrix and its decision."""
    stats = decision.statistics
    if stats is None:
        raise ValueError(
            "the promotion decision must carry its statistics to build a report"
        )

    secondary: dict[str, MetricReportRow] = {}
    for name, metric_stats in stats.secondary.items():
        secondary[name] = _row_from_stats(metric_stats, name, recorded=True)
    for name in stats.missing_metrics:
        secondary[name] = MetricReportRow(metric=name, recorded=False)

    return ConfirmationReport(
        confirmation_id=state.confirmation_id,
        pilot_winner_id=state.pilot_winner_id,
        seeds=list(state.baseline_seeds),
        primary_metric=decision.primary_metric,
        target_delta=decision.target_delta,
        primary=_row_from_stats(stats.primary, stats.primary.metric, recorded=True),
        secondary=secondary,
        missing_metrics=list(stats.missing_metrics),
        decision=decision,
    )


def write_confirmation_report(
    report: ConfirmationReport,
    yaml_path: Path | str,
    markdown_path: Path | str,
) -> None:
    """Write both renderings (§7: confirmation_report.yaml + Markdown)."""
    report.to_yaml(yaml_path)
    Path(markdown_path).write_text(report.to_markdown(), encoding="utf-8")


__all__ = [
    "ConfirmationReport",
    "MetricReportRow",
    "REPORT_SCHEMA_VERSION",
    "build_confirmation_report",
    "write_confirmation_report",
]
