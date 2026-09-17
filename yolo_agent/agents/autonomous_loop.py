"""Bounded autonomous optimization loop state machine.

The Prompt-15 contract: Task Goal → Baseline Evidence → Error Facts →
Hypotheses → Eligible Families → Action Retrieval → Compatibility/Maturity
Filter → Candidate Portfolio → Matched Comparison → ASHA Budget → Candidate
Evidence → **full Error Delta** → Promotion/Reject/Refine/Rollback → Next
Round → Stopping.  Every round transition consumes the complete
:class:`ErrorDeltaProfile` and the deterministic :func:`decide_next_round`
verdict — the loop is delta-driven by construction, and stopping is honest:
``stop_exhausted`` only fires when no evidence-supported action remains.
Orchestration only.  Execution is delegated to an injected runner; tests use
a deterministic fake and no training ever runs here.
"""

from __future__ import annotations

from typing import Any, Protocol

from pydantic import BaseModel, Field

from yolo_agent.agents.action_space import ActionCatalog, build_diagnosis_action_chain
from yolo_agent.agents.action_space_schemas import ActionSpec
from yolo_agent.agents.bounded_hpo import HpoScope, hpo_scope_for_action
from yolo_agent.agents.error_delta_profile import ErrorDeltaProfile
from yolo_agent.agents.loop_decision_engine import (
    DecisionEngineConfig,
    LoopDecisionType,
    NextRoundDecision,
    decide_next_round,
)
from yolo_agent.core.error_facts import ErrorFact
from yolo_agent.core.task_spec import TaskSpec


class LoopOutcomeRecord(BaseModel):
    """One executed candidate and the profile it produced."""

    round_index: int
    action_id: str
    action_family: str
    scope: HpoScope
    candidate_id: str
    profile: ErrorDeltaProfile


class RoundRecord(BaseModel):
    """Complete audit of one loop round."""

    round_index: int
    decision: LoopDecisionType
    reasons: list[str] = Field(default_factory=list)
    pareto_weighted_improvement: float = 0.0
    pareto_axes: dict[str, float] = Field(default_factory=dict)
    problem_tags: list[str] = Field(default_factory=list)
    eligible_action_ids: list[str] = Field(default_factory=list)
    blocked_action_ids: list[str] = Field(default_factory=list)
    blocked_reasons: list[str] = Field(default_factory=list)
    executed_action_id: str | None = None
    stop_reason: str | None = None


class AutonomousLoopReport(BaseModel):
    """Replayable transcript of a bounded loop run."""

    rounds: list[RoundRecord] = Field(default_factory=list)
    outcomes: list[LoopOutcomeRecord] = Field(default_factory=list)
    stopped: bool = False
    final_decision: LoopDecisionType | None = None
    stop_reason: str | None = None

    def round(self, index: int) -> RoundRecord:
        for record in self.rounds:
            if record.round_index == index:
                return record
        raise KeyError(f"round {index} not recorded")


class FakeExperimentRunner(Protocol):
    """Runner protocol: deterministic fakes satisfy this; no training here."""

    def run_candidate(
        self,
        action: ActionSpec,
        round_index: int,
        overrides: dict[str, Any],
    ) -> ErrorDeltaProfile: ...


class CompatibilityMaturityFilter(BaseModel):
    """Deterministic eligibility gate over retrieved actions."""

    implementation_ready_ids: list[str] = Field(default_factory=list)
    synthetic_smoke_only_ids: list[str] = Field(default_factory=list)

    def evaluate(self, actions: list[ActionSpec]) -> tuple[list[ActionSpec], list[tuple[str, str]]]:
        """Split actions into (eligible, [(action_id, blocked_reason), ...])."""

        eligible: list[ActionSpec] = []
        blocked: list[tuple[str, str]] = []
        for action in actions:
            if action.action_id not in self.implementation_ready_ids:
                blocked.append(
                    (action.action_id, "not_implementation_ready")
                )
                continue
            if action.action_id in self.synthetic_smoke_only_ids:
                blocked.append((action.action_id, "synthetic_smoke_only_not_runtime_verified"))
                continue
            eligible.append(action)
        return eligible, blocked


class BoundedAutonomousLoop:
    """Drive diagnosis → action portfolio → matched comparison → decision."""

    def __init__(
        self,
        *,
        catalog: ActionCatalog,
        runner: FakeExperimentRunner,
        config: DecisionEngineConfig | None = None,
    ) -> None:
        self.catalog = catalog
        self.runner = runner
        self.config = config or DecisionEngineConfig()

    def run(
        self,
        *,
        task_spec: TaskSpec,
        initial_facts: list[ErrorFact],
        max_rounds: int,
        filter_: CompatibilityMaturityFilter,
        goal_met_round: int | None = None,
        budget_remaining_rounds: set[int] | None = None,
        extra_tags_by_round: dict[int, list[str]] | None = None,
        forced_tags_by_round: dict[int, list[str]] | None = None,
        profile_by_round_action: dict[int, dict[str, ErrorDeltaProfile]] | None = None,
    ) -> AutonomousLoopReport:
        """Run the bounded loop to a stopping decision.

        ``profile_by_round_action`` lets deterministic tests supply exactly the
        delta each executed action produces per round; the loop itself never
        fabricates profiles.
        """

        report = AutonomousLoopReport()
        facts = list(initial_facts)
        tried_action_ids: set[str] = set()
        banned: set[str] = set()
        budget = budget_remaining_rounds
        extra_tags = extra_tags_by_round or {}
        forced_tags = forced_tags_by_round or {}
        supplied = profile_by_round_action or {}

        for round_index in range(1, max_rounds + 1):
            tags = sorted(set(extra_tags.get(round_index, [])) | set(forced_tags.get(round_index, [])))
            chain = build_diagnosis_action_chain(facts, self.catalog, extra_tags=tags or None)
            retrieved = [
                spec
                for selection in chain.selections
                for spec in selection.action_specs
            ]
            unique: dict[str, ActionSpec] = {}
            for spec in retrieved:
                unique.setdefault(spec.action_id, spec)
            eligible, blocked = filter_.evaluate(list(unique.values()))
            untried = [
                action for action in eligible
                if action.action_id not in tried_action_ids
                and action.action_id not in banned
            ]
            record = RoundRecord(
                round_index=round_index,
                decision="refine",
                problem_tags=list(chain.problem_tags),
                eligible_action_ids=[a.action_id for a in eligible],
                blocked_action_ids=[item[0] for item in blocked],
                blocked_reasons=[item[1] for item in blocked],
            )

            remaining_budget = True if budget is None else round_index in budget
            if not untried or not remaining_budget:
                if not untried:
                    record.decision = "stop_exhausted"
                    record.stop_reason = "no_evidence_supported_action"
                else:
                    record.decision = "stop_budget"
                    record.stop_reason = "budget_exhausted"
                report.rounds.append(record)
                report.stopped = True
                report.final_decision = record.decision  # type: ignore[assignment]
                report.stop_reason = record.stop_reason
                return report

            chosen = _select_action(untried)
            executed_profile: ErrorDeltaProfile | None = None
            if round_index in supplied and chosen.action_id in supplied[round_index]:
                executed_profile = supplied[round_index][chosen.action_id]
            else:
                executed_profile = self.runner.run_candidate(
                    chosen, round_index, dict.fromkeys(chosen.parameters or {})
                )
            tried_action_ids.add(chosen.action_id)
            record.executed_action_id = chosen.action_id
            report.outcomes.append(
                LoopOutcomeRecord(
                    round_index=round_index,
                    action_id=chosen.action_id,
                    action_family=chosen.family,
                    scope=hpo_scope_for_action(chosen),
                    candidate_id=f"{chosen.action_id}@r{round_index}",
                    profile=executed_profile,
                )
            )

            goal = goal_met_round is not None and round_index >= goal_met_round
            decision = decide_next_round(
                executed_profile,
                round_index=round_index,
                supported_action_ids=[a.action_id for a in untried[1:]],
                task_spec=task_spec,
                budget_remaining=remaining_budget,
                goal_met=goal,
                config=self.config,
            )
            record.decision = decision.decision
            record.reasons = list(decision.reasons)
            record.pareto_weighted_improvement = decision.pareto.weighted_improvement
            record.pareto_axes = dict(decision.pareto.per_axis)
            report.rounds.append(record)

            if decision.decision in {"stop_goal_met", "stop_exhausted", "stop_budget"}:
                report.stopped = True
                report.final_decision = decision.decision
                report.stop_reason = decision.stop_reason
                return report
            if decision.decision == "rollback":
                # The candidate is banned (constraint violation), but the
                # action itself stays untried so a *different* candidate for
                # the same problem can still be selected next round.
                banned.add(chosen.action_id)
            if decision.decision == "reject":
                tried_action_ids.add(chosen.action_id)
                continue
            # Delta-driven next round: the next diagnosis expands the problem
            # tags this round's *full profile* still shows — a promoted or
            # refined action's residual regressions (e.g. latency after an
            # AP_small win) steer round N+1, not a single mAP scalar.
            next_tags = _residual_problem_tags(executed_profile, decision)
            if next_tags:
                extra_tags.setdefault(round_index + 1, [])
                extra_tags[round_index + 1] = sorted(
                    set(extra_tags[round_index + 1]) | next_tags
                )
            if decision.decision == "request_annotation":
                annotation_tags = ["missing_labels"]
                extra_tags.setdefault(round_index + 1, [])
                extra_tags[round_index + 1] = sorted(
                    set(extra_tags[round_index + 1]) | annotation_tags
                )
        report.stopped = True
        report.final_decision = "refine"
        report.stop_reason = "max_rounds_reached"
        return report

__all__ = [
    "AutonomousLoopReport",
    "BoundedAutonomousLoop",
    "CompatibilityMaturityFilter",
    "FakeExperimentRunner",
    "LoopOutcomeRecord",
    "RoundRecord",
]


def _select_action(untried: list[ActionSpec]) -> ActionSpec:
    """Deterministic selection: paper lineage first, then declaration order."""

    paper = [action for action in untried if action.is_paper_action]
    return paper[0] if paper else untried[0]


def _residual_problem_tags(profile: ErrorDeltaProfile, decision: NextRoundDecision) -> set[str]:
    """Problem tags implied by the executed round's residual delta profile."""

    tags: set[str] = set()
    if decision.primary_problem_tag:
        tags.add(decision.primary_problem_tag)
    for delta in profile.regressions():
        if delta.metric_name == "precision":
            tags.add("background_fp")
        if delta.metric_name == "recall":
            tags.add("small_object_fn")
        if delta.scope == "scale" and "small" in (delta.subject or delta.metric_name):
            tags.add("small_object_fn")
    return tags
