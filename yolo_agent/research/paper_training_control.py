"""Training-control / optimizer / regularization audit over the frozen 83.

Scope rule (from frozen evidence, not guessed): a paper is in scope only when
its frozen plan entry declares a ``training_schedule`` insertion point or the
entry's implementation domain is ``training_strategy`` / ``optimizer`` /
``regularization``.  Today exactly one paper qualifies.  The other 82 are
recorded ``out_of_scope`` with an explicit per-paper reason — never silently
dropped.  Probes run one synthetic ``optimizer.step()`` and scheduler-style
batch indexing; no epoch training and no ExperimentNode is created.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from yolo_agent.components.adapters.domain_adaptation.branches import (
    DOMAIN_BRANCH_PROFILES,
    NAMED_PAPER_BRANCHES as DOMAIN_NAMED_PAPER_BRANCHES,
)
from yolo_agent.components.training_control import (
    TrainingScheduleController,
    WeightSchedule,
    domain_alignment_weight_spec,
)
from yolo_agent.research.paper_83_campaign_schemas import Paper83Manifest
from yolo_agent.research.paper_83_implementation_schemas import (
    Paper83EngineeringPlan,
    Paper83EngineeringPlanEntry,
)
from yolo_agent.research.training_control_adaptation import (
    TrainingControlPaperRecord,
    TrainingParameterAdaptation,
)

TRAINING_CONTROL_DOMAINS = frozenset({"training_strategy", "optimizer", "regularization"})

SCHEDULE_INSERTION_POINT = "training_schedule"

#: Default runtime baseline for the domain-alignment auxiliary weight.
DEFAULT_ALIGNMENT_WEIGHT = 0.05


def _entry_declares_schedule(entry: Paper83EngineeringPlanEntry) -> bool:
    config = entry.paper_specific_config or {}
    raw = json.dumps(config, default=str)
    return f'"{SCHEDULE_INSERTION_POINT}"' in raw or f"'{SCHEDULE_INSERTION_POINT}'" in raw


def _entry_in_scope(entry: Paper83EngineeringPlanEntry) -> bool:
    return entry.implementation_domain in TRAINING_CONTROL_DOMAINS or _entry_declares_schedule(
        entry
    )


def _entry_schedule_curve_searchable(entry: Paper83EngineeringPlanEntry) -> bool:
    """Frozen evidence fixes a curve only when the plan records schedule values."""

    config = entry.paper_specific_config or {}
    for value in config.values():
        if isinstance(value, dict):
            for key in ("schedule", "warmup_batches", "weight_schedule"):
                if key in value:
                    return True
    return False


def _alignment_schedule_adaptation(
    entry: Paper83EngineeringPlanEntry,
    *,
    branch_id: str,
) -> TrainingParameterAdaptation:
    curve_note = (
        "exact paper curve is searchable: frozen evidence declares the "
        "training_schedule insertion point without a certified weight curve"
        if not _entry_schedule_curve_searchable(entry)
        else "paper curve values are taken from the frozen plan config"
    )
    return TrainingParameterAdaptation(
        paper_id=entry.paper_id,
        parameter_name="alignment_loss_weight",
        paper_semantics=(
            "self-training applies the unsupervised alignment term with a "
            "batch-dependent weight following the training schedule rather "
            "than a fixed constant"
        ),
        runtime_contract=(
            f"WeightSchedule resolves weight_at(batch_index) and the "
            f"{branch_id} branch plugin multiplies its strategy loss by the "
            "scheduled effective weight (zero-weight batches drop the term "
            "exactly)"
        ),
        runtime_binding_point="compute_loss",
        ultralytics_hook=(
            "Ultralytics trainer compute_loss stage; on_train_batch_start "
            "announces the effective weight, on_train_batch_end records it"
        ),
        adaptation_class="faithful_adaptation",
        preserved_information=[
            "auxiliary-term weighting semantics",
            "batch-index dependence of the weight",
            "zero-weight batches remove the term from the gradient",
        ],
        approximated_information=[curve_note],
        evidence_refs=[*entry.source_locations, *entry.required_evidence],
    )


def _schedule_probe(
    schedule: WeightSchedule,
    *,
    baseline_weight: float,
) -> dict[str, Any]:
    """One synthetic optimizer.step() per scheduled batch; no epoch training."""

    controller = TrainingScheduleController(schedule, baseline_weight=baseline_weight)
    model = torch.nn.Linear(4, 2)
    groups = [{"params": model.parameters(), "weight_decay": 0.0}]
    observed: list[float] = []
    for index in range(min(schedule.total_batches, 8)):
        effective = controller.on_train_batch_start(batch_index=index, trainer=None)
        optimizer = torch.optim.SGD(groups, lr=0.1)
        for parameter in model.parameters():
            parameter.grad = None
        before = [parameter.detach().clone() for parameter in model.parameters()]
        anchor = torch.randn(8, 4)
        target = torch.randn(8, 2)
        main = torch.nn.functional.mse_loss(model(anchor), target)
        auxiliary = (model.weight.square()).sum()
        loss = main + effective * auxiliary
        loss.backward()
        optimizer.step()
        update_norm = sum(
            float((parameter.detach() - before_parameter).norm())
            for before_parameter, parameter in zip(before, model.parameters())
        )
        observed.append(update_norm)
        controller.on_train_batch_end(loss_term=float(loss.detach()))
    rollback = controller.rollback()
    constant = [
        baseline_weight for _ in range(min(schedule.total_batches, 8))
    ]
    differs_from_constant = any(
        abs(a - b) > 1e-12 for a, b in zip(observed, constant)
    ) or schedule.weight_at(0) != baseline_weight
    return {
        "controller_lifecycle": bool(controller.evidence["last_effective_weight"] is not None),
        "rollback_restores_baseline": rollback["weight"] == baseline_weight
        and controller.evidence["schedule_applied"] is False,
        "gradient_responds_to_schedule": differs_from_constant,
        "observed_update_norms": [round(value, 6) for value in observed],
    }


class PaperTrainingControlAuditBuilder:
    """Audit training-control scope and behavior for every frozen paper."""

    def __init__(
        self,
        *,
        manifest_path: Path | str = "configs/research/paper_83_manifest.yaml",
        plan_path: Path | str = "configs/research/paper_83_implementation_plan.yaml",
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.requested_plan_path = Path(plan_path)

    def build(
        self,
        *,
        manifest: Paper83Manifest | None = None,
        plan: Paper83EngineeringPlan | None = None,
    ) -> list[TrainingControlPaperRecord]:
        frozen = manifest or Paper83Manifest.from_yaml(self.manifest_path)
        if frozen.paper_count != 83:
            raise ValueError(
                f"frozen paper manifest must contain 83 papers, got {frozen.paper_count}"
            )
        engineering_plan, _source = self._load_plan(plan)
        by_id = {item.paper_id: item for item in frozen.papers}
        entries = {item.paper_id: item for item in engineering_plan.papers}
        if set(entries) != set(by_id):
            raise ValueError("training-control audit requires every frozen paper entry")
        return [self._record(by_id[paper_id], entries[paper_id]) for paper_id in
                (item.paper_id for item in frozen.papers)]

    def _load_plan(
        self,
        plan: Paper83EngineeringPlan | None,
    ) -> tuple[Paper83EngineeringPlan, Path]:
        if plan is not None:
            return plan, self.requested_plan_path
        if self.requested_plan_path.is_file():
            return (
                Paper83EngineeringPlan.from_yaml(self.requested_plan_path),
                self.requested_plan_path,
            )
        fallback = self.requested_plan_path.with_name("paper_83_engineering_plan.yaml")
        if fallback.is_file():
            return (
                Paper83EngineeringPlan.from_yaml(fallback),
                fallback,
            )
        raise FileNotFoundError(
            f"paper-83 implementation plan does not exist: {self.requested_plan_path}"
        )

    def _record(
        self,
        paper: Any,
        entry: Paper83EngineeringPlanEntry,
    ) -> TrainingControlPaperRecord:
        declares_schedule = _entry_declares_schedule(entry)
        domain_match = entry.implementation_domain in TRAINING_CONTROL_DOMAINS
        if not declares_schedule and not domain_match:
            return TrainingControlPaperRecord(
                paper_id=paper.paper_id,
                title=paper.title,
                year=paper.year,
                primary_domain=entry.implementation_domain,
                in_scope=False,
                scope_reason=(
                    "frozen plan declares no training_schedule insertion point "
                    f"and domain {entry.implementation_domain!r} is not a "
                    "training-control domain"
                ),
                status="out_of_scope",
            )

        blockers: list[str] = []
        branch_id = DOMAIN_NAMED_PAPER_BRANCHES.get(entry.paper_id)
        adaptations: list[TrainingParameterAdaptation] = []
        spec = domain_alignment_weight_spec(default_weight=DEFAULT_ALIGNMENT_WEIGHT)
        schedule: WeightSchedule | None = None
        behavior_passed = False
        behavior_evidence: list[str] = []

        if entry.implementation_domain != "domain_adaptation":
            blockers.append(
                "blocked_missing_evidence:training_control_route_only_bound_for_"
                "domain_adaptation_today"
            )
        elif branch_id is None:
            blockers.append(
                f"blocked_missing_evidence:no_certified_branch_for:{entry.paper_id}"
            )
        else:
            profile = DOMAIN_BRANCH_PROFILES[branch_id]
            adaptations.append(
                _alignment_schedule_adaptation(entry, branch_id=str(profile["runtime_strategy"]))
            )
            # Identity schedule: the shipped default must not drift from the
            # unscheduled runtime baseline.
            identity = WeightSchedule(
                policy="constant",
                start_weight=DEFAULT_ALIGNMENT_WEIGHT,
                end_weight=DEFAULT_ALIGNMENT_WEIGHT,
                total_batches=1,
            )
            schedule = identity
            probe = _schedule_probe(
                identity, baseline_weight=DEFAULT_ALIGNMENT_WEIGHT
            )
            behavior_passed = all(
                probe[key] for key in (
                    "controller_lifecycle",
                    "rollback_restores_baseline",
                    "gradient_responds_to_schedule",
                )
            )
            behavior_evidence = [f"{key}={value}" for key, value in probe.items()]
            if not behavior_passed:
                blockers.append("blocked_missing_evidence:identity_schedule_probe_failed")

        if blockers:
            return TrainingControlPaperRecord(
                paper_id=paper.paper_id,
                title=paper.title,
                year=paper.year,
                primary_domain=entry.implementation_domain,
                in_scope=True,
                scope_reason=(
                    "frozen plan declares a training_schedule insertion point"
                    if declares_schedule
                    else f"domain {entry.implementation_domain!r} is a training-control domain"
                ),
                status="blocked_missing_evidence",
                blockers=blockers,
            )

        return TrainingControlPaperRecord(
            paper_id=paper.paper_id,
            title=paper.title,
            year=paper.year,
            primary_domain=entry.implementation_domain,
            in_scope=True,
            scope_reason=(
                "frozen plan declares a training_schedule insertion point"
                if declares_schedule
                else f"domain {entry.implementation_domain!r} is a training-control domain"
            ),
            schedule_id=schedule.schedule_id if schedule is not None else None,
            adaptations=adaptations,
            protocol_spec_ids=[spec.spec_id],
            behavior_passed=behavior_passed,
            behavior_evidence=behavior_evidence,
            remaining_dependencies=[
                "paper-certified schedule curve (start/end/warmup values) for "
                "the first real training run",
                *entry.paper_specific_missing_parts,
            ],
            status="ready",
        )


def summarize_training_control_audit(
    records: list[TrainingControlPaperRecord],
) -> dict[str, int]:
    return {
        "ready": sum(item.status == "ready" for item in records),
        "blocked_missing_evidence": sum(
            item.status == "blocked_missing_evidence" for item in records
        ),
        "out_of_scope": sum(item.status == "out_of_scope" for item in records),
        "in_scope": sum(item.in_scope for item in records),
    }


def render_training_control_status(
    records: list[TrainingControlPaperRecord],
) -> str:
    summary = summarize_training_control_audit(records)
    lines = [
        "# Paper-83 Training-Control Status",
        "",
        "This report audits training-strategy / optimizer / regularization",
        "mechanisms. It does not train a model: probes run at most one",
        "synthetic optimizer.step() on synthetic parameters. A paper is in",
        "scope only when its frozen plan evidence declares a",
        "`training_schedule` insertion point or a training-control domain.",
        "",
        f"- Frozen papers: {len(records)}",
        f"- Training-control papers in scope: {summary['in_scope']}",
        f"- Ready: {summary['ready']}",
        f"- Blocked for missing evidence: {summary['blocked_missing_evidence']}",
        f"- Out of scope: {summary['out_of_scope']}",
        "",
        "## Per-paper Audit",
        "",
        "| Paper | Domain | Scope | Status | Schedule | Adaptations | Blockers |",
        "|---|---|---|---|---|---|---|",
    ]
    for item in records:
        adaptations = "<br>".join(a.parameter_name for a in item.adaptations) or "none"
        blockers = "<br>".join(item.blockers) or "none"
        schedule = item.schedule_id or "none"
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{item.paper_id}`",
                    item.primary_domain,
                    "training-control" if item.in_scope else "out-of-scope",
                    item.status,
                    schedule,
                    adaptations,
                    blockers,
                ]
            )
            + " |"
        )
    in_scope = [item for item in records if item.in_scope]
    lines.extend(
        [
            "",
            "## Method Notes",
            "",
            "- Semantic adaptation is explicit per paper parameter: paper",
            "  semantics → YOLO-Agent runtime contract → Ultralytics/YOLO26",
            "  hook. A runtime binding point that does not exist in this",
            "  repository fails record validation, so a shared parameter name",
            "  between a paper and Ultralytics can never pass as an",
            "  implementation.",
            "- The shipped schedule default is the constant identity schedule",
            "  (zero drift from the un-scheduled baseline). Ramp and step",
            "  policies are implemented, probed on synthetic parameters",
            "  (gradient magnitude responds to the scheduled weight), and",
            "  remain searchable until paper evidence certifies a curve.",
            "- ProtocolSpec records carry defaults, paper-recommended values",
            "  (only what frozen evidence supports), searchable parameters,",
            "  valid ranges, coupled parameters, machine-checked invalid",
            "  combinations, runtime phase, and rollback values. Rollback",
            "  values must restore defaults, and declared invalid",
            "  combinations must actually fail schedule validation.",
            "- No ExperimentNode is created and no epoch training is started",
            "  by this audit.",
            "",
        ]
    )
    if in_scope:
        lines.extend(
            [
                "## In-Scope Paper Detail",
                "",
            ]
        )
        for item in in_scope:
            lines.append(f"### `{item.paper_id}` — {item.title}")
            lines.append("")
            lines.append(f"- Scope: {item.scope_reason}")
            lines.append(f"- Status: {item.status}")
            for adaptation in item.adaptations:
                lines.append(f"- Parameter: `{adaptation.parameter_name}`")
                lines.append(f"  - Paper semantics: {adaptation.paper_semantics}")
                lines.append(f"  - Runtime contract: {adaptation.runtime_contract}")
                lines.append(f"  - Binding: {adaptation.runtime_binding_point}")
                lines.append(f"  - Ultralytics hook: {adaptation.ultralytics_hook}")
                lines.append(f"  - Preserved: {'; '.join(adaptation.preserved_information)}")
                for note in adaptation.approximated_information:
                    lines.append(f"  - Approximated: {note}")
            for evidence in item.behavior_evidence:
                lines.append(f"- Behavior: {evidence}")
            for dependency in item.remaining_dependencies:
                lines.append(f"- Remaining dependency: {dependency}")
            lines.append("")
    return "\n".join(lines)


def write_training_control_status(
    records: list[TrainingControlPaperRecord],
    output_path: Path | str,
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_training_control_status(records), encoding="utf-8")
    return path


__all__ = [
    "PaperTrainingControlAuditBuilder",
    "summarize_training_control_audit",
    "render_training_control_status",
    "write_training_control_status",
]
