"""Delta-driven next-round portfolio (Prompt-18J §3).

``DetectionErrorDelta`` → remaining/regressed error facts →
``EvidenceLinkedHypothesis`` → action families → eligible ``ActionSpec``
candidates from the catalog → a ranked candidate portfolio for the next
round.  The pipeline consumes the *movement surface*, not the headline
mAP: hypotheses are re-derived from the candidate's post-round profile
(what errors remain) and re-weighted by which surfaces actually regressed.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from yolo_agent.agents.action_space import ActionCatalog
from yolo_agent.core.detection_error_delta import DetectionErrorDelta
from yolo_agent.core.detection_error_profile import DetectionErrorProfile
from yolo_agent.core.error_round_decision import RoundDecision
from yolo_agent.core.error_root_cause import (
    EvidenceLinkedHypothesis,
    derive_root_cause_hypotheses,
)

NEXT_ROUND_SCHEMA_VERSION = "next_round_portfolio.v1"


class PortfolioEntry(BaseModel):
    """One eligible next-round candidate with its selection rationale."""

    model_config = ConfigDict(extra="forbid")

    action_id: str
    family: str
    hypothesis_id: str
    rationale: str
    rank: int


class NextRoundPortfolio(BaseModel):
    """The ranked candidate portfolio for the next optimization round."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = NEXT_ROUND_SCHEMA_VERSION
    run_id: str
    round_basis: str
    hypotheses: list[EvidenceLinkedHypothesis] = Field(default_factory=list)
    entries: list[PortfolioEntry] = Field(default_factory=list)
    families_without_specs: list[str] = Field(default_factory=list)

    @property
    def eligible_action_ids(self) -> list[str]:
        return [item.action_id for item in self.entries]


def build_next_round_portfolio(
    profile: DetectionErrorProfile,
    delta: DetectionErrorDelta | None,
    decision: RoundDecision,
    catalog: ActionCatalog,
) -> NextRoundPortfolio:
    """Rank the next round's candidates from the measured delta surface.

    Hypotheses come from the candidate's post-round profile (the errors that
    *remain*), and entries whose family addresses a regressed surface are
    ranked first — a round that gained AP_small but regressed background FP
    prioritises hard-negative/threshold/postprocess work over piling on more
    P2-style capacity.
    """
    if decision.decision not in {"refine", "promote"}:
        raise ValueError(
            "a portfolio can only follow a budget-consuming decision "
            f"(refine/promote); got '{decision.decision}'"
        )

    hypotheses = derive_root_cause_hypotheses(profile, delta)
    regressed_sections = {
        name.split(".", 1)[0] for name in delta.regressions()
    } if delta is not None else set()
    # Pattern-level tags so the catalog lookup can find the specs each
    # hypothesis is eligible for (the linked hypothesis carries evidence,
    # not tags; the pattern determines the problem surface).
    pattern_tags = _PATTERN_TAGS

    entries: list[PortfolioEntry] = []
    seen_actions: set[str] = set()
    families_without_specs: list[str] = []
    for hypothesis in hypotheses:
        tags = pattern_tags.get(hypothesis.pattern, [])
        for family in hypothesis.candidate_action_families:
            specs = catalog.by_problem_tags(list(tags), families=[family])
            if not specs:
                if family not in families_without_specs:
                    families_without_specs.append(family)
                continue
            addresses_regression = _family_addresses(family, regressed_sections)
            for spec in specs:
                if spec.action_id in seen_actions:
                    continue
                seen_actions.add(spec.action_id)
                entries.append(
                    PortfolioEntry(
                        action_id=spec.action_id,
                        family=spec.family,
                        hypothesis_id=hypothesis.hypothesis_id,
                        rationale=(
                            "addresses regressed surface: "
                            + ", ".join(sorted(regressed_sections))
                            if addresses_regression
                            else "addresses remaining error structure"
                        ),
                        # Regression-addressing entries rank first; the rest
                        # keep catalog order for determinism.
                        rank=0 if addresses_regression else 1,
                    )
                )

    entries.sort(key=lambda item: (item.rank, item.action_id))
    for position, entry in enumerate(entries):
        entry.rank = position

    return NextRoundPortfolio(
        run_id=profile.run_id,
        round_basis=(
            f"delta:{delta.candidate_profile_id}" if delta is not None
            else f"profile:{profile.profile_id}"
        ),
        hypotheses=hypotheses,
        entries=entries,
        families_without_specs=families_without_specs,
    )


_SECTION_FAMILY_HINTS: dict[str, set[str]] = {
    # Regressed surface -> families that plausibly address it.
    "false_positive": {
        "sampling", "augmentation", "classification_loss", "threshold",
        "postprocess", "calibration",
    },
    "false_negative": {
        "sampling", "input_resolution", "head", "assignment", "auxiliary_loss",
        "annotation", "inference",
    },
    "localization": {"bbox_loss", "assignment", "input_resolution", "annotation"},
    "classification": {"classification_loss", "augmentation", "sampling"},
    "confidence": {"calibration", "threshold", "postprocess"},
    "scale": {"input_resolution", "head", "neck", "feature_fusion", "sampling"},
    "global": set(),
    "per_class": {"classification_loss", "sampling", "augmentation"},
    "resources": {"model_scale", "inference", "postprocess"},
}


# Root-cause pattern -> the problem tags its specs match in the catalog.
_PATTERN_TAGS: dict[str, tuple[str, ...]] = {
    "small_object_feature_loss": ("small_object_fn",),
    "background_false_positive": ("background_fp",),
    "localization_error": ("localization_error",),
    "class_confusion": ("classification_confusion",),
    "confidence_miscalibration": ("low_confidence", "overconfidence"),
}


def _family_addresses(family: str, regressed_sections: set[str]) -> bool:
    for section in regressed_sections:
        if family in _SECTION_FAMILY_HINTS.get(section, set()):
            return True
    return False


__all__ = [
    "NextRoundPortfolio",
    "PortfolioEntry",
    "build_next_round_portfolio",
]
