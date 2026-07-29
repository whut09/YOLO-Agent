"""Evidence recovery boundary before paper recipe materialization."""

from __future__ import annotations

from collections.abc import Iterable

from yolo_agent.agents.paper_recipe_materialization.schemas import (
    PaperRecipeEvidenceRecovery,
)
from yolo_agent.core.error_facts import ErrorFact


def current_materialization_error_facts(
    facts: Iterable[ErrorFact],
) -> list[ErrorFact]:
    """Only current observations can bind a paper recipe."""
    return [item for item in facts if item.evidence_role == "current_observation"]


def evidence_recovery_for_facts(
    facts: Iterable[ErrorFact],
) -> PaperRecipeEvidenceRecovery | None:
    current = current_materialization_error_facts(facts)
    if current:
        return None
    return PaperRecipeEvidenceRecovery(
        required_evidence=[
            "current_node_coco_post_eval",
            "current_node_error_facts",
            "same_protocol_hash",
        ],
        reason=(
            "No current-run, current-node error facts are available; only evidence "
            "recovery may be scheduled before paper recipe training."
        ),
    )


__all__ = ["current_materialization_error_facts", "evidence_recovery_for_facts"]
