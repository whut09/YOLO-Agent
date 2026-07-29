"""Non-executable requests emitted when paper adapters are missing."""

from __future__ import annotations

from yolo_agent.agents.paper_recipe_materialization.schemas import (
    PaperRecipeImplementationRequest,
)
from yolo_agent.recipes.paper_priors import RecipePrior
from yolo_agent.recipes.recipe_materializer import RecipeMaterialization


def implementation_request_from_materialization(
    prior: RecipePrior,
    materialization: RecipeMaterialization,
) -> PaperRecipeImplementationRequest:
    action = materialization.implementation_action
    return PaperRecipeImplementationRequest(
        prior_id=prior.prior_id,
        component_ids=(action.component_ids if action is not None else prior.component_ids),
        required_adapters=(action.required_adapters if action is not None else prior.required_adapter),
        reason=(
            action.reason
            if action is not None
            else "A runtime-integrated adapter is required before paper recipe materialization."
        ),
    )


def runtime_implementation_request(
    prior: RecipePrior,
    reasons: list[str],
) -> PaperRecipeImplementationRequest:
    return PaperRecipeImplementationRequest(
        prior_id=prior.prior_id,
        component_ids=list(prior.component_ids),
        required_adapters=list(prior.required_adapter),
        reason=(
            "Runtime adapter lookup, dry-run, smoke, or payload verification failed: "
            + "; ".join(reasons)
        ),
    )


__all__ = [
    "implementation_request_from_materialization",
    "runtime_implementation_request",
]
