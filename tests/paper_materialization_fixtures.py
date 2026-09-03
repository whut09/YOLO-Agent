from __future__ import annotations

import hashlib
from pathlib import Path

from yolo_agent.agents.candidate_generator import CandidateConfig
from yolo_agent.agents.decision_bundle import DecisionContext
from yolo_agent.agents.paper_component_gate import PaperEligibilityBudget
from yolo_agent.agents.paper_recipe_materialization.schemas import (
    PaperRecipeCandidateInput,
)
from yolo_agent.components.adapters import ComponentAdapterRegistry, DummyAdapter as BaseDummyAdapter
from yolo_agent.components.compatibility import CompatibilityResult
from yolo_agent.components.contracts import ComponentContract
from yolo_agent.components.maturity import ComponentMaturityArtifact
from yolo_agent.core.command_spec import CommandSpec
from yolo_agent.core.error_facts import ErrorFact
from yolo_agent.core.experiment_graph import ExperimentNode
from yolo_agent.core.optimization_objective import OptimizationObjective
from yolo_agent.recipes.paper_priors import RecipePrior, RecipePriorEvidence
from yolo_agent.research.snapshot import ResearchSnapshot, research_snapshot_hash
from yolo_agent.research.method_profiles import (
    PaperImplementationDecision,
    PaperMethodProfile,
)


PROTOCOL_HASH = "paper-protocol-640"
SNAPSHOT_PAYLOAD = {
    "schema_version": "research_snapshot.v1",
    "papers_version": "papers-v1",
    "component_registry_version": "components-v1",
    "recipe_registry_version": "recipes-v1",
    "classifications_version": "classifications-v1",
    "extractions_version": "extractions-v1",
    "compatibility_version": "compatibility-v1",
    "reproduction_queue_version": "reproduction-v1",
    "paper_count": 1,
    "component_count": 1,
    "recipe_count": 1,
}
SNAPSHOT_HASH = research_snapshot_hash(SNAPSHOT_PAYLOAD)


class DummyAdapter(BaseDummyAdapter):
    """Dummy fixture whose CPU smoke is explicit local readiness evidence."""

    def smoke_test(self, context):  # type: ignore[no-untyped-def]
        return super().smoke_test(context).model_copy(update={"evidence_kind": "local"})


def snapshot() -> ResearchSnapshot:
    return ResearchSnapshot(
        **SNAPSHOT_PAYLOAD,
        snapshot_hash=SNAPSHOT_HASH,
        paper_intelligence="available",
        frozen=True,
    )


def context(run_id: str = "paper-run") -> DecisionContext:
    return DecisionContext(
        run_id=run_id,
        research_snapshot_hash=SNAPSHOT_HASH,
        research_snapshot_verified=True,
        paper_intelligence="available",
    )


def contract(
    component_id: str = "dummy.component",
    *,
    maturity: str = "smoke_passed",
    implementation_path: str | None = "yolo_agent.components.adapters.dummy",
    adapter_class: str | None = "DummyAdapter",
) -> ComponentContract:
    artifact_path = Path(__file__)
    maturity_artifacts = (
        [
            ComponentMaturityArtifact(
                component_id=component_id,
                target_maturity="smoke_passed",
                artifact_type="smoke_report",
                artifact_path=artifact_path,
                artifact_sha256=hashlib.sha256(artifact_path.read_bytes()).hexdigest(),
                status="passed",
                producer="pytest_fixture",
            )
        ]
        if maturity == "smoke_passed"
        else []
    )
    return ComponentContract(
        component_id=component_id,
        display_name="Dummy paper component",
        category="augmentation",
        implementation_path=implementation_path,
        adapter_class=adapter_class,
        maturity=maturity,
        maturity_artifacts=maturity_artifacts,
        fixed_imgsz_compatible=True,
        checkpoint_compatibility="unchanged_graph",
        supports_amp=True,
    )


def prior(
    prior_id: str = "paper-prior-dummy",
    component_id: str = "dummy.component",
) -> RecipePrior:
    return RecipePrior(
        prior_id=prior_id,
        research_snapshot_hash=SNAPSHOT_HASH,
        paper_ids=["paper-dummy"],
        component_ids=[component_id],
        target_error_facts=[{"fact_type": "area_metric", "subject": "small"}],
        target_metrics=["ap_small"],
        suggested_changed_variables=["data.sampling_policy"],
        baseline_protocol={"imgsz": 640, "protocol_hashes": [PROTOCOL_HASH]},
        evidence_prior=[RecipePriorEvidence(
            paper_id="paper-dummy",
            claim="Sampling may improve small-object recall.",
            source_location="paper:summary",
            evidence_level="paper_claim",
        )],
        expected_paper_effect={"ap_small": "unknown"},
        implementation_status="smoke_passed",
        yolo26_compatibility="compatible",
        required_adapter=["DummyAdapter"],
        confidence=0.8,
        source_locations=["paper:summary"],
    )


def node(
    candidate_id: str = "paper-candidate",
    *,
    control: bool = False,
    imgsz: int = 640,
) -> ExperimentNode:
    command = CommandSpec.ultralytics_train(
        model="yolo26n.pt",
        data="coco.yaml",
        project="runs/ultralytics",
        name=candidate_id,
        epochs=3,
        imgsz=imgsz,
        batch=16,
        seed=1,
        metadata={
            "objective_hash": "objective-placeholder",
            "protocol_hash": PROTOCOL_HASH,
            "run_protocol_hash": PROTOCOL_HASH,
            "matched_baseline_control": control,
            "dataset_manifest_sha256": "dataset-1",
            "split": "val2017",
            "fidelity": "pilot_3",
            "subset_manifest_sha256": "subset-1",
            "batch_policy_hash": "batch-16",
            "eval_protocol_hash": "coco-eval-v1",
            "ultralytics_version": "test",
            "epochs": 3,
            "seed": 1,
        },
    )
    return ExperimentNode(
        node_id=f"node-{candidate_id}",
        candidate_config=CandidateConfig(
            candidate_id=candidate_id,
            base_model="yolo26n.pt",
            scale="n",
            framework="ultralytics",
            components=[] if control else ["dummy.component"],
        ),
        data_version="coco2017",
        seed=1,
        command=command.display(),
        command_spec=command,
        changed_variables={} if control else {"data.sampling_policy": "paper"},
    )


def error_fact(
    *,
    run_id: str = "paper-run",
    evidence_role: str = "current_observation",
    protocol_hash: str = PROTOCOL_HASH,
) -> ErrorFact:
    return ErrorFact(
        run_id=run_id,
        candidate_id="baseline",
        node_id="node-baseline",
        protocol_hash=protocol_hash,
        evidence_role=evidence_role,
        fact_type="area_metric",
        subject="small",
        area="small",
        metric_name="ap_small",
        value=0.2,
        source="coco_post_eval",
    )


def objective() -> OptimizationObjective:
    return OptimizationObjective(
        baseline_run_id="paper-run",
        baseline_candidate_id="baseline",
        baseline_protocol_hash=PROTOCOL_HASH,
        max_gpu_hours=4.0,
    )


def budget() -> PaperEligibilityBudget:
    return PaperEligibilityBudget(
        max_gpu_hours=4.0,
        estimated_candidate_gpu_hours=0.25,
    )


def candidate_input(
    prior_id: str = "paper-prior-dummy",
    candidate_id: str = "paper-candidate",
) -> PaperRecipeCandidateInput:
    profile = PaperMethodProfile(
        profile_id="profile-paper-dummy",
        paper_id="paper-dummy",
        method_name="dummy sampling",
        paper_component_ids=[prior(prior_id).component_ids[0]],
        canonical_component_ids=["dummy.component"],
        source_locations=["paper:summary"],
    )
    decision = PaperImplementationDecision(
        paper_id="paper-dummy",
        profile_id=profile.profile_id,
        decision="reuse_existing_adapter",
        canonical_component_ids=["dummy.component"],
        reusable_adapter_ids=["dummy.component"],
        reasons=["test fixture reuses the dummy adapter"],
        source_locations=["paper:summary"],
    ).with_hash()
    return PaperRecipeCandidateInput(
        prior=prior(prior_id),
        method_profile=profile,
        implementation_decision=decision,
        compatibility=CompatibilityResult(ok=True),
        source_node=node(candidate_id),
        matched_control_node=node(f"baseline-{candidate_id}", control=True),
        component_family="sampling",
        bucket="exploitation",
    )


def adapter_registry(component_id: str = "dummy.component") -> ComponentAdapterRegistry:
    registry = ComponentAdapterRegistry()
    registry.register(component_id, DummyAdapter)
    return registry


def gate_kwargs(tmp_path: Path, *, candidates: list[PaperRecipeCandidateInput]) -> dict:
    return {
        "run_id": "paper-run",
        "decision_context": context(),
        "research_snapshot": snapshot(),
        "candidates": candidates,
        "current_error_facts": [error_fact()],
        "component_contracts": {"dummy.component": contract()},
        "objective": objective(),
        "budget": budget(),
        "round_index": 1,
    }


__all__ = [
    "PROTOCOL_HASH",
    "adapter_registry",
    "budget",
    "candidate_input",
    "context",
    "contract",
    "error_fact",
    "gate_kwargs",
    "node",
    "objective",
    "prior",
    "snapshot",
]
