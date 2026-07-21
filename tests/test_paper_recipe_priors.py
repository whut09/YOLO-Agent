from __future__ import annotations

import pytest

from yolo_agent.components.compatibility import CompatibilityResult
from yolo_agent.components.contracts import ComponentContract
from yolo_agent.core.error_facts import ErrorFact
from yolo_agent.recipes.paper_priors import PaperRecipePriorBuilder, RecipePriorBuildError
from yolo_agent.research.harness_hint_parser import PaperDiagnosticHint
from yolo_agent.research.note_parser import PaperMethodClaim
from yolo_agent.research.schemas import PaperRecord
from yolo_agent.research.snapshot import ResearchSnapshot


def _paper() -> PaperRecord:
    return PaperRecord(
        paper_id="paper-small",
        title="Small Object Sampling",
        abstract="A paper-only small-object prior.",
        year=2025,
        detector_family="yolo26",
        datasets=["COCO"],
        component_ids=["sampling.small_object", "paper.unselected_component"],
    )


def _claim(
    components: list[str] | None = None,
    variables: list[str] | None = None,
) -> PaperMethodClaim:
    return PaperMethodClaim(
        method_name="Small Object Sampling",
        component_ids=components or ["sampling.small_object"],
        changed_variables=variables or ["data.sampler"],
        reported_delta={"ap_small": "+1.2"},
        dataset="COCO",
        model_family="yolo26",
        source_location="notes/paper-small.md#method",
    )


def _hint() -> PaperDiagnosticHint:
    return PaperDiagnosticHint(
        paper_id="paper-small",
        symptom="AP_small is low",
        likely_cause="small objects are under-sampled",
        evidence_needed=["per_class_ap_ar"],
        candidate_component_ids=["sampling.small_object"],
        target_metrics=["ap_small"],
        target_error_facts=["small_object"],
        confidence=0.8,
        source_location="harness_hints[0]",
    )


def _contract(component_id: str = "sampling.small_object", maturity: str = "metadata_only") -> ComponentContract:
    return ComponentContract(
        component_id=component_id,
        display_name=component_id,
        category="sampling",
        implementation_path="yolo_agent.components.adapters.sampling" if maturity != "metadata_only" else None,
        adapter_class="SmallObjectSamplingAdapter" if maturity != "metadata_only" else None,
        maturity=maturity,
        fixed_imgsz_compatible=True,
        supported_detector_families=["yolo26"],
    )


def _fact(*, inherited: bool = False) -> ErrorFact:
    return ErrorFact(
        run_id="run",
        candidate_id="candidate",
        node_id="node",
        fact_type="area_metric",
        subject="small_object AP is low",
        area="small",
        metric_name="ap_small",
        severity="high",
        protocol_hash="protocol-1",
        imgsz=640,
        evidence_role="inherited_context" if inherited else "current_observation",
    )


def _snapshot(*, available: bool = True, frozen: bool = True) -> ResearchSnapshot:
    return ResearchSnapshot.model_construct(
        snapshot_hash="snapshot-1",
        paper_intelligence="available" if available else "unavailable",
        frozen=frozen,
    )


def _compatibility(ok: bool = True) -> CompatibilityResult:
    return CompatibilityResult(
        ok=ok,
        errors=[] if ok else ["one_to_one_head_uses_nms_recipe"],
        yolo26={
            "compatible": ok,
            "incompatible": not ok,
            "blocked_by": [] if ok else ["one_to_one_head_uses_nms_recipe"],
            "required_adapters": [],
        },
    )


def test_builds_non_executable_prior_bound_to_current_error_fact() -> None:
    prior = PaperRecipePriorBuilder().build(
        paper=_paper(),
        method_claim=_claim(),
        diagnostic_hints=[_hint()],
        component_contracts={"sampling.small_object": _contract()},
        compatibility=_compatibility(),
        research_snapshot=_snapshot(),
        current_error_facts=[_fact()],
    )

    assert prior.executable is False
    assert prior.component_ids == ["sampling.small_object"]
    assert "paper.unselected_component" not in prior.component_ids
    assert prior.target_error_facts[0]["protocol_hash"] == "protocol-1"
    assert prior.baseline_protocol == {
        "imgsz": 640,
        "protocol_hashes": ["protocol-1"],
        "research_snapshot_hash": "snapshot-1",
        "evidence_role": "current_observation",
    }
    assert prior.suggested_changed_variables == ["data.sampler"]
    assert prior.evidence_prior and all(item.evidence_level == "paper_claim" for item in prior.evidence_prior)
    assert prior.implementation_status == "metadata_only"
    assert prior.yolo26_compatibility == "adapter_required"


@pytest.mark.parametrize("facts", [[], [_fact(inherited=True)]])
def test_missing_current_evidence_cannot_build_prior(facts: list[ErrorFact]) -> None:
    with pytest.raises(RecipePriorBuildError, match="current error fact"):
        PaperRecipePriorBuilder().build(
            paper=_paper(),
            method_claim=_claim(),
            diagnostic_hints=[_hint()],
            component_contracts={"sampling.small_object": _contract()},
            compatibility=_compatibility(),
            research_snapshot=_snapshot(),
            current_error_facts=facts,
        )


def test_missing_changed_variable_is_rejected() -> None:
    claim = _claim()
    claim = claim.model_copy(update={"changed_variables": []})
    with pytest.raises(RecipePriorBuildError, match="changed variable"):
        PaperRecipePriorBuilder().build(
            paper=_paper(),
            method_claim=claim,
            diagnostic_hints=[_hint()],
            component_contracts={"sampling.small_object": _contract()},
            compatibility=_compatibility(),
            research_snapshot=_snapshot(),
            current_error_facts=[_fact()],
        )


def test_incompatible_paper_prior_is_explicitly_marked() -> None:
    prior = PaperRecipePriorBuilder().build(
        paper=_paper(),
        method_claim=_claim(),
        diagnostic_hints=[_hint()],
        component_contracts={"sampling.small_object": _contract("sampling.small_object", "smoke_passed")},
        compatibility=_compatibility(ok=False),
        research_snapshot=_snapshot(),
        current_error_facts=[_fact()],
    )
    assert prior.yolo26_compatibility == "incompatible"


def test_multi_component_prior_requires_reason_variables_and_ablation() -> None:
    contracts = {
        "sampling.small_object": _contract("sampling.small_object", "smoke_passed"),
        "head.p2_small_object": _contract("head.p2_small_object", "smoke_passed"),
    }
    claim = _claim(
        ["sampling.small_object", "head.p2_small_object"],
        ["data.sampler", "model.head"],
    )
    builder = PaperRecipePriorBuilder()
    with pytest.raises(RecipePriorBuildError, match="coupling reason"):
        builder.build(
            paper=_paper(), method_claim=claim, diagnostic_hints=[_hint()],
            component_contracts=contracts, compatibility=_compatibility(),
            research_snapshot=_snapshot(), current_error_facts=[_fact()],
        )
    plan = [
        {"variant": "baseline"},
        {"variant": "sampler"},
        {"variant": "p2"},
        {"variant": "sampler+p2"},
    ]
    prior = builder.build(
        paper=_paper(), method_claim=claim, diagnostic_hints=[_hint()],
        component_contracts=contracts, compatibility=_compatibility(),
        research_snapshot=_snapshot(), current_error_facts=[_fact()],
        coupling_reason="The paper reports the sampler and P2 path as a coupled method.",
        internal_ablation_plan=plan,
    )
    assert prior.coupling_reason
    assert prior.internal_ablation_plan == plan


@pytest.mark.parametrize("snapshot", [_snapshot(available=False), _snapshot(frozen=False)])
def test_snapshot_must_be_available_and_frozen(snapshot: ResearchSnapshot) -> None:
    with pytest.raises(RecipePriorBuildError):
        PaperRecipePriorBuilder().build(
            paper=_paper(), method_claim=_claim(), diagnostic_hints=[_hint()],
            component_contracts={"sampling.small_object": _contract()},
            compatibility=_compatibility(), research_snapshot=snapshot,
            current_error_facts=[_fact()],
        )
