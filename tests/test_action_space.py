"""Unified action space tests.

Pins the unified-action-space contract: the decision object is the whole
detection system's controllable action, one error never maps to a single
hardcoded answer, inference-only actions stay out of the training graph,
data actions are never mislabeled as model components, paper actions keep
their lineage, and local non-paper actions are first-class citizens.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from yolo_agent.agents.action_space import (
    ActionCatalog,
    build_diagnosis_action_chain,
    hypotheses_for_problem_tags,
)
from yolo_agent.agents.action_space_schemas import (
    ACTION_SPACE_SCHEMA_VERSION,
    ActionFamily,
    ActionSpec,
    DATA_FAMILIES,
    INFERENCE_ONLY_FAMILIES,
    LLM_FORBIDDEN_OPERATIONS,
    MODEL_GRAPH_FAMILIES,
    RootCauseHypothesis,
    is_inference_only_family,
    is_train_time_family,
)
from yolo_agent.core.error_facts import ErrorFact

CATALOG_PATH = Path("configs/actions/detection_action_catalog.yaml")

#: Every problem tag the unified diagnosis layer must support.
REQUIRED_PROBLEM_TAGS = [
    "small_object_fn",
    "medium_object_fn",
    "large_object_fn",
    "background_fp",
    "duplicate_detection",
    "classification_confusion",
    "localization_error",
    "low_confidence",
    "overconfidence",
    "long_tail",
    "class_imbalance",
    "label_noise",
    "missing_labels",
    "low_contrast",
    "blur",
    "occlusion",
    "domain_shift",
    "overfitting",
    "underfitting",
    "training_instability",
]


@pytest.fixture(scope="module")
def catalog() -> ActionCatalog:
    return ActionCatalog.from_yaml(CATALOG_PATH)


def _area_fact(area: str) -> ErrorFact:
    return ErrorFact(
        run_id="run-x",
        candidate_id="cand-x",
        node_id="node-x",
        fact_type="area_metric",
        subject=area,
        area=area,
        metric_name=f"ap_{area}",
        value=0.2,
    )


# ------------------------------------------------------------- taxonomy ----


def test_taxonomy_covers_every_prompt_family() -> None:
    expected = {
        "annotation", "data_cleaning", "data_selection", "sampling",
        "augmentation", "preprocessing", "model_scale", "input_resolution",
        "backbone", "neck", "feature_fusion", "head", "assignment",
        "bbox_loss", "classification_loss", "auxiliary_loss",
        "distillation", "domain_adaptation", "semi_supervised",
        "training_strategy", "optimizer", "regularization",
        "postprocess", "threshold", "inference", "calibration",
        "active_learning",
    }
    assert set(ActionFamily.__args__) == expected  # type: ignore[attr-defined]
    assert len(expected) == 27


def test_catalog_covers_all_families_with_paper_and_local(catalog: ActionCatalog) -> None:
    assert {action.family for action in catalog.actions} == set(ActionFamily.__args__)  # type: ignore[attr-defined]
    assert catalog.paper_actions(), "paper-lineage actions must exist"
    assert catalog.local_actions(), "local actions must exist as first-class citizens"
    for action in catalog.paper_actions():
        assert action.paper_ids
        assert action.is_paper_action
    for action in catalog.local_actions():
        assert action.paper_lineage == "local"
        assert not action.paper_ids


def test_catalog_yaml_round_trip(catalog: ActionCatalog, tmp_path: Path) -> None:
    target = tmp_path / "catalog.yaml"
    import yaml

    target.write_text(
        yaml.safe_dump({"actions": [a.model_dump(mode="json") for a in catalog.actions]}),
        encoding="utf-8",
    )
    reloaded = ActionCatalog.from_yaml(target)
    assert [a.action_id for a in reloaded.actions] == [a.action_id for a in catalog.actions]


# ----------------------------------------------------- chain properties ----


def test_small_object_fn_expands_to_multiple_families() -> None:
    hypotheses = hypotheses_for_problem_tags(["small_object_fn"])
    assert len(hypotheses) >= 5
    families = {family for hyp in hypotheses for family in hyp.candidate_action_families}
    assert len(families) >= 6
    # data side, model side, objective side, inference side all reachable
    assert families & DATA_FAMILIES
    assert families & MODEL_GRAPH_FAMILIES
    assert families & {"assignment", "bbox_loss"}
    assert families & INFERENCE_ONLY_FAMILIES


def test_every_required_problem_tag_resolves(catalog: ActionCatalog) -> None:
    for tag in REQUIRED_PROBLEM_TAGS:
        hypotheses = hypotheses_for_problem_tags([tag])
        assert hypotheses, f"no hypothesis covers {tag}"
        specs = catalog.by_problem_tags(
            [tag],
            families=[f for h in hypotheses for f in h.candidate_action_families],
        )
        assert specs, f"no catalog action matches {tag}"


def test_area_facts_expand_to_concrete_actions_on_all_sides(catalog: ActionCatalog) -> None:
    for area in ("small", "medium", "large"):
        chain = build_diagnosis_action_chain([_area_fact(area)], catalog)
        family_hits = {spec.family for sel in chain.selections for spec in sel.action_specs}
        assert family_hits & DATA_FAMILIES, area
        assert family_hits & MODEL_GRAPH_FAMILIES, area


def test_chain_hypotheses_carry_rule_tags_and_unique_ids(catalog: ActionCatalog) -> None:
    chain = build_diagnosis_action_chain([_area_fact("small")], catalog)
    assert chain.problem_tags == sorted(set(chain.problem_tags))
    ids = [h.hypothesis_id for h in chain.hypotheses]
    assert len(ids) == len(set(ids))
    for hypothesis in chain.hypotheses:
        assert hypothesis.problem_tags
        assert hypothesis.origin == "deterministic_rule"


def test_duplicate_error_never_routes_to_a_single_answer(catalog: ActionCatalog) -> None:
    fact = ErrorFact(
        run_id="run-x",
        candidate_id="cand-x",
        node_id="node-x",
        fact_type="duplicate_prediction",
        subject="duplicates",
        count=42,
    )
    chain = build_diagnosis_action_chain([fact], catalog)
    families = {f for h in chain.hypotheses for f in h.candidate_action_families}
    assert len(families) >= 2


def test_uncovered_problem_tag_raises_not_silently_empty(catalog: ActionCatalog) -> None:
    # a genuinely uncovered tag must fail loudly, not return an empty plan
    unmapped_fact = ErrorFact(
        run_id="run-x",
        candidate_id="cand-x",
        node_id="node-x",
        fact_type="subset_performance",
        subject="custom_slice",
    )
    assert not catalog.by_problem_tags(["totally_unknown_tag"])
    with pytest.raises(ValueError, match="no hypothesis rule covers"):
        build_diagnosis_action_chain(
            [unmapped_fact],
            catalog,
            extra_tags=["totally_unknown_tag"],
        )


# --------------------------------------------------------- phase boundary ----


def test_inference_only_actions_never_enter_train_graph(catalog: ActionCatalog) -> None:
    for action in catalog.actions:
        if action.family in INFERENCE_ONLY_FAMILIES:
            assert not action.train_time
            assert action.inference_only
            assert action.runtime_phase in {"inference", "eval"}


def test_train_graph_action_cannot_claim_inference_phase() -> None:
    with pytest.raises(ValueError, match="cannot declare runtime_phase=inference"):
        ActionSpec.model_validate(
            {
                "action_id": "x.bad_train_action",
                "family": "augmentation",
                "title": "Augmentation cannot run at inference",
                "expected_metrics": ["mean_ap"],
                "required_evidence": ["any"],
                "runtime_phase": "inference",
                "rollback": {"strategy": "config_revert", "instructions": "revert", "values": {"a": 1}},
            }
        )


def test_data_actions_are_not_model_components(catalog: ActionCatalog) -> None:
    for action in catalog.actions:
        if action.family in DATA_FAMILIES:
            assert action.family not in MODEL_GRAPH_FAMILIES
            assert action.runtime_phase in {"data", "train"}


def test_model_graph_actions_declare_train_phase(catalog: ActionCatalog) -> None:
    for action in catalog.actions:
        if action.family in MODEL_GRAPH_FAMILIES:
            assert action.train_time
            assert action.runtime_phase == "train"


def test_inference_only_family_helpers() -> None:
    assert is_inference_only_family("postprocess")
    assert not is_inference_only_family("sampling")
    assert is_train_time_family("sampling")
    assert not is_train_time_family("calibration")


# ---------------------------------------------------------------- lineage ----


def test_paper_lineage_requires_real_paper_ids(catalog: ActionCatalog) -> None:
    known_ids = {
        "arxiv:2212.07784",  # RTMDet
        "arxiv:2309.11331",  # Gold-YOLO
        "arxiv:2108.07755",  # TOOD
        "arxiv:2103.14259",  # OTA
        "arxiv:2208.00817",  # DSLA
        "arxiv:2301.01019",  # Correlation loss
        "arxiv:2303.14404",  # BPC loss
        "arxiv:2109.05986",  # Mutual supervision
        "arxiv:2104.14082",  # Pseudo-IoU
        "arxiv:2503.23220",  # DA branch
        "cvf:cvpr2021:Guo_Distilling_Object_Detectors_via_Decoupled_Features",
    }
    for action in catalog.paper_actions():
        assert set(action.paper_ids) & known_ids, action.action_id


def test_paper_ids_without_paper_lineage_rejected() -> None:
    with pytest.raises(ValueError, match="lists paper_ids but declares local lineage"):
        ActionSpec.model_validate(
            {
                "action_id": "x.bad_local",
                "family": "sampling",
                "title": "Local action cannot cite papers",
                "paper_ids": ["arxiv:2103.14259"],
                "expected_metrics": ["mean_ap"],
                "required_evidence": ["any"],
                "runtime_phase": "data",
                "rollback": {"strategy": "config_revert", "instructions": "revert", "values": {"a": 1}},
            }
        )


def test_paper_lineage_without_paper_ids_rejected() -> None:
    with pytest.raises(ValueError, match="declares paper lineage without paper_ids"):
        ActionSpec.model_validate(
            {
                "action_id": "x.bad_paper",
                "family": "assignment",
                "title": "Paper action must cite papers",
                "paper_lineage": "paper",
                "expected_metrics": ["mean_ap"],
                "required_evidence": ["any"],
                "runtime_phase": "train",
                "rollback": {"strategy": "config_revert", "instructions": "revert", "values": {"a": 1}},
            }
        )


def test_local_action_is_valid_first_class(catalog: ActionCatalog) -> None:
    local = catalog.by_family("threshold")
    assert local and all(a.paper_lineage == "local" for a in local)


# ------------------------------------------------- deterministic boundary ----


def test_llm_boundary_forbids_guarantee_operations() -> None:
    from yolo_agent.agents.action_space_schemas import LLM_ALLOWED_OPERATIONS, LLMActionBoundary

    boundary = LLMActionBoundary(allowed_operations=sorted(LLM_ALLOWED_OPERATIONS))
    assert boundary.propose("generate_hypothesis")
    assert boundary.propose("rank_candidates")
    for forbidden in LLM_FORBIDDEN_OPERATIONS:
        assert not boundary.propose(forbidden)


def test_llm_boundary_rejects_forbidden_allowlist() -> None:
    import pydantic

    with pytest.raises(pydantic.ValidationError):
        from yolo_agent.agents.action_space_schemas import LLMActionBoundary

        LLMActionBoundary(allowed_operations=["mark_implementation_ready"])


def test_hypothesis_origin_vocabulary_keeps_llm_on_proposal_side() -> None:
    proposal = RootCauseHypothesis.model_validate(
        {
            "hypothesis_id": "h.llm_idea",
            "description": "LLM may propose; it may not authorize.",
            "candidate_action_families": ["sampling"],
            "origin": "llm_proposal",
        }
    )
    assert proposal.origin == "llm_proposal"
    assert proposal.origin != "deterministic_rule"


def test_schema_version_is_frozen_constant() -> None:
    assert ACTION_SPACE_SCHEMA_VERSION == "detection_action_space.v1"
