"""Materialization and ASHA protocol gate tests. No GPU training."""

from __future__ import annotations

from yolo_agent.agents.auto_optimization_loop import _paper_protocol_runtime_blockers
from yolo_agent.agents.candidate_generator import CandidateConfig
from yolo_agent.agents.paper_recipe_materialization.gate import _paper_protocol_block
from yolo_agent.core.experiment_graph import ExperimentNode
from yolo_agent.research.paper_protocol_contract import (
    PaperProtocolRegistry,
    authorize_paper_ids_or_missing,
)
from tests.paper_materialization_fixtures import prior


def test_dummy_prior_does_not_require_catalog_protocol() -> None:
    assert _paper_protocol_block(
        type("Item", (), {"prior": prior(), "source_node": None})()
    ) is None


def test_certified_domain_paper_blocks_materialization_without_domains() -> None:
    item_prior = prior()
    item_prior.paper_ids = [
        "cvf:cvpr2022:Li_SIGMA_Semantic-Complete_Graph_Matching_for_Domain_Adaptive_Object_Detection"
    ]
    blocked = _paper_protocol_block(type("Item", (), {"prior": item_prior, "source_node": None})())
    assert blocked is not None
    assert blocked.allows_materialization is False
    assert blocked.disposition == "evidence_recovery"


def test_missing_real_paper_protocol_blocks_asha() -> None:
    node = ExperimentNode(
        node_id="n1",
        candidate_config=CandidateConfig(
            candidate_id="c1",
            base_model="yolo26n",
            scale="n",
            framework="ultralytics",
            components=["dummy.component"],
        ),
        data_version="v1",
        changed_variables={"paper_ids": ["arxiv:0000.00000"]},
    )
    blocked = _paper_protocol_runtime_blockers(node)
    assert blocked is not None
    assert blocked.reason_codes == ["paper_protocol_missing"]
    assert blocked.allows_asha_registration is False


def test_inference_component_cannot_enter_training_asha() -> None:
    node = ExperimentNode(
        node_id="n2",
        candidate_config=CandidateConfig(
            candidate_id="sahi",
            base_model="yolo26n",
            scale="n",
            framework="ultralytics",
            components=["inference.sahi_slicing"],
        ),
        data_version="v1",
    )
    blocked = _paper_protocol_runtime_blockers(node)
    assert blocked is not None
    assert blocked.execution_class == "inference_candidate"
    assert blocked.allows_asha_registration is False


def test_empty_registry_blocks_catalog_paper() -> None:
    blocked = authorize_paper_ids_or_missing(
        ["arxiv:2103.14259"],
        registry=PaperProtocolRegistry(),
    )
    assert blocked is not None
    assert blocked.reason_codes == ["paper_protocol_missing"]
