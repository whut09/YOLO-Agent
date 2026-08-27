"""CPU tests for paper-specific domain-adaptation source-target routes.

Covers the 40 per-paper routes, the eight canonical mechanism contracts,
payload and changed-variable independence, and fail-closed certification
without GPU training.  COCO may never stand in for a paper domain.
"""

from __future__ import annotations

import hashlib
import importlib
from pathlib import Path

import pytest
import yaml

from yolo_agent.components.adapters.base import AdapterContext
from yolo_agent.components.adapters.domain_adaptation import (
    DomainAdaptationMethodRegistry,
    DomainPaperRoute,
    build_domain_paper_routes,
    default_domain_paper_route_registry,
    domain_paper_route_coverage,
)
from yolo_agent.components.adapters.domain_adaptation.branches import (
    BASE_PAYLOAD_SCHEMA,
    CANONICAL_DOMAIN_BRANCHES,
    DOMAIN_BRANCH_PROFILES,
    NAMED_PAPER_BRANCHES,
    build_branch,
)
from yolo_agent.components.adapters.domain_adaptation.domain_evidence import (
    manifest_from_file,
    resolve_domain_protocol,
)
from yolo_agent.components.contracts import ComponentContract, load_contracts


REPO_ROOT = Path(__file__).resolve().parents[1]
RECIPES_PATH = REPO_ROOT / "configs" / "recipes" / "paper_specific_methods.yaml"
CONTRACTS_PATH = REPO_ROOT / "configs" / "components" / "domain_adaptation"

FEATURE_PAPER_A = "arxiv:2303.13853"
FEATURE_PAPER_B = (
    "cvf:cvpr2022:Zhou_Multi-Granularity_Alignment_Domain_Adaptation_"
    "for_Object_Detection"
)
SOURCE_FREE_PAPER = (
    "cvf:cvpr2023:VS_Instance_Relation_Graph_Guided_Source-Free_Domain_"
    "Adaptive_Object_Detection"
)
TEACHER_PAPER = "cvf:cvpr2022:Li_Cross-Domain_Adaptive_Teacher_for_Object_Detection"
DATASET_HASH_SOURCE = "s" * 64
DATASET_HASH_TARGET = "t" * 64


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def routes() -> list[DomainPaperRoute]:
    return build_domain_paper_routes()


@pytest.fixture(scope="module")
def registry():
    return default_domain_paper_route_registry()


def test_all_40_papers_own_exactly_one_route(routes) -> None:
    assert len(routes) == 40
    assert {item.paper_id for item in routes} == set(NAMED_PAPER_BRANCHES)
    assert len({item.component_id for item in routes}) == 40
    assert len({item.recipe_id for item in routes}) == 40
    assert len({item.paper_specific_mechanism_id for item in routes}) == 40
    assert len({item.method_profile_id for item in routes}) == 40


def test_route_identities_are_independent_per_paper(routes) -> None:
    assert len({item.execution_fingerprint for item in routes}) == 40
    assert len({item.protocol_hash for item in routes}) == 40
    assert len({item.adapter_class for item in routes}) == 40
    assert len({item.adapter_hash for item in routes}) == 40
    assert len({item.adapter_version for item in routes}) == 40


def test_routes_match_paper_specific_recipes(routes) -> None:
    raw = yaml.safe_load(RECIPES_PATH.read_text(encoding="utf-8-sig"))
    recipes = {item["recipe_id"]: item for item in raw["recipes"]}
    for route in routes:
        recipe = recipes[route.recipe_id]
        assert (
            recipe["paper_specific_mechanism_id"]
            == route.paper_specific_mechanism_id
        )
        assert route.paper_id in recipe["paper_ids"]
        assert route.component_id in recipe["component_ids"]
        assert (
            recipe["train_overrides"]["domain_adaptation.method"]
            == route.paper_specific_mechanism_id
        )
        assert route.method_profile_id in recipe["method_profile_ids"]


def test_no_generic_adapter_collapse(routes) -> None:
    branch_components = {
        f"domain_adaptation.{branch_id}" for branch_id in CANONICAL_DOMAIN_BRANCHES
    }
    for route in routes:
        assert route.component_id == route.paper_specific_mechanism_id
        assert route.component_id.startswith("domain_adaptation.")
        assert route.component_id not in branch_components
        assert route.adapter_class != "DomainAdaptationBranchAdapter"
        assert route.adapter_class.startswith("DomainAdaptation")
        assert route.adapter_class.endswith("Adapter")
        assert route.branch_component_id in branch_components


def test_route_source_target_protocols_are_fail_closed(routes) -> None:
    for route in routes:
        source = route.source_protocol
        target = route.target_protocol
        assert target["manifest_required"] is True
        assert target["sha256_required"] is True
        assert target["dataset_hash_required"] is True
        assert target["split_required"] is True
        assert target["coco_supervised_forbidden"] is True
        assert target["mock_forbidden"] is True
        assert target["imgsz"] == 640
        if route.source_free:
            assert source["manifest_required"] is False
            assert route.requires_source_domain is False
            assert route.adaptation_mode == "source_free"
        else:
            assert source["manifest_required"] is True
            assert source["label_availability"] == "labeled"
            assert source["coco_supervised_forbidden"] is True
            assert route.requires_source_domain is True


def test_eight_canonical_mechanisms_have_independent_contracts() -> None:
    contracts = load_contracts(CONTRACTS_PATH)
    by_component = {item.component_id: item for item in contracts}
    fingerprints = set()
    changed_variables = set()
    for branch_id in CANONICAL_DOMAIN_BRANCHES:
        component_id = f"domain_adaptation.{branch_id}"
        contract = by_component[component_id]
        assert contract.implementation_family == (
            f"domain_adaptation.{branch_id}"
        )
        module = importlib.import_module(contract.implementation_path)
        assert getattr(module, contract.adapter_class) is not None
        spec = build_branch(branch_id)
        assert spec.execution_fingerprint
        assert spec.execution_fingerprint not in fingerprints
        fingerprints.add(spec.execution_fingerprint)
        assert (
            spec.changed_variable
            == DOMAIN_BRANCH_PROFILES[branch_id]["changed_variable"]
        )
        assert spec.changed_variable.startswith("loss.domain_")
        assert spec.changed_variable not in changed_variables
        changed_variables.add(spec.changed_variable)
        for key in BASE_PAYLOAD_SCHEMA:
            assert key in spec.payload_schema
        for key in DOMAIN_BRANCH_PROFILES[branch_id]["payload_schema"]:
            assert key in spec.payload_schema
        assert spec.coco_as_domain_allowed is False
        assert spec.adapter_alone_authorizes_asha is False


def test_eight_mechanism_losses_backward_on_cpu() -> None:
    import torch

    from yolo_agent.components.adapters.domain_adaptation.branch_runtime import (
        DomainAdaptationBranchPlugin,
    )

    from yolo_agent.components.adapters.domain_adaptation.domain_evidence import (
        DomainDatasetManifest,
    )

    source = DomainDatasetManifest(
        path="source-domain.yaml",
        sha256="source-domain-sha",
        dataset_hash=DATASET_HASH_SOURCE,
        domain_id="0",
        domain_name="source",
        role="source",
        split="source_train",
        label_availability="labeled",
    )
    target = DomainDatasetManifest(
        path="target-domain.yaml",
        sha256="target-domain-sha",
        dataset_hash=DATASET_HASH_TARGET,
        domain_id="1",
        domain_name="target",
        role="target",
        split="target_train",
        label_availability="unlabeled",
    )
    protocol = resolve_domain_protocol(
        source=source,
        target=target,
        adaptation_mode="unsupervised",
    ).model_dump(mode="json")
    batch_evidence = {
        "pseudo_label_adaptation": {"pseudo_labels": torch.tensor([0.2, 0.4])},
        "domain_distillation": {"teacher_features": torch.randn(2, 3)},
        "cross_domain_teacher": {"teacher_features": torch.randn(2, 3)},
        "source_free_adaptation": {"source_model_features": torch.randn(2, 3)},
        "contrastive_domain_alignment": {
            "contrastive_pairs": torch.randn(2, 2, 3),
        },
        "active_domain_adaptation": {"query_ids": torch.tensor([1, 0])},
    }
    for branch_id in CANONICAL_DOMAIN_BRANCHES:
        branch = build_branch(branch_id)
        options: dict = {
            "branch_id": branch_id,
            "weight": 0.1,
            "source_manifest": "source-domain.yaml",
            "target_manifest": "target-domain.yaml",
            "domain_protocol": protocol,
            "imgsz": 640,
            "runtime_strategy": branch.runtime_strategy,
            "source_domain_id": 0,
            "target_domain_id": 1,
        }
        if branch_id == "pseudo_label_adaptation":
            options["pseudo_label_manifest"] = "pseudo.yaml"
        elif branch_id in {"domain_distillation", "cross_domain_teacher"}:
            options["teacher_checkpoint"] = "teacher.pt"
            options["teacher_sha256"] = "t" * 64
        elif branch_id == "source_free_adaptation":
            options["source_model_checkpoint"] = "source-model.pt"
            options["source_model_sha256"] = "s" * 64
        elif branch_id == "contrastive_domain_alignment":
            options["contrastive_pair_manifest"] = "pairs.yaml"
            options["temperature"] = 0.1
        elif branch_id == "active_domain_adaptation":
            options["query_manifest"] = "queries.yaml"
            options["label_budget"] = 2
        plugin = DomainAdaptationBranchPlugin(**options)
        if branch_id == "adversarial_alignment":
            plugin.build_model(
                context=None,
                trainer=None,
                model=torch.nn.Linear(3, 2),
            )
        features = [torch.randn(4, 3, 2, 2, requires_grad=True)]
        domains = torch.tensor([0, 0, 1, 1])
        if branch_id == "adversarial_alignment":
            loss = plugin.compute_loss(features, domains)
        else:
            loss = plugin._compute_runtime_strategy_loss(
                features,
                domains,
                batch=batch_evidence.get(branch_id),
                device=features[0].device,
            )
        assert torch.isfinite(loss)
        loss.backward()
        assert features[0].grad is not None
        features[0].grad.zero_()


def test_branch_assignment_covers_all_40_papers() -> None:
    registry = DomainAdaptationMethodRegistry()
    coverage = registry.coverage()
    assert coverage.papers_total == 40
    assert coverage.silent_drops == []


def test_paper_route_coverage_has_no_silent_drops(routes) -> None:
    coverage = domain_paper_route_coverage()
    assert coverage.papers_total == 40
    assert coverage.source_bound == 36
    assert coverage.source_free == 4
    assert coverage.silent_drops == []
    assert {item.paper_id for item in coverage.routes} == set(
        NAMED_PAPER_BRANCHES
    )


def test_coverage_rejects_silent_drops(routes) -> None:
    from yolo_agent.components.adapters.domain_adaptation.domain_paper_routes import (
        DomainPaperRouteCoverage,
    )

    with pytest.raises(Exception, match="silent drops"):
        DomainPaperRouteCoverage.model_validate(
            {
                "papers_total": 40,
                "source_bound": 36,
                "source_free": 4,
                "routes": [item.model_dump() for item in routes],
                "silent_drops": [FEATURE_PAPER_A],
            }
        )


def test_route_payload_schema_rejects_missing_keys(routes) -> None:
    
    route = routes[0]
    data = route.model_dump()
    data["execution_fingerprint"] = ""
    data["protocol_hash"] = ""
    del data["route_payload_schema"]["domain_pair_id"]
    with pytest.raises(ValueError, match="domain_pair_id"):
        DomainPaperRoute.model_validate(data)


def test_route_rejects_branch_component_reuse(routes) -> None:
    
    route = next(item for item in routes if item.source_free)
    data = route.model_dump()
    data["execution_fingerprint"] = ""
    data["protocol_hash"] = ""
    data["component_id"] = data["branch_component_id"]
    data["paper_specific_mechanism_id"] = data["branch_component_id"]
    with pytest.raises(ValueError, match="must not reuse the branch"):
        DomainPaperRoute.model_validate(data)


def test_route_rejects_missing_changed_variable(routes) -> None:
    
    route = routes[0]
    data = route.model_dump()
    data["execution_fingerprint"] = ""
    data["protocol_hash"] = ""
    data["changed_variables"] = {}
    with pytest.raises(ValueError, match="changed variables"):
        DomainPaperRoute.model_validate(data)


def test_route_rejects_fingerprint_mismatch(routes) -> None:
    
    route = routes[0]
    data = route.model_dump()
    data["execution_fingerprint"] = "f" * 64
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        DomainPaperRoute.model_validate(data)


def _domain_context(workspace: Path, route: DomainPaperRoute, options: dict):
    contract = ComponentContract(
        component_id=route.component_id,
        display_name=route.paper_id,
        category="domain_adaptation",
    )
    return AdapterContext(
        contract=contract,
        detector_family="yolo26",
        workspace=workspace,
        options=options,
    )


def _fixture_protocol(workspace: Path):
    source = manifest_from_file(
        workspace / "source.yaml",
        role="source",
        dataset_hash=DATASET_HASH_SOURCE,
        domain_id="city",
        domain_name="City",
        split="train",
        label_availability="labeled",
    )
    target = manifest_from_file(
        workspace / "target.yaml",
        role="target",
        dataset_hash=DATASET_HASH_TARGET,
        domain_id="night",
        domain_name="Night",
        split="train",
        label_availability="unlabeled",
    )
    return resolve_domain_protocol(
        source=source,
        target=target,
        adaptation_mode="unsupervised",
    )


@pytest.fixture()
def domain_workspace(tmp_path: Path) -> Path:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "source.yaml").write_text(
        "source domain manifest", encoding="utf-8"
    )
    (workspace / "target.yaml").write_text(
        "target domain manifest", encoding="utf-8"
    )
    return workspace


def test_per_paper_payloads_are_independent(
    registry, domain_workspace
) -> None:
    from yolo_agent.components.adapters.domain_adaptation import (
        create_domain_paper_route_adapter,
    )

    protocol = _fixture_protocol(domain_workspace)
    payloads = {}
    for paper_id in (FEATURE_PAPER_A, FEATURE_PAPER_B):
        route = registry.route(paper_id)
        adapter_cls = create_domain_paper_route_adapter(route)
        context = _domain_context(
            domain_workspace,
            route,
            {
                "domain_protocol": protocol,
                "weight": 0.05,
                "source_manifest": str(domain_workspace / "source.yaml"),
                "target_manifest": str(domain_workspace / "target.yaml"),
            },
        )
        payload = adapter_cls().build_runtime_payload(
            context,
            protocol_hash="p" * 64,
            base_command=["train"],
            generated_config={},
        )
        payloads[paper_id] = (route, payload)
    (route_a, payload_a), (route_b, payload_b) = (
        payloads[FEATURE_PAPER_A],
        payloads[FEATURE_PAPER_B],
    )
    assert payload_a.component_ids == [route_a.component_id]
    assert payload_b.component_ids == [route_b.component_id]
    assert payload_a.component_ids != payload_b.component_ids
    assert payload_a.adapter_classes == [route_a.adapter_class]
    assert payload_b.adapter_classes == [route_b.adapter_class]
    assert (
        payload_a.changed_variables["domain_adaptation.method"]
        == route_a.paper_specific_mechanism_id
    )
    assert (
        payload_b.changed_variables["domain_adaptation.method"]
        == route_b.paper_specific_mechanism_id
    )
    assert payload_a.payload_hash != payload_b.payload_hash

    options = payload_a.loss_plugin[0].options
    assert options["paper_id"] == FEATURE_PAPER_A
    assert options["paper_route_fingerprint"] == route_a.execution_fingerprint
    assert options["paper_component_id"] == route_a.component_id
    assert options["branch_id"] == route_a.branch_id
    assert options["domain_pair_id"] == "city->night"
    assert options["domain_protocol_hash"] == protocol.protocol_hash


def test_source_free_paper_payload_stays_paper_bound(
    registry, domain_workspace
) -> None:
    from yolo_agent.components.adapters.domain_adaptation import (
        create_domain_paper_route_adapter,
    )

    route = registry.route(SOURCE_FREE_PAPER)
    target = manifest_from_file(
        domain_workspace / "target.yaml",
        role="target",
        dataset_hash=DATASET_HASH_TARGET,
        domain_id="night",
        domain_name="Night",
        split="train",
        label_availability="unlabeled",
    )
    protocol = resolve_domain_protocol(
        source=None,
        target=target,
        adaptation_mode="source_free",
        source_free=True,
        source_model_checkpoint_sha256="m" * 64,
        source_model_protocol_hash="p" * 64,
    )
    adapter_cls = create_domain_paper_route_adapter(route)
    context = _domain_context(
        domain_workspace,
        route,
        {
            "domain_protocol": protocol,
            "weight": 0.05,
            "target_manifest": str(domain_workspace / "target.yaml"),
            "source_free": True,
            "source_model_checkpoint": "frozen-source-model.pt",
            "source_model_sha256": "m" * 64,
            "source_model_protocol_hash": "p" * 64,
        },
    )
    payload = adapter_cls().build_runtime_payload(
        context,
        protocol_hash="p" * 64,
        base_command=["train"],
        generated_config={},
    )
    options = payload.loss_plugin[0].options
    assert payload.component_ids == [route.component_id]
    assert payload.adapter_classes == [route.adapter_class]
    assert options["paper_id"] == SOURCE_FREE_PAPER
    assert options["paper_route_fingerprint"] == route.execution_fingerprint
    assert options["source_free"] is True
    assert (
        options["source_model_checkpoint_sha256"] == "m" * 64
    )
