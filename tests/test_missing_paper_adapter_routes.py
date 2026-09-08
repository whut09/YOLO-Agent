"""Regression coverage for the paper-level adapter route boundary."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from yolo_agent.components.adapters import AdapterContext
from yolo_agent.components.adapters.registry import ComponentAdapterRegistry
from yolo_agent.components.adapters.validation import validate_runtime_plugin_hooks
from yolo_agent.components.adapters.audit_contract import (
    validate_audited_runtime_payload,
)
from yolo_agent.components.contracts import load_contracts
from yolo_agent.research.paper_execution_requirements import (
    PaperExecutionRequirementsBuilder,
)
from yolo_agent.research.paper_execution_schemas import PaperExecutionInventory


ROOT = Path(__file__).resolve().parents[1]
INVENTORY_PATH = ROOT / "runs" / "coverage-audit" / "paper_execution_inventory.yaml"


@pytest.fixture(scope="module")
def requirements():  # type: ignore[no-untyped-def]
    inventory = PaperExecutionInventory.from_yaml(INVENTORY_PATH)
    return PaperExecutionRequirementsBuilder().build(
        inventory,
        source_inventory_path=INVENTORY_PATH,
    )


def test_production_missing_adapter_scan_is_empty(requirements) -> None:  # type: ignore[no-untyped-def]
    missing = [
        item
        for item in requirements.requirements
        if not item.required_adapter
    ]
    assert missing == []


def test_production_unresolved_routes_have_specific_blockers(requirements) -> None:  # type: ignore[no-untyped-def]
    for row in requirements.requirements:
        if row.training_candidate_allowed:
            continue
        assert row.exact_blocker
        assert "unknown" not in row.exact_blocker.lower()


def test_distillation_routes_keep_independent_adapter_identity(requirements) -> None:  # type: ignore[no-untyped-def]
    rows = [
        item
        for item in requirements.requirements
        if any(
            mechanism.startswith("distillation.")
            for mechanism in item.paper_specific_mechanism_ids
        )
    ]
    assert len(rows) == 32
    modules = [
        importlib.import_module(
            "yolo_agent.components.adapters.distillation.paper_routes"
        ),
        importlib.import_module(
            "yolo_agent.components.adapters.domain_adaptation.domain_paper_routes"
        ),
    ]
    for row in rows:
        assert row.required_adapter
        assert row.required_changed_variables
        assert row.required_runtime_payload
        assert row.recipe_ids
        assert not row.training_candidate_allowed
        adapter_type = next(
            (
                getattr(module, row.required_adapter, None)
                for module in modules
                if getattr(module, row.required_adapter, None) is not None
            ),
            None,
        )
        assert adapter_type is not None, (
            f"missing paper route adapter {row.required_adapter} "
            f"for {row.paper_id}"
        )
        assert adapter_type.__name__ == row.required_adapter


def test_distillation_requirements_keep_paper_route_fingerprints(
    requirements,
) -> None:  # type: ignore[no-untyped-def]
    rows = [
        item
        for item in requirements.requirements
        if any(
            mechanism.startswith("distillation.")
            for mechanism in item.paper_specific_mechanism_ids
        )
    ]
    assert len(rows) == 32
    assert len({item.execution_fingerprint for item in rows}) == len(rows)


def test_identity_recovery_routes_bind_identity_before_cpu_smoke(
    requirements, tmp_path: Path
) -> None:  # type: ignore[no-untyped-def]
    """The route exists, but unresolved paper identity remains non-trainable."""
    from yolo_agent.components.adapters.distillation.paper_routes import (
        default_paper_route_registry,
    )

    contracts = {}
    for path in (ROOT / "configs" / "components").glob("*/*.yaml"):
        contracts.update({item.component_id: item for item in load_contracts(path)})
    rows = [
        item
        for item in requirements.requirements
        if item.paper_specific_mechanism.startswith("distillation.")
        and "distillation_branch_unmapped" in (item.exact_blocker or "")
    ]
    assert len(rows) == 13
    routes = default_paper_route_registry()
    module = importlib.import_module(
        "yolo_agent.components.adapters.distillation.paper_routes"
    )
    for row in rows:
        route = routes.route(row.paper_id)
        contract = contracts[row.paper_specific_mechanism]
        adapter_type = getattr(module, row.required_adapter)
        context = AdapterContext(
            contract=contract,
            detector_family="yolo26",
            head="one_to_one",
            imgsz=640,
            workspace=tmp_path,
            options={
                "paper_id": row.paper_id,
                "paper_route_fingerprint": route.execution_fingerprint,
            },
        )
        smoke = adapter_type().smoke_test(context)
        assert smoke.passed, (row.paper_id, smoke.errors)
        payload = adapter_type().build_runtime_payload(
            context,
            protocol_hash="p" * 64,
            base_command=["python", "-m", "ultralytics", "train"],
            generated_config={"imgsz": 640},
        )
        assert payload.component_ids == [row.paper_specific_mechanism]
        assert payload.adapter_classes == [row.required_adapter]
        assert row.paper_id in payload.loss_plugin[0].options["paper_id"]
        assert row.current_disposition == "implementation_request"
        assert not row.training_candidate_allowed
        assert "paper_method_identity_missing" in (row.exact_blocker or "")


def test_mutual_supervision_route_has_contract_payload_and_cpu_smoke(
    tmp_path: Path,
) -> None:
    contract = load_contracts(
        ROOT / "configs" / "components" / "loss" / "mutual_supervision.yaml"
    )[0]
    adapter = ComponentAdapterRegistry().create_for_contract(contract)
    context = AdapterContext(
        contract=contract,
        detector_family="yolo26",
        head="one_to_one",
        imgsz=640,
        workspace=tmp_path,
        options={"paper_id": "arxiv:2109.05986"},
    )

    compatibility = adapter.validate_compatibility(context)
    assert compatibility.ok, compatibility.errors
    smoke = adapter.smoke_test(context)
    assert smoke.passed, smoke.errors
    payload = adapter.build_runtime_payload(
        context,
        protocol_hash="mutual-supervision-protocol",
        base_command=["yolo", "detect", "train", "model=yolo26n.pt", "imgsz=640"],
        generated_config={"imgsz": 640},
    )
    assert payload.changed_variables == {
        "loss.mutual_supervision.weight": 0.1
    }
    payload.verify_imports()
    hooks = validate_runtime_plugin_hooks(payload)
    assert hooks["runtime_plugin_hooks_verified"] >= 1
    assert validate_audited_runtime_payload(
        payload, "loss.2109_05986"
    )["audited_runtime_component"] is True


def test_mutual_supervision_route_rejects_unsafe_protocol_inputs(
    tmp_path: Path,
) -> None:
    contract = load_contracts(
        ROOT / "configs" / "components" / "loss" / "mutual_supervision.yaml"
    )[0]
    adapter = ComponentAdapterRegistry().create_for_contract(contract)
    invalid_context = AdapterContext(
        contract=contract,
        detector_family="yolo26",
        head="one_to_one",
        imgsz=608,
        workspace=tmp_path,
    )
    result = adapter.validate_compatibility(invalid_context)
    assert not result.ok
    assert "imgsz=640" in " ".join(result.errors)

    with pytest.raises(ValueError, match="imgsz=640"):
        adapter.build_runtime_payload(
            invalid_context,
            protocol_hash="invalid",
            base_command=["yolo", "detect", "train", "imgsz=608"],
            generated_config={},
        )
