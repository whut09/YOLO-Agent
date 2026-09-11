from __future__ import annotations

from pathlib import Path

import pytest
from torch import nn

from yolo_agent.components.adapters import (
    AdapterContext,
    AdapterValidationReport,
    ComponentAdapterRegistry,
    DummyAdapter,
    LossAdapter,
    RuntimeAdapterContractError,
    RuntimeAdapterFacade,
    RuntimeAdapterProtocol,
    inspect_fake_implementation,
    validate_runtime_adapter,
)
from yolo_agent.components.contracts import ComponentContract


def _contract(*, runtime_hook: str | None = "build_model") -> ComponentContract:
    return ComponentContract(
        component_id="loss.synthetic_runtime",
        display_name="Synthetic runtime loss",
        category="loss",
        source_papers=["paper-a", "paper-b"],
        implementation_path="yolo_agent.components.adapters.dummy",
        adapter_class="DummyAdapter",
        changed_variable="training.adapter_marker",
        insertion_point="trainer.loss",
        runtime_hook=runtime_hook,
        supported_detector_families=["yolo26"],
        supported_yolo_versions=["26"],
        fixed_imgsz_compatible=True,
    )


def _context(contract: ComponentContract, tmp_path: Path) -> AdapterContext:
    return AdapterContext(
        contract=contract,
        detector_family="yolo26",
        head="one_to_one",
        imgsz=640,
        workspace=tmp_path,
    )


def test_runtime_facade_roundtrips_payload_and_exposes_common_lifecycle(
    tmp_path: Path,
) -> None:
    contract = _contract()
    facade = RuntimeAdapterFacade(
        DummyAdapter(),
        contract,
        _context(contract, tmp_path),
        protocol_hash="protocol-1",
        supported_paper_ids=["paper-a", "paper-b"],
    )

    payload = facade.typed_runtime_payload()
    restored = type(payload).model_validate(payload.model_dump(mode="json"))
    assert restored.payload_hash == payload.payload_hash
    assert facade.metadata.adapter_id == "loss.synthetic_runtime"
    assert facade.metadata.supported_paper_ids == ["paper-a", "paper-b"]
    assert facade.metadata.supported_yolo_versions == ["26"]
    assert isinstance(facade, RuntimeAdapterProtocol)
    assert isinstance(facade, LossAdapter)
    assert facade.validate_contract().ok

    module = facade.build()
    assert module["component_id"] == "loss.synthetic_runtime"
    preview = facade.apply({"model": "yolo26n.pt"}, {"imgsz": 640})
    assert preview.operations[0].field == "adapter_marker"
    assert facade.rollback().actions == ["discard generated adapter patch"]


def test_registry_creates_facade_without_breaking_legacy_factory(
    tmp_path: Path,
) -> None:
    registry = ComponentAdapterRegistry()
    contract = _contract()
    context = _context(contract, tmp_path)

    legacy = registry.create_for_contract(contract)
    runtime = registry.create_runtime_adapter(
        contract,
        context,
        protocol_hash="protocol-1",
        supported_paper_ids=["paper-a"],
    )

    assert isinstance(legacy, DummyAdapter)
    assert isinstance(runtime, RuntimeAdapterFacade)
    assert runtime.supported_paper_ids == ["paper-a"]


def test_fingerprint_changes_with_paper_configuration_and_protocol(
    tmp_path: Path,
) -> None:
    contract = _contract()
    facade = RuntimeAdapterFacade(
        DummyAdapter(),
        contract,
        _context(contract, tmp_path),
        protocol_hash="protocol-1",
        supported_paper_ids=["paper-a", "paper-b"],
    )
    payload = facade.typed_runtime_payload()

    changed_context = _context(contract, tmp_path).model_copy(
        update={"options": {"loss_weight": 0.5}}
    )
    changed_configuration = RuntimeAdapterFacade(
        DummyAdapter(),
        contract,
        changed_context,
        protocol_hash="protocol-1",
        supported_paper_ids=["paper-a", "paper-b"],
    )
    assert facade.fingerprint(payload=payload) != changed_configuration.fingerprint(
        payload=changed_configuration.typed_runtime_payload()
    )
    assert facade.fingerprint(configuration={"paper_id": "paper-a"}, payload=payload) != facade.fingerprint(
        configuration={"paper_id": "paper-b"}, payload=payload
    )
    other_protocol = RuntimeAdapterFacade(
        DummyAdapter(),
        contract,
        _context(contract, tmp_path),
        protocol_hash="protocol-2",
        supported_paper_ids=["paper-a", "paper-b"],
    )
    assert facade.fingerprint(configuration={"paper_id": "paper-a"}, payload=payload) != other_protocol.fingerprint(
        configuration={"paper_id": "paper-a"}, payload=other_protocol.typed_runtime_payload()
    )


def test_runtime_hook_must_be_registered_by_payload(tmp_path: Path) -> None:
    contract = _contract(runtime_hook="compute_loss")
    facade = RuntimeAdapterFacade(
        DummyAdapter(),
        contract,
        _context(contract, tmp_path),
        protocol_hash="protocol-1",
    )

    report = facade.validate_contract()
    assert not report.ok
    assert any("declared runtime hook is not registered" in error for error in report.errors)


def test_supported_yolo_version_is_part_of_runtime_compatibility(
    tmp_path: Path,
) -> None:
    contract = _contract()
    facade = RuntimeAdapterFacade(DummyAdapter(), contract, _context(contract, tmp_path))
    incompatible_context = _context(contract, tmp_path).model_copy(
        update={"yolo_version": "25"}
    )
    incompatible = RuntimeAdapterFacade(DummyAdapter(), contract, incompatible_context)

    assert facade.validate_contract().ok
    report = incompatible.validate_contract()
    assert not report.ok
    assert "yolo_version_unsupported:25" in report.errors


def test_generic_detector_family_remains_a_wildcard(tmp_path: Path) -> None:
    contract = _contract().model_copy(update={"supported_detector_families": ["generic"]})
    facade = RuntimeAdapterFacade(DummyAdapter(), contract, _context(contract, tmp_path))

    report = facade.validate_contract()

    assert report.ok
    assert report.checks["detector_family_supported"] is True


def test_mock_smoke_cannot_become_non_mock_readiness(tmp_path: Path) -> None:
    contract = _contract()
    facade = RuntimeAdapterFacade(DummyAdapter(), contract, _context(contract, tmp_path))

    smoke = facade.validate_non_mock_smoke()
    assert not smoke.passed
    assert "mock_smoke_provenance_rejected" in smoke.errors

    report = validate_runtime_adapter(
        DummyAdapter(),
        contract=contract,
        context=_context(contract, tmp_path),
        check_smoke=True,
    )
    assert not report.ok
    assert report.checks["non_mock_smoke"] is False


class _IdentityAdapter(DummyAdapter):
    def build_module(self, context: AdapterContext) -> nn.Module:
        return nn.Identity()


class _PassthroughAdapter(DummyAdapter):
    def apply(self, value: object) -> object:
        return value


class _ConstantLossAdapter(DummyAdapter):
    def compute_loss(self, predictions: object) -> int:
        return 0


class _IntentionalPassthroughAdapter(DummyAdapter):
    intentional_passthrough = True
    modified_training_fields = frozenset()

    def patch_training_config(
        self,
        config: dict[str, object],
        context: AdapterContext,
        *,
        dry_run: bool = True,
    ) -> dict[str, object]:
        return config


class _ExplodingAdapter(DummyAdapter):
    def validate_environment(self, context: AdapterContext) -> AdapterValidationReport:
        raise RuntimeError("synthetic environment failure")


def test_fake_implementation_checks_reject_identity_passthrough_and_constant_loss(
    tmp_path: Path,
) -> None:
    contract = _contract()
    context = _context(contract, tmp_path)

    identity = RuntimeAdapterFacade(_IdentityAdapter(), contract, context)
    with pytest.raises(RuntimeAdapterContractError, match="fake_identity_only_implementation"):
        identity.build()
    assert "fake_identity_only_implementation" in inspect_fake_implementation(
        _IdentityAdapter(), contract=contract
    )

    passthrough_findings = inspect_fake_implementation(
        _PassthroughAdapter(), contract=contract
    )
    assert "fake_passthrough_apply" in passthrough_findings

    constant_findings = inspect_fake_implementation(
        _ConstantLossAdapter(), contract=contract
    )
    assert "fake_constant_loss_or_forward" in constant_findings


class _MetadataOnly:
    adapter_version = "metadata.v1"
    source_commit = "metadata-commit"
    strategy = "callback"


def test_metadata_only_adapter_is_invalid(tmp_path: Path) -> None:
    contract = _contract()
    facade = RuntimeAdapterFacade(_MetadataOnly(), contract, _context(contract, tmp_path))

    report = facade.validate_contract()
    assert not report.ok
    assert any("adapter_callable_missing" in error for error in report.errors)


def test_intentional_passthrough_must_be_explicit(tmp_path: Path) -> None:
    contract = _contract()
    facade = RuntimeAdapterFacade(
        _IntentionalPassthroughAdapter(),
        contract,
        _context(contract, tmp_path),
    )

    preview = facade.apply({"model": "yolo26n.pt"}, {"imgsz": 640})
    assert preview.operations == []


def test_runtime_contract_converts_adapter_exceptions_to_failed_report(
    tmp_path: Path,
) -> None:
    contract = _contract()
    report = RuntimeAdapterFacade(
        _ExplodingAdapter(),
        contract,
        _context(contract, tmp_path),
    ).validate_contract()

    assert not report.ok
    assert "synthetic environment failure" in report.errors


def test_runtime_contract_signature_changes_with_runtime_binding() -> None:
    contract = _contract()
    changed = contract.model_copy(
        update={"runtime_hook": "compute_loss", "supported_yolo_versions": ["27"]}
    )

    assert contract.runtime_contract_signature != changed.runtime_contract_signature
