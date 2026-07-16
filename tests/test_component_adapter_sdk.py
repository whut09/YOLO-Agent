from pathlib import Path

import pytest

from yolo_agent.components.adapters import AdapterContext, ComponentAdapter, ComponentAdapterRegistry, DummyAdapter
from yolo_agent.components.adapters.base import ExpectedArtifact, RollbackPlan, SmokeTestResult
from yolo_agent.components.contracts import ComponentContract


def _context(tmp_path: Path) -> AdapterContext:
    contract = ComponentContract(
        component_id="dummy.component", display_name="Dummy", category="augmentation",
        implementation_path="tests", adapter_class="DummyAdapter", maturity="smoke_passed",
        fixed_imgsz_compatible=True,
    )
    return AdapterContext(contract=contract, workspace=tmp_path, imgsz=640)


def test_dummy_adapter_is_dry_run_and_idempotent(tmp_path: Path) -> None:
    adapter = DummyAdapter()
    context = _context(tmp_path)
    model, training = {"depth": 1}, {"epochs": 1}
    first = adapter.prepare_patch(model, training, context, dry_run=True)
    second = adapter.prepare_patch(model, training, context, dry_run=True)
    assert model == {"depth": 1} and training == {"epochs": 1}
    assert first.patched_training_config["adapter_marker"] == "dummy.component"
    assert first.operations == second.operations
    assert first.idempotency_key == second.idempotency_key
    assert first.dry_run is True and not first.rollback.restores_global_source


def test_dummy_adapter_declares_changed_fields(tmp_path: Path) -> None:
    preview = DummyAdapter().prepare_patch({}, {}, _context(tmp_path))
    assert preview.declared_modified_fields == ["training_config.adapter_marker"]
    assert preview.operations[0].field == "adapter_marker"


def test_adapter_compatibility_failure_is_explicit(tmp_path: Path) -> None:
    context = _context(tmp_path).model_copy(update={"imgsz": 1280})
    with pytest.raises(ValueError, match="imgsz=640"):
        DummyAdapter().prepare_patch({}, {}, context)


def test_registry_requires_component_adapter_subclass() -> None:
    registry = ComponentAdapterRegistry()
    registry.register("dummy.component", DummyAdapter)
    assert registry.ids() == ["dummy.component"]
    assert isinstance(registry.create("dummy.component"), DummyAdapter)
    with pytest.raises(TypeError):
        registry.register("bad", object)  # type: ignore[arg-type]
    with pytest.raises(KeyError):
        registry.create("missing")


def test_adapter_contract_is_abstract() -> None:
    assert ComponentAdapter.__abstractmethods__ == {
        "validate_environment", "validate_compatibility", "patch_model_config",
        "patch_training_config", "build_module", "load_pretrained_weights",
        "smoke_test", "expected_artifacts", "rollback_plan",
    }


def test_undeclared_patch_is_rejected(tmp_path: Path) -> None:
    class BadAdapter(DummyAdapter):
        def patch_training_config(self, config, context, *, dry_run=True):
            config["undeclared"] = True
            return config

    with pytest.raises(ValueError, match="undeclared"):
        BadAdapter().prepare_patch({}, {}, _context(tmp_path))


def test_rollback_cannot_escape_workspace(tmp_path: Path) -> None:
    class BadRollbackAdapter(DummyAdapter):
        def rollback_plan(self, context):
            return RollbackPlan(files_to_remove=[Path("..") / "outside.yaml"])

    with pytest.raises(ValueError, match="escapes"):
        BadRollbackAdapter().prepare_patch({}, {}, _context(tmp_path))


def test_expected_artifacts_and_smoke_result_are_structured(tmp_path: Path) -> None:
    adapter, context = DummyAdapter(), _context(tmp_path)
    assert adapter.smoke_test(context) == SmokeTestResult(passed=True, checks={"local_only": True})
    assert adapter.expected_artifacts(context) == [ExpectedArtifact(name="adapter_patch", relative_path=Path("adapter_patch.yaml"))]
    assert adapter.load_pretrained_weights({}, None, context).loaded is False
