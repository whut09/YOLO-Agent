"""Discovery of reusable runtime adapters eligible for batch certification."""

from __future__ import annotations

import importlib
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from yolo_agent.certification.component_runner import (
    component_certification_protocol_hash,
)
from yolo_agent.certification.paper_adapter_factory_schemas import (
    AdapterCertificationIdentity,
)
from yolo_agent.components.adapters.base import ComponentAdapter
from yolo_agent.components.adapters.dummy import DummyAdapter
from yolo_agent.components.contracts import ComponentContract, load_contracts
from yolo_agent.components.maturity_registry import (
    adapter_source_hash,
    current_code_commit,
    installed_ultralytics_version,
)
from yolo_agent.resources import ResourcePaths


class ReusableAdapterDescriptor(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    component_id: str
    contract: ComponentContract
    contract_path: Path
    adapter_qualified_name: str
    identity: AdapterCertificationIdentity


class ReusableAdapterDiscoveryResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    adapters: list[ReusableAdapterDescriptor]
    errors: dict[str, str]


class ReusablePaperAdapterDiscovery:
    """Find concrete reusable adapters from typed component contracts."""

    def __init__(self, contract_paths: list[Path | str] | None = None) -> None:
        self.contract_paths = (
            [Path(item) for item in contract_paths]
            if contract_paths is not None
            else [
                ResourcePaths.COMPONENT_COMPATIBILITY,
                *sorted(ResourcePaths.COMPONENTS_DIR.rglob("*.yaml")),
            ]
        )

    def discover(self) -> ReusableAdapterDiscoveryResult:
        contracts: dict[str, tuple[ComponentContract, Path]] = {}
        errors: dict[str, str] = {}
        for path in self.contract_paths:
            if not path.is_file():
                continue
            try:
                loaded = load_contracts(path)
            except (KeyError, TypeError, ValueError):
                continue
            for contract in loaded:
                contracts[contract.component_id] = (contract, path.resolve())

        code_commit = current_code_commit()
        ultralytics_version = installed_ultralytics_version()
        descriptors: list[ReusableAdapterDescriptor] = []
        for component_id, (contract, path) in sorted(contracts.items()):
            if not contract.implementation_path or not contract.adapter_class:
                continue
            qualified = f"{contract.implementation_path}:{contract.adapter_class}"
            try:
                module = importlib.import_module(contract.implementation_path)
                adapter_type = getattr(module, contract.adapter_class)
                if not isinstance(adapter_type, type) or not issubclass(
                    adapter_type, ComponentAdapter
                ):
                    raise TypeError("configured adapter is not a ComponentAdapter")
                if adapter_type is DummyAdapter:
                    continue
                adapter_hash = adapter_source_hash(contract, adapter=adapter_type)
            except (AttributeError, ImportError, OSError, TypeError, ValueError) as exc:
                errors[component_id] = f"{qualified}: {exc}"
                continue
            protocol_hash = component_certification_protocol_hash(
                component_id=component_id,
                adapter_hash=adapter_hash,
                ultralytics_version=ultralytics_version,
            )
            descriptors.append(
                ReusableAdapterDescriptor(
                    component_id=component_id,
                    contract=contract,
                    contract_path=path,
                    adapter_qualified_name=qualified,
                    identity=AdapterCertificationIdentity(
                        component_id=component_id,
                        adapter_hash=adapter_hash,
                        code_commit=code_commit,
                        ultralytics_version=ultralytics_version,
                        protocol_hash=protocol_hash,
                    ),
                )
            )
        return ReusableAdapterDiscoveryResult(adapters=descriptors, errors=errors)


__all__ = [
    "ReusableAdapterDescriptor",
    "ReusableAdapterDiscoveryResult",
    "ReusablePaperAdapterDiscovery",
]
