"""Fail-closed CPU certification for the thirteen independent routes.

Each independent route must carry a complete ``ComponentContract``
identity: implementation path, adapter class, changed variable, runtime
hook, runtime payload schema, evidence artifact, adapter source hash,
protocol (payload) hash, fixed ``imgsz=640``, YOLO26 one-to-one head
compatibility, native DFL-free regression compatibility, and a matched
baseline requirement.  A route missing any single field is blocked and
can never report runtime readiness.  Certification is CPU-only and
GPU-free: it probes contract, shape, forward, and backward, but a
certified route is still not a runtime-ready ASHA candidate.
"""

from __future__ import annotations

import hashlib
import importlib
import inspect
import json
import tempfile
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from yolo_agent.components.adapters import (
    AdapterContext,
    ComponentAdapterRegistry,
)
from yolo_agent.components.adapters.audit_contract import (
    EXPECTED_RUNTIME_ADAPTERS,
    validate_audited_runtime_payload,
)
from yolo_agent.components.independent_component_router import (
    COMPONENT_CATALOG,
    INDEPENDENT_COMPONENT_IDS,
    IndependentComponentId,
)
from yolo_agent.core.yaml_io import YAMLModelMixin
from yolo_agent.research.component_aliases import ComponentAliasResolver


IndependentRouteDisposition = Literal[
    "certified_route",
    "blocked_missing_field",
    "probe_failed",
]


REQUIRED_ROUTE_CHECKS: tuple[str, ...] = (
    "contract_present",
    "implementation_path",
    "adapter_class",
    "changed_variable",
    "runtime_hook",
    "payload_schema",
    "evidence_artifact",
    "adapter_hash",
    "protocol_hash",
    "fixed_imgsz_640",
    "yolo26_one_to_one_head",
    "native_dfl_free_regression",
    "matched_baseline",
    "cpu_smoke",
)


class IndependentComponentRouteReport(BaseModel, YAMLModelMixin):
    """CPU certification result for one independent component route."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "independent_component_route_report.v1"
    component_id: IndependentComponentId
    recipe_id: str = ""
    graph_identity: str = ""
    implementation_path: str | None = None
    adapter_class: str | None = None
    changed_variable: str | None = None
    runtime_hook: str | None = None
    runtime_payload_field: str | None = None
    evidence_artifact: str | None = None
    inference_only: bool = False
    requires_shadow_evidence: bool = False
    paired_baseline_required: bool = True
    fixed_imgsz: int = 640
    checks: dict[str, bool] = Field(default_factory=dict)
    cpu_smoke_checks: dict[str, object] = Field(default_factory=dict)
    adapter_source_sha256: str | None = None
    protocol_hash: str | None = None
    disposition: IndependentRouteDisposition
    reason_codes: list[str] = Field(default_factory=list)
    runtime_ready: bool = False
    report_hash: str = ""

    @model_validator(mode="after")
    def bind_report_hash(self) -> "IndependentComponentRouteReport":
        if self.runtime_ready:
            raise ValueError(
                "independent route certification is CPU-only and can never be runtime ready"
            )
        expected = compute_independent_route_report_hash(self)
        if self.report_hash and self.report_hash != expected:
            raise ValueError("independent route report hash mismatch")
        self.report_hash = expected
        return self


def compute_independent_route_report_hash(
    report: IndependentComponentRouteReport,
) -> str:
    payload = report.model_dump(
        mode="json", exclude={"report_hash", "schema_version"}
    )
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def verify_independent_route_report_hash(report: IndependentComponentRouteReport) -> bool:
    return bool(report.report_hash) and (
        report.report_hash == compute_independent_route_report_hash(report)
    )


class IndependentComponentRouteCertificationSummary(
    BaseModel, YAMLModelMixin
):
    """Persistent coverage summary for one independent-route certification run."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "independent_component_route_certification.v1"
    components_total: int
    certified_routes: int
    blocked_missing_field: int
    probe_failed: int
    inference_only_components: list[str] = Field(default_factory=list)
    shadow_evidence_components: list[str] = Field(default_factory=list)
    silent_drops: list[str] = Field(default_factory=list)
    reports: list[IndependentComponentRouteReport]
    summary_hash: str = ""

    @model_validator(mode="after")
    def bind_summary(self) -> "IndependentComponentRouteCertificationSummary":
        if self.silent_drops:
            raise ValueError(
                f"independent route certification silent drops: {self.silent_drops}"
            )
        component_ids = [item.component_id for item in self.reports]
        if len(component_ids) != len(set(component_ids)):
            raise ValueError(
                "independent route certification contains duplicate components"
            )
        if self.components_total != len(component_ids):
            raise ValueError(
                "every requested independent component must carry exactly one report"
            )
        if any(item.runtime_ready for item in self.reports):
            raise ValueError(
                "independent route certification can never be runtime ready"
            )
        counts = (self.certified_routes, self.blocked_missing_field, self.probe_failed)
        if sum(counts) != self.components_total:
            raise ValueError("independent route dispositions must cover every component")
        expected = compute_independent_certification_summary_hash(self)
        if self.summary_hash and self.summary_hash != expected:
            raise ValueError("independent route certification summary hash mismatch")
        self.summary_hash = expected
        return self


def compute_independent_certification_summary_hash(
    summary: IndependentComponentRouteCertificationSummary,
) -> str:
    payload = summary.model_dump(
        mode="json", exclude={"summary_hash", "schema_version"}
    )
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _disposition_from_reasons(
    reasons: list[str], probe_failed: bool
) -> IndependentRouteDisposition:
    if probe_failed:
        return "probe_failed"
    if reasons:
        return "blocked_missing_field"
    return "certified_route"


def certify_independent_component_route(
    component_id: IndependentComponentId,
    *,
    workspace: Path | str | None = None,
    protocol_hash: str = "independent-component-cpu-certification",
    matched_baseline: bool = True,
    imgsz: int = 640,
) -> IndependentComponentRouteReport:
    """Certify one independent route without granting runtime readiness."""
    catalog = COMPONENT_CATALOG[component_id]
    expectation = EXPECTED_RUNTIME_ADAPTERS.get(component_id)
    checks: dict[str, bool] = {name: False for name in REQUIRED_ROUTE_CHECKS}
    reasons: list[str] = []
    fields: dict[str, Any] = {
        "recipe_id": str(catalog.get("recipe_id", "")),
        "graph_identity": str(catalog.get("graph_identity", component_id)),
        "runtime_hook": str(catalog.get("runtime_hook", "")),
        "runtime_payload_field": str(catalog.get("runtime_payload_field", "")),
        "evidence_artifact": str(catalog.get("evidence_artifact", "")),
        "inference_only": component_id == "inference.sahi_slicing",
        "requires_shadow_evidence": bool(catalog.get("requires_shadow_evidence")),
        "fixed_imgsz": imgsz,
    }
    checks["changed_variable"] = bool(str(catalog.get("changed_variable", "")))
    checks["matched_baseline"] = bool(matched_baseline)
    probe_failed = False
    contract = ComponentAliasResolver.from_yaml().contracts.get(component_id)
    if contract is None or contract.component_id != component_id:
        reasons.append("missing_field:contract")
    else:
        checks["contract_present"] = True
        fields["implementation_path"] = str(contract.implementation_path or "")
        fields["adapter_class"] = str(contract.adapter_class or "")
        fields["changed_variable"] = str(contract.changed_variable or "")
        checks["implementation_path"] = bool(contract.implementation_path)
        checks["adapter_class"] = bool(contract.adapter_class)
        checks["payload_schema"] = bool(contract.runtime_payload_schema)
        checks["evidence_artifact"] = bool(contract.evidence_protocol) and bool(
            fields["evidence_artifact"]
        )
        checks["yolo26_one_to_one_head"] = (
            "yolo26" in contract.supported_detector_families
            and "one_to_one" in contract.supported_heads
        )
        constraints = dict(contract.tensor_input_contract or {})
        dfl_flags = dict(constraints.get("compatibility_constraints") or {})
        checks["native_dfl_free_regression"] = dfl_flags.get("requires_dfl") is not True
        checks["fixed_imgsz_640"] = (
            bool(contract.fixed_imgsz_compatible) and imgsz == 640
        )
        try:
            module = importlib.import_module(str(contract.implementation_path))
            adapter_type = getattr(module, str(contract.adapter_class))
            if not inspect.isclass(adapter_type):
                raise TypeError("adapter_class is not a class")
            if str(contract.implementation_path) != str(
                catalog.get("implementation_path", "")
            ):
                reasons.append("implementation_path_mismatch")
            if str(contract.adapter_class) != str(catalog.get("adapter_class", "")):
                reasons.append("adapter_class_mismatch")
            source_path = Path(inspect.getfile(adapter_type)).resolve()
            fields["adapter_source_sha256"] = _sha256_file(source_path)
            checks["adapter_hash"] = len(
                str(fields["adapter_source_sha256"] or "")
            ) == 64
            with tempfile.TemporaryDirectory(
                prefix="yolo26-independent-cert-",
                dir=workspace,
            ) as temp_dir:
                context = AdapterContext(
                    contract=contract,
                    detector_family="yolo26",
                    head="one_to_one",
                    imgsz=imgsz,
                    workspace=Path(temp_dir),
                )
                adapter = ComponentAdapterRegistry().create_for_contract(contract)
                smoke = adapter.smoke_test(context)
                smoke_checks = dict(getattr(smoke, "checks", {}) or {})
                fields["cpu_smoke_checks"] = smoke_checks
                checks["cpu_smoke"] = bool(smoke.passed) and bool(smoke_checks)
                payload = adapter.build_runtime_payload(
                    context,
                    protocol_hash=protocol_hash,
                    base_command=[
                        "python",
                        "-m",
                        "yolo_agent.adapters.ultralytics.runtime_entrypoint",
                    ],
                    generated_config={"imgsz": imgsz},
                )
                if payload is None:
                    reasons.append("missing_field:runtime_payload")
                else:
                    payload.verify_imports()
                    validate_audited_runtime_payload(payload, component_id)
                    fields["protocol_hash"] = payload.payload_hash
                    checks["protocol_hash"] = len(
                        str(payload.payload_hash or "")
                    ) == 64
                    if (
                        contract.changed_variable
                        and str(catalog.get("changed_variable", ""))
                        and contract.changed_variable
                        != str(catalog["changed_variable"])
                    ):
                        reasons.append("changed_variable_contract_mismatch")
                    if expectation is not None:
                        references = getattr(payload, expectation.plugin_kind)
                        checks["runtime_hook"] = bool(
                            references
                            and expectation.required_hook in references[0].required_hooks
                        )
        except (
            AttributeError,
            ImportError,
            ModuleNotFoundError,
            OSError,
            TypeError,
            ValueError,
        ) as exc:
            probe_failed = True
            reasons.append(f"probe_failed:{type(exc).__name__}")
    for name in REQUIRED_ROUTE_CHECKS:
        if not checks[name] and f"missing_field:{name}" not in reasons:
            reasons.append(f"missing_field:{name}")
    return IndependentComponentRouteReport(
        component_id=component_id,
        **fields,
        checks=checks,
        disposition=_disposition_from_reasons(reasons, probe_failed),
        reason_codes=list(dict.fromkeys(reasons)),
    )


def certify_independent_component_routes(
    *,
    workspace: Path | str | None = None,
    component_ids: tuple[IndependentComponentId, ...] | None = None,
    protocol_hash: str = "independent-component-cpu-certification",
    matched_baseline: bool = True,
    imgsz: int = 640,
) -> list[IndependentComponentRouteReport]:
    """Certify every requested independent route without silent drops."""
    if component_ids is None:
        ids = INDEPENDENT_COMPONENT_IDS
    else:
        ids = tuple(component_ids)
    return [
        certify_independent_component_route(
            component_id,
            workspace=workspace,
            protocol_hash=protocol_hash,
            matched_baseline=matched_baseline,
            imgsz=imgsz,
        )
        for component_id in ids
    ]


def certify_all_independent_component_routes(
    *,
    output_path: Path | str,
    workspace: Path | str | None = None,
    component_ids: tuple[IndependentComponentId, ...] | None = None,
    protocol_hash: str = "independent-component-cpu-certification",
    matched_baseline: bool = True,
    imgsz: int = 640,
) -> IndependentComponentRouteCertificationSummary:
    """Certify every requested independent route and persist one summary."""
    reports = certify_independent_component_routes(
        workspace=workspace,
        component_ids=component_ids,
        protocol_hash=protocol_hash,
        matched_baseline=matched_baseline,
        imgsz=imgsz,
    )
    requested = (
        INDEPENDENT_COMPONENT_IDS if component_ids is None else tuple(component_ids)
    )
    found = {item.component_id for item in reports}
    summary = IndependentComponentRouteCertificationSummary(
        components_total=len(requested),
        certified_routes=sum(
            item.disposition == "certified_route" for item in reports
        ),
        blocked_missing_field=sum(
            item.disposition == "blocked_missing_field" for item in reports
        ),
        probe_failed=sum(item.disposition == "probe_failed" for item in reports),
        inference_only_components=[
            item.component_id for item in reports if item.inference_only
        ],
        shadow_evidence_components=[
            item.component_id for item in reports if item.requires_shadow_evidence
        ],
        silent_drops=[item for item in requested if item not in found],
        reports=reports,
    )
    summary.to_yaml(output_path, sort_keys=False)
    return summary


__all__ = [
    "IndependentComponentRouteCertificationSummary",
    "IndependentComponentRouteReport",
    "IndependentRouteDisposition",
    "REQUIRED_ROUTE_CHECKS",
    "certify_all_independent_component_routes",
    "certify_independent_component_route",
    "certify_independent_component_routes",
    "compute_independent_certification_summary_hash",
    "compute_independent_route_report_hash",
    "verify_independent_route_report_hash",
]
