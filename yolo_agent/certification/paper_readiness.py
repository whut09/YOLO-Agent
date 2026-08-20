"""Offline, paper-scoped readiness preflight for the ASHA boundary.

This module deliberately stops before training.  It combines the paper
execution inventory with CPU adapter evidence and protocol prerequisites so a
paper can be admitted, recovered, or blocked without silently disappearing.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

from yolo_agent.certification.component_schemas import ComponentCertificationReport
from yolo_agent.certification.paper_adapter_discovery import (
    ReusableAdapterDescriptor,
    ReusablePaperAdapterDiscovery,
)
from yolo_agent.certification.paper_adapter_factory import (
    PaperAdapterCertificationFactory,
)
from yolo_agent.certification.paper_adapter_factory_schemas import (
    PaperAdapterCertificationReport,
)
from yolo_agent.core.yaml_io import YAMLModelMixin
from yolo_agent.research.paper_execution_schemas import (
    PaperExecutionDisposition,
    PaperExecutionInventory,
    PaperExecutionSpec,
)
from yolo_agent.research.paper_protocol_catalog import build_paper_protocol_contract


ReadinessStatus = str


class ReadinessCheck(BaseModel):
    """One named readiness result; never an accuracy measurement."""

    model_config = ConfigDict(extra="forbid")

    passed: bool
    status: ReadinessStatus = "passed"
    blocker: str | None = None
    evidence: list[str] = Field(default_factory=list)


class PaperReadinessRecord(BaseModel, YAMLModelMixin):
    """Complete readiness decision for exactly one paper."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "paper_readiness_record.v1"
    paper_id: str
    mechanism_id: str | None = None
    recipe_id: str | None = None
    adapter_hash: str = "missing"
    protocol_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    cache_key: str = Field(pattern=r"^[0-9a-f]{64}$")
    cache_hit: bool = False
    cpu_contract_result: ReadinessCheck
    shape_result: ReadinessCheck
    forward_result: ReadinessCheck
    backward_result: ReadinessCheck
    payload_result: ReadinessCheck
    dataset_evidence_result: ReadinessCheck
    teacher_evidence_result: ReadinessCheck
    graph_evidence_result: ReadinessCheck
    matched_control_readiness: ReadinessCheck
    asha_eligibility: bool = False
    final_disposition: PaperExecutionDisposition
    exact_blocker: str | None = None
    source_inventory_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @model_validator(mode="after")
    def validate_decision(self) -> "PaperReadinessRecord":
        checks = (
            self.cpu_contract_result,
            self.shape_result,
            self.forward_result,
            self.backward_result,
            self.payload_result,
            self.dataset_evidence_result,
            self.teacher_evidence_result,
            self.graph_evidence_result,
            self.matched_control_readiness,
        )
        if self.asha_eligibility:
            if self.final_disposition != "runtime_ready" or not all(
                item.passed for item in checks
            ):
                raise ValueError("ASHA-eligible paper must pass every readiness check")
            if self.exact_blocker is not None:
                raise ValueError("ASHA-eligible paper cannot retain a blocker")
        elif not self.exact_blocker:
            raise ValueError("non-ASHA paper readiness requires an exact blocker")
        return self


class PaperReadinessReport(BaseModel, YAMLModelMixin):
    """Persistent report containing every compatible paper, including failures."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "paper_readiness_report.v1"
    status: str
    inventory_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    paper_count: int = 0
    registry_hash: str
    model: str
    data: str
    imgsz: int = 640
    cpu_only: bool = True
    training_started: bool = False
    resource_policy: str = "cpu_only_no_gpu_probe"
    accuracy_claim: str = "none"
    gpu_probe: str = "not_run"
    records: list[PaperReadinessRecord] = Field(default_factory=list)
    disposition_counts: dict[str, int] = Field(default_factory=dict)
    cache_hits: int = 0
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    report_hash: str = ""

    @model_validator(mode="after")
    def validate_report(self) -> "PaperReadinessReport":
        paper_ids = [item.paper_id for item in self.records]
        if len(paper_ids) != len(set(paper_ids)):
            raise ValueError("paper readiness report contains duplicate paper IDs")
        if paper_ids != sorted(paper_ids):
            raise ValueError("paper readiness records must be sorted by paper_id")
        if self.paper_count != len(self.records):
            raise ValueError("paper_count must equal readiness record count")
        if self.disposition_counts != _counts(self.records):
            raise ValueError("paper readiness disposition counts do not match records")
        if self.cache_hits != sum(item.cache_hit for item in self.records):
            raise ValueError("paper readiness cache_hits does not match records")
        if self.report_hash and self.report_hash != self.calculate_hash():
            raise ValueError("paper readiness report hash mismatch")
        return self

    def calculate_hash(self) -> str:
        payload = self.model_dump(
            mode="json",
            exclude={"report_hash", "generated_at"},
        )
        for record in payload["records"]:
            record.pop("generated_at", None)
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    def with_hash(self) -> "PaperReadinessReport":
        return self.model_copy(update={"report_hash": self.calculate_hash()})


class PaperReadinessFactoryProtocol(Protocol):
    def run(self, **kwargs: object) -> PaperAdapterCertificationReport: ...


class PaperReadinessPreflight:
    """Evaluate all inventory rows with reusable CPU evidence."""

    def __init__(
        self,
        *,
        discovery: ReusablePaperAdapterDiscovery | None = None,
        certification_factory: PaperReadinessFactoryProtocol | None = None,
    ) -> None:
        self.discovery = discovery or ReusablePaperAdapterDiscovery()
        self.certification_factory = certification_factory or PaperAdapterCertificationFactory()

    def run(
        self,
        *,
        inventory: PaperExecutionInventory,
        registry_path: Path | str,
        model: str,
        data: Path | str,
        output_path: Path | str,
        certification_root: Path | str | None = None,
        run_cpu_certification: bool = True,
    ) -> PaperReadinessReport:
        output = Path(output_path).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        registry = Path(registry_path).resolve()
        data_path = Path(data).resolve()
        registry_hash = _file_hash(registry)
        descriptors_result = self.discovery.discover()
        descriptors = {item.component_id: item for item in descriptors_result.adapters}
        required_components = sorted(
            {
                component_id
                for record in inventory.records
                for component_id in record.canonical_component_ids
                if component_id not in {"inference.sahi_slicing"}
            }
        )
        certification = self._certify_components(
            required_components=required_components,
            descriptors=descriptors,
            registry=registry,
            model=model,
            data=data_path,
            root=Path(certification_root or output.parent / "adapter-certification"),
            run_cpu_certification=run_cpu_certification,
        )
        adapter_results = {
            item.component_id: item for item in certification.results
        } if certification is not None else {}
        cache_dir = output.parent / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        records = [
            self._evaluate_record(
                item,
                inventory=inventory,
                descriptors=descriptors,
                adapter_results=adapter_results,
                registry=registry,
                registry_hash=registry_hash,
                model=model,
                data=data_path,
                cache_dir=cache_dir,
            )
            for item in inventory.records
        ]
        records.sort(key=lambda item: item.paper_id)
        counts = _counts(records)
        report = PaperReadinessReport(
            status=("passed" if all(item.asha_eligibility for item in records) else "partial"),
            inventory_hash=inventory.inventory_hash,
            paper_count=len(inventory.records),
            registry_hash=registry_hash,
            model=model,
            data=str(data_path),
            records=records,
            disposition_counts=counts,
            cache_hits=sum(item.cache_hit for item in records),
        ).with_hash()
        report.to_yaml(output, exclude_none=True, sort_keys=False)
        return report

    @staticmethod
    def load_report(path: Path | str) -> PaperReadinessReport:
        """Load a report and verify its content hash before reuse."""
        return PaperReadinessReport.from_yaml(path)

    def _certify_components(
        self,
        *,
        required_components: list[str],
        descriptors: dict[str, ReusableAdapterDescriptor],
        registry: Path,
        model: str,
        data: Path,
        root: Path,
        run_cpu_certification: bool,
    ) -> PaperAdapterCertificationReport | None:
        selected = [item for item in required_components if item in descriptors]
        if not run_cpu_certification or not selected:
            return None
        return self.certification_factory.run(
            workdir=root,
            registry_path=registry,
            mode="cpu",
            model=model,
            data=str(data),
            device="cpu",
            execute_real_gpu=False,
            resume=True,
            changed_only=True,
            component_ids=selected,
        )

    def _evaluate_record(
        self,
        record: PaperExecutionSpec,
        *,
        inventory: PaperExecutionInventory,
        descriptors: dict[str, ReusableAdapterDescriptor],
        adapter_results: dict[str, Any],
        registry: Path,
        registry_hash: str,
        model: str,
        data: Path,
        cache_dir: Path,
    ) -> PaperReadinessRecord:
        component_ids = [
            item
            for item in record.canonical_component_ids
            if not item.startswith("inference.")
        ]
        adapters = [descriptors[item] for item in component_ids if item in descriptors]
        adapter_hash = (
            adapters[0].identity.adapter_hash
            if len(adapters) == 1
            else _combined_hash(
                {item.component_id: item.identity.adapter_hash for item in adapters}
            )
            if adapters
            else "missing"
        )
        protocol = _protocol(record)
        protocol_hash = (
            protocol.protocol_hash
            if protocol is not None
            else _paper_protocol_hash(record)
        )
        cache_key = _cache_key(
            record,
            adapter_hash=adapter_hash,
            protocol_hash=protocol_hash,
            registry_hash=registry_hash,
            model=model,
            data=data,
        )
        cache_path = cache_dir / f"{cache_key}.yaml"
        if cache_path.is_file():
            try:
                cached = PaperReadinessRecord.from_yaml(cache_path)
                if cached.cache_key == cache_key and cached.source_inventory_hash == inventory.inventory_hash:
                    return cached.model_copy(update={"cache_hit": True})
            except (OSError, TypeError, ValueError):
                pass
        results = [
            self._component_check(
                component_id,
                descriptors=descriptors,
                adapter_results=adapter_results,
            )
            for component_id in component_ids
        ]
        cpu_contract = _all(results, "cpu_contract")
        shape = _all(results, "shape")
        forward = _all(results, "forward")
        backward = _all(results, "backward")
        payload = _all(results, "payload")
        dataset = _dataset_check(record, data, protocol)
        teacher = _teacher_check(record)
        graph = _graph_check(record, protocol)
        matched = _matched_control_check(record, protocol)
        checks = [cpu_contract, shape, forward, backward, payload, dataset, teacher, graph, matched]
        # Surface protocol/data blockers before adapter blockers.  A missing
        # teacher or domain manifest is actionable even when its adapter is
        # also not ready.
        blocker = next(
            (
                item.blocker
                for item in (dataset, teacher, graph, matched, cpu_contract, shape, forward, backward, payload)
                if not item.passed and item.blocker
            ),
            None,
        )
        inference_only = _is_inference_only(record, protocol)
        if inference_only and blocker is None:
            blocker = "inference_only_not_training_candidate"
        asha = not inference_only and all(item.passed for item in checks)
        disposition = _final_disposition(record, blocker, asha, inference_only)
        result = PaperReadinessRecord(
            paper_id=record.paper_id,
            mechanism_id=(record.paper_specific_mechanism_ids[0] if record.paper_specific_mechanism_ids else None),
            recipe_id=(record.recipe_ids[0] if record.recipe_ids else None),
            adapter_hash=adapter_hash,
            protocol_hash=protocol_hash,
            cache_key=cache_key,
            cpu_contract_result=cpu_contract,
            shape_result=shape,
            forward_result=forward,
            backward_result=backward,
            payload_result=payload,
            dataset_evidence_result=dataset,
            teacher_evidence_result=teacher,
            graph_evidence_result=graph,
            matched_control_readiness=matched,
            asha_eligibility=asha,
            final_disposition=disposition,
            exact_blocker=blocker,
            source_inventory_hash=inventory.inventory_hash,
        )
        result.to_yaml(cache_path, exclude_none=True, sort_keys=False)
        return result

    @staticmethod
    def _component_check(
        component_id: str,
        *,
        descriptors: dict[str, ReusableAdapterDescriptor],
        adapter_results: dict[str, Any],
    ) -> dict[str, ReadinessCheck]:
        descriptor = descriptors.get(component_id)
        if descriptor is None:
            missing = ReadinessCheck(passed=False, status="blocked", blocker=f"adapter_missing:{component_id}")
            return {name: missing for name in ("cpu_contract", "shape", "forward", "backward", "payload")}
        result = adapter_results.get(component_id)
        report = _load_cpu_report(result)
        if result is None or report is None:
            missing = ReadinessCheck(passed=False, status="blocked", blocker=f"cpu_certification_missing:{component_id}")
            return {name: missing for name in ("cpu_contract", "shape", "forward", "backward", "payload")}
        passed = result.status in {"passed", "skipped_resume", "skipped_unchanged"}
        stages = {item.stage_id: item for item in (report.stages if report else [])}
        def stage_check(name: str, aliases: tuple[str, ...]) -> ReadinessCheck:
            stage = next((stages[item] for item in aliases if item in stages), None)
            ok = passed and (stage is None or stage.status == "passed")
            blocker = None if ok else f"cpu_{name}_failed:{component_id}"
            return ReadinessCheck(passed=ok, status="passed" if ok else "blocked", blocker=blocker)
        return {
            "cpu_contract": ReadinessCheck(passed=passed, status="passed" if passed else "blocked", blocker=None if passed else f"cpu_contract_failed:{component_id}"),
            "shape": stage_check("shape", ("isolated_smoke", "adapter_import")),
            "forward": stage_check("forward", ("isolated_smoke", "hook_signature")),
            "backward": stage_check("backward", ("isolated_smoke", "unit_tests")),
            "payload": stage_check("payload", ("runtime_payload",)),
        }


def _load_cpu_report(result: Any) -> ComponentCertificationReport | None:
    path = getattr(result, "cpu_report", None)
    if path is None or not Path(path).is_file():
        return None
    try:
        return ComponentCertificationReport.from_yaml(path)
    except (OSError, TypeError, ValueError):
        return None


def _all(results: list[dict[str, ReadinessCheck]], name: str) -> ReadinessCheck:
    if not results:
        return ReadinessCheck(passed=True, status="not_applicable")
    return next((item[name] for item in results if not item[name].passed), results[0][name])


def _protocol(record: PaperExecutionSpec) -> Any | None:
    try:
        return build_paper_protocol_contract(record.paper_id, record.canonical_component_ids)
    except (KeyError, TypeError, ValueError):
        return None


def _dataset_check(record: PaperExecutionSpec, data: Path, protocol: Any | None) -> ReadinessCheck:
    required = set(record.required_evidence)
    if not data.is_file():
        return ReadinessCheck(passed=False, status="evidence_recovery", blocker="dataset_manifest_missing")
    if (protocol is not None and protocol.is_domain_adaptation) or any(
        item.startswith("domain_adaptation.") for item in record.canonical_component_ids
    ):
        return ReadinessCheck(passed=False, status="evidence_recovery", blocker="target_domain_dataset_missing")
    if any("hard_negative" in item for item in required):
        return ReadinessCheck(passed=False, status="evidence_recovery", blocker="hard_negative_manifest_missing")
    return ReadinessCheck(passed=True, evidence=[str(data)])


def _teacher_check(record: PaperExecutionSpec) -> ReadinessCheck:
    required = set(record.required_evidence) | set(record.required_checkpoints)
    if not any("teacher" in item for item in required):
        return ReadinessCheck(passed=True, status="not_applicable")
    existing = [Path(item) for item in record.required_checkpoints if Path(item).is_file()]
    if existing:
        return ReadinessCheck(passed=True, evidence=[str(item) for item in existing])
    return ReadinessCheck(passed=False, status="evidence_recovery", blocker="teacher_checkpoint_missing")


def _graph_check(record: PaperExecutionSpec, protocol: Any | None) -> ReadinessCheck:
    if protocol is None or not protocol.is_model_graph:
        return ReadinessCheck(passed=True, status="not_applicable")
    if not protocol.graph_identity:
        return ReadinessCheck(passed=False, status="blocked", blocker="graph_identity_missing")
    if not protocol.yolo26_one_to_one_head or not protocol.native_dfl_free_regression:
        return ReadinessCheck(passed=False, status="blocked", blocker="yolo26_graph_contract_missing")
    return ReadinessCheck(passed=True, evidence=[protocol.graph_identity, "imgsz=640"])


def _matched_control_check(record: PaperExecutionSpec, protocol: Any | None) -> ReadinessCheck:
    if protocol is None or not protocol.paired_baseline_requirement:
        return ReadinessCheck(passed=False, status="blocked", blocker="matched_control_protocol_missing")
    return ReadinessCheck(passed=True, evidence=["matched_control_required", "protocol_bound"])


def _is_inference_only(record: PaperExecutionSpec, protocol: Any | None) -> bool:
    return bool(protocol is not None and protocol.is_inference_only) or any(
        item.startswith("inference.") for item in record.canonical_component_ids
    )


def _final_disposition(
    record: PaperExecutionSpec,
    blocker: str | None,
    asha: bool,
    inference_only: bool,
) -> PaperExecutionDisposition:
    if asha:
        return "runtime_ready"
    if inference_only:
        return "incompatible"
    if blocker and any(
        token in blocker
        for token in ("dataset", "teacher", "hard_negative", "target_domain", "evidence")
    ):
        return "evidence_recovery"
    if blocker:
        return "blocked_runtime"
    return record.current_disposition if record.current_disposition != "runtime_ready" else "implementation_request"


def _paper_protocol_hash(record: PaperExecutionSpec) -> str:
    return _stable_hash({"paper_id": record.paper_id, "mechanisms": record.canonical_component_ids, "required_protocol": record.required_dataset_protocol})


def _cache_key(record: PaperExecutionSpec, *, adapter_hash: str, protocol_hash: str, registry_hash: str, model: str, data: Path) -> str:
    return _stable_hash({"execution_fingerprint": record.execution_fingerprint, "adapter_hash": adapter_hash, "protocol_hash": protocol_hash, "registry_hash": registry_hash, "model": model, "data": str(data), "data_hash": _file_hash(data), "imgsz": 640})


def _combined_hash(values: dict[str, str]) -> str:
    return _stable_hash(values)


def _stable_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def _file_hash(path: Path) -> str:
    if not path.is_file():
        return "missing"
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _counts(records: list[PaperReadinessRecord]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        counts[record.final_disposition] = counts.get(record.final_disposition, 0) + 1
    return dict(sorted(counts.items()))


__all__ = ["PaperReadinessPreflight", "PaperReadinessRecord", "PaperReadinessReport", "ReadinessCheck"]
