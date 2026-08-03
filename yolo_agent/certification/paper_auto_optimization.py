"""Opt-in entrypoint for multi-mechanism paper optimization acceptance."""

from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
from typing import Iterator, Protocol
from uuid import uuid4

from yolo_agent.certification.paper_auto_optimization_multi import (
    run_multi_mechanism_acceptance,
)
from yolo_agent.certification.paper_auto_optimization_research import (
    PaperAcceptanceResearchContext,
    PaperAcceptanceResearchPreparer,
)
from yolo_agent.certification.paper_auto_optimization_schemas import (
    PaperAutoOptimizationReport,
)
from yolo_agent.certification.runner import GpuAcceptanceBackend, UltralyticsGpuBackend


class PaperResearchPreparerProtocol(Protocol):
    def prepare(self, output_path: Path | str) -> PaperAcceptanceResearchContext: ...


class PaperAutoOptimizationAcceptanceSuite:
    """Certify four paper mechanism families without granting full-run consent."""

    report_name = "paper_auto_optimization_report.yaml"

    def __init__(
        self,
        backend: GpuAcceptanceBackend | None = None,
        research_preparer: PaperResearchPreparerProtocol | None = None,
    ) -> None:
        self.backend = backend or UltralyticsGpuBackend()
        self.research_preparer = research_preparer

    def run(
        self,
        *,
        workdir: Path | str,
        research_root: Path | str = "research",
        source: Path | str | None = None,
        maturity_registry: Path | str = "runs/component_maturity_registry.yaml",
        policy_memory_root: Path | str = "runs",
        model: str = "yolo26n.pt",
        device: str = "0",
        source_commit: str | None = None,
        execute_real_gpu: bool = False,
    ) -> PaperAutoOptimizationReport:
        """Run a matched pilot_3 cohort and ASHA-owned pilot_10 survivors."""
        root = Path(workdir).resolve()
        root.mkdir(parents=True, exist_ok=True)
        with _acceptance_workdir_lock(root):
            preparer = None
            if execute_real_gpu:
                preparer = self.research_preparer or PaperAcceptanceResearchPreparer(
                    research_root=research_root,
                    source=source,
                    maturity_registry=maturity_registry,
                    source_commit=source_commit,
                )
            report = run_multi_mechanism_acceptance(
                root=root,
                backend=self.backend,
                research_preparer=preparer,
                maturity_registry=maturity_registry,
                policy_memory_root=policy_memory_root,
                model=model,
                device=device,
                execute_real_gpu=execute_real_gpu,
            )
            return self._write_report(root, report)

    @classmethod
    def _write_report(
        cls,
        root: Path,
        report: PaperAutoOptimizationReport,
    ) -> PaperAutoOptimizationReport:
        path = root / cls.report_name
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        report.to_yaml(temporary, exclude_none=True, sort_keys=False)
        temporary.replace(path)
        return report


@contextmanager
def _acceptance_workdir_lock(root: Path) -> Iterator[None]:
    """Prevent concurrent acceptance processes from contaminating one workdir."""
    lock_path = root / ".paper_auto_optimization.lock"
    token = uuid4().hex
    payload = json.dumps({"pid": os.getpid(), "token": token})
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        owner = lock_path.read_text(encoding="utf-8-sig", errors="replace")
        raise RuntimeError(
            "paper auto-optimization workdir is already active; "
            f"choose a fresh --workdir or wait for its owner: {owner}"
        ) from exc
    try:
        os.write(descriptor, payload.encode("utf-8"))
        os.close(descriptor)
        yield
    finally:
        try:
            current = json.loads(lock_path.read_text(encoding="utf-8-sig"))
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            current = {}
        if current.get("token") == token:
            lock_path.unlink(missing_ok=True)


__all__ = [
    "PaperAutoOptimizationAcceptanceSuite",
    "PaperResearchPreparerProtocol",
]
