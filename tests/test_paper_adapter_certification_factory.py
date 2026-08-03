from pathlib import Path

from yolo_agent.certification.component_schemas import (
    ComponentCertificationReport,
    ComponentCertificationStage,
)
from yolo_agent.certification.paper_adapter_discovery import (
    ReusableAdapterDescriptor,
    ReusableAdapterDiscoveryResult,
)
from yolo_agent.certification.paper_adapter_factory import (
    PaperAdapterCertificationFactory,
)
from yolo_agent.certification.paper_adapter_factory_schemas import (
    AdapterCertificationIdentity,
)
from yolo_agent.components.contracts import ComponentContract


def _descriptor(component_id: str) -> ReusableAdapterDescriptor:
    return ReusableAdapterDescriptor(
        component_id=component_id,
        contract=ComponentContract(
            component_id=component_id,
            display_name=component_id,
            category="sampling",
            implementation_path="example.adapters",
            adapter_class="ExampleAdapter",
            maturity="adapter_implemented",
        ),
        contract_path=Path("components.yaml"),
        adapter_qualified_name="example.adapters:ExampleAdapter",
        identity=AdapterCertificationIdentity(
            component_id=component_id,
            adapter_hash=("a" if component_id.endswith("a") else "b") * 64,
            code_commit="commit-one",
            ultralytics_version="8.4.0",
            protocol_hash=f"protocol-{component_id}",
        ),
    )


class _Discovery:
    def discover(self) -> ReusableAdapterDiscoveryResult:
        return ReusableAdapterDiscoveryResult(
            adapters=[_descriptor("component.a"), _descriptor("component.b")],
            errors={},
        )


class _Runner:
    def __init__(self, *, fail_component: str | None = None) -> None:
        self.fail_component = fail_component
        self.calls: list[dict[str, object]] = []

    def run(self, **kwargs: object) -> ComponentCertificationReport:
        self.calls.append(kwargs)
        component_id = str(kwargs["component_id"])
        mode = str(kwargs["mode"])
        failed = component_id == self.fail_component and mode == "cpu"
        workdir = Path(str(kwargs["workdir"]))
        workdir.mkdir(parents=True, exist_ok=True)
        stages = (
            [
                ComponentCertificationStage(stage_id=stage, status="passed")
                for stage in (
                    "adapter_import",
                    "runtime_payload",
                    "hook_signature",
                    "unit_tests",
                    "isolated_smoke",
                )
            ]
            if mode == "cpu" and not failed
            else [
                ComponentCertificationStage(
                    stage_id="cpu_smoke_precondition", status="passed"
                ),
                ComponentCertificationStage(
                    stage_id="isolated_gpu_smoke", status="passed"
                ),
            ]
            if mode == "gpu"
            else []
        )
        report = ComponentCertificationReport(
            component_id=component_id,
            mode=mode,
            status="failed" if failed else "passed",
            initial_maturity=(
                "smoke_passed" if mode == "gpu" else "adapter_implemented"
            ),
            final_maturity=(
                "gpu_certified"
                if mode == "gpu"
                else "adapter_implemented"
                if failed
                else "smoke_passed"
            ),
            next_maturity="pilot_reproduced" if mode == "gpu" else "gpu_certified",
            protocol_hash=str(kwargs["protocol_hash"]),
            adapter_hash="a" * 64,
            code_commit="commit-one",
            ultralytics_version="8.4.0",
            registry_path=Path(str(kwargs["registry_path"])),
            workdir=workdir,
            stages=stages,
            missing_artifacts=["smoke_passed"] if failed else [],
            errors=["synthetic CPU failure"] if failed else [],
        )
        report.to_yaml(
            workdir / f"component_certification.{mode}.yaml",
            exclude_none=True,
            sort_keys=False,
        )
        return report


def test_cpu_factory_continues_after_independent_adapter_failure(tmp_path: Path) -> None:
    runner = _Runner(fail_component="component.a")
    report = PaperAdapterCertificationFactory(
        discovery=_Discovery(), runner=runner
    ).run(
        workdir=tmp_path / "batch",
        registry_path=tmp_path / "registry.yaml",
    )

    assert report.status == "partial"
    assert [item.status for item in report.results] == ["failed", "passed"]
    assert len(runner.calls) == 2
    assert all(
        (tmp_path / "batch" / item.component_id / "batch_result.yaml").is_file()
        for item in report.results
    )
    assert (tmp_path / "batch" / "paper_adapter_certification.yaml").is_file()


def test_gpu_factory_runs_cpu_then_gpu_for_each_adapter(tmp_path: Path) -> None:
    runner = _Runner()
    report = PaperAdapterCertificationFactory(
        discovery=_Discovery(), runner=runner
    ).run(
        workdir=tmp_path / "batch",
        registry_path=tmp_path / "registry.yaml",
        mode="gpu",
        execute_real_gpu=True,
    )

    assert report.status == "passed"
    assert [str(item["mode"]) for item in runner.calls] == [
        "cpu",
        "gpu",
        "cpu",
        "gpu",
    ]
    assert all(item.final_maturity == "gpu_certified" for item in report.results)


def test_gpu_factory_is_blocked_without_explicit_opt_in(tmp_path: Path) -> None:
    runner = _Runner()
    report = PaperAdapterCertificationFactory(
        discovery=_Discovery(), runner=runner
    ).run(
        workdir=tmp_path / "batch",
        registry_path=tmp_path / "registry.yaml",
        mode="gpu",
        execute_real_gpu=False,
    )

    assert report.status == "failed"
    assert runner.calls == []
    assert {item.status for item in report.results} == {"blocked"}
    assert {item.errors[0] for item in report.results} == {
        "gpu_execution_not_confirmed"
    }
