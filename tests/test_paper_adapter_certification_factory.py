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
    def __init__(
        self,
        *,
        adapter_hash_a: str | None = None,
        ultralytics_version_a: str | None = None,
        protocol_hash_a: str | None = None,
    ) -> None:
        self.adapter_hash_a = adapter_hash_a
        self.ultralytics_version_a = ultralytics_version_a
        self.protocol_hash_a = protocol_hash_a

    def discover(self) -> ReusableAdapterDiscoveryResult:
        first = _descriptor("component.a")
        first = first.model_copy(
            update={
                "identity": first.identity.model_copy(
                    update={
                        **(
                            {"adapter_hash": self.adapter_hash_a}
                            if self.adapter_hash_a
                            else {}
                        ),
                        **(
                            {"ultralytics_version": self.ultralytics_version_a}
                            if self.ultralytics_version_a
                            else {}
                        ),
                        **(
                            {"protocol_hash": self.protocol_hash_a}
                            if self.protocol_hash_a
                            else {}
                        ),
                    }
                )
            }
        )
        return ReusableAdapterDiscoveryResult(
            adapters=[first, _descriptor("component.b")],
            errors={},
        )


class _Runner:
    def __init__(
        self,
        *,
        fail_component: str | None = None,
        adapter_hashes: dict[str, str] | None = None,
        ultralytics_versions: dict[str, str] | None = None,
    ) -> None:
        self.fail_component = fail_component
        self.adapter_hashes = adapter_hashes or {}
        self.ultralytics_versions = ultralytics_versions or {}
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
            adapter_hash=self.adapter_hashes.get(
                component_id,
                ("a" if component_id.endswith("a") else "b") * 64,
            ),
            code_commit="commit-one",
            ultralytics_version=self.ultralytics_versions.get(
                component_id, "8.4.0"
            ),
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


class _FixtureBuilder:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls: list[dict[str, object]] = []

    def build(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        if self.fail:
            raise ValueError("synthetic matched fixture failure")
        output = Path(str(kwargs["output"]))
        output.write_text("matched: true\n", encoding="utf-8")
        return object()


class _CoverageUpdater:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls: list[dict[str, object]] = []

    def refresh(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        if self.fail:
            raise ValueError("synthetic coverage failure")
        Path(str(kwargs["output_path"])).write_text(
            "coverage: updated\n", encoding="utf-8"
        )
        return object()


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
    fixtures = _FixtureBuilder()
    report = PaperAdapterCertificationFactory(
        discovery=_Discovery(), runner=runner, fixture_builder=fixtures
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
    assert len(fixtures.calls) == 2
    assert all(
        item.matched_pilot_fixture is not None
        and item.matched_pilot_fixture.is_file()
        for item in report.results
    )


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

    assert report.status == "blocked"
    assert runner.calls == []
    assert {item.status for item in report.results} == {"blocked"}
    assert {item.errors[0] for item in report.results} == {
        "gpu_execution_not_confirmed"
    }


def test_matched_fixture_failure_is_retained_per_adapter(tmp_path: Path) -> None:
    report = PaperAdapterCertificationFactory(
        discovery=_Discovery(),
        runner=_Runner(),
        fixture_builder=_FixtureBuilder(fail=True),
    ).run(
        workdir=tmp_path / "batch",
        registry_path=tmp_path / "registry.yaml",
        mode="gpu",
        execute_real_gpu=True,
    )

    assert report.status == "failed"
    assert {item.status for item in report.results} == {"failed"}
    assert {item.final_maturity for item in report.results} == {"gpu_certified"}
    assert all(
        item.selection_reason == "matched_pilot_fixture_failed"
        for item in report.results
    )


def test_resume_reuses_only_matching_verified_component_reports(tmp_path: Path) -> None:
    first_runner = _Runner()
    factory = PaperAdapterCertificationFactory(
        discovery=_Discovery(), runner=first_runner
    )
    first = factory.run(
        workdir=tmp_path / "batch",
        registry_path=tmp_path / "registry.yaml",
    )
    second_runner = _Runner()
    second = PaperAdapterCertificationFactory(
        discovery=_Discovery(), runner=second_runner
    ).run(
        workdir=tmp_path / "batch",
        registry_path=tmp_path / "registry.yaml",
        resume=True,
    )

    assert first.status == second.status == "passed"
    assert second.resumed_from_report_hash == first.report_hash
    assert second_runner.calls == []
    assert {item.status for item in second.results} == {"skipped_resume"}


def test_changed_only_runs_changed_identity_and_skips_unchanged(tmp_path: Path) -> None:
    PaperAdapterCertificationFactory(
        discovery=_Discovery(), runner=_Runner()
    ).run(
        workdir=tmp_path / "batch",
        registry_path=tmp_path / "registry.yaml",
    )
    runner = _Runner(adapter_hashes={"component.a": "c" * 64})
    report = PaperAdapterCertificationFactory(
        discovery=_Discovery(
            adapter_hash_a="c" * 64,
            protocol_hash_a="protocol-component.a-v2",
        ),
        runner=runner,
    ).run(
        workdir=tmp_path / "batch",
        registry_path=tmp_path / "registry.yaml",
        changed_only=True,
    )

    assert [str(item["component_id"]) for item in runner.calls] == ["component.a"]
    by_component = {item.component_id: item for item in report.results}
    assert by_component["component.a"].status == "passed"
    assert by_component["component.a"].selection_reason == (
        "identity_changed:adapter_hash,protocol_hash"
    )
    assert by_component["component.b"].status == "skipped_unchanged"


def test_resume_invalidates_ultralytics_version_change(tmp_path: Path) -> None:
    PaperAdapterCertificationFactory(
        discovery=_Discovery(), runner=_Runner()
    ).run(
        workdir=tmp_path / "batch",
        registry_path=tmp_path / "registry.yaml",
    )
    runner = _Runner(ultralytics_versions={"component.a": "8.5.0"})
    report = PaperAdapterCertificationFactory(
        discovery=_Discovery(
            ultralytics_version_a="8.5.0",
            protocol_hash_a="protocol-component.a-v2",
        ),
        runner=runner,
    ).run(
        workdir=tmp_path / "batch",
        registry_path=tmp_path / "registry.yaml",
        resume=True,
    )

    assert [str(item["component_id"]) for item in runner.calls] == ["component.a"]
    assert report.results[0].selection_reason == (
        "identity_changed:ultralytics_version,protocol_hash"
    )
    assert report.results[1].status == "skipped_resume"


def test_coverage_refresh_failure_is_reported_without_losing_results(
    tmp_path: Path,
) -> None:
    coverage = _CoverageUpdater(fail=True)
    report = PaperAdapterCertificationFactory(
        discovery=_Discovery(),
        runner=_Runner(),
        coverage_updater=coverage,
    ).run(
        workdir=tmp_path / "batch",
        registry_path=tmp_path / "registry.yaml",
    )

    assert report.status == "partial"
    assert report.coverage_error == "synthetic coverage failure"
    assert report.coverage_report_path is None
    assert {item.status for item in report.results} == {"passed"}
    assert len(coverage.calls) == 1


def test_passed_runner_report_with_wrong_identity_is_rejected(tmp_path: Path) -> None:
    report = PaperAdapterCertificationFactory(
        discovery=_Discovery(),
        runner=_Runner(adapter_hashes={"component.a": "f" * 64}),
    ).run(
        workdir=tmp_path / "batch",
        registry_path=tmp_path / "registry.yaml",
        component_ids=["component.a"],
    )

    assert report.status == "failed"
    assert report.results[0].selection_reason == "certification_identity_mismatch"
    assert report.results[0].errors == [
        "certification_identity_mismatch:adapter_hash"
    ]


def test_real_cpu_factory_certifies_sampling_adapter_without_gpu(
    tmp_path: Path,
) -> None:
    report = PaperAdapterCertificationFactory().run(
        workdir=tmp_path / "batch",
        registry_path=tmp_path / "registry.yaml",
        mode="cpu",
        component_ids=["sampling.small_object"],
    )

    assert report.status == "passed", report.results[0].errors
    assert report.execute_real_gpu is False
    assert report.results[0].final_maturity == "smoke_passed"
    assert report.results[0].cpu_report is not None
    assert report.results[0].cpu_report.is_file()
    assert report.coverage_report_path is not None
    assert report.coverage_report_path.is_file()
    assert (tmp_path / "registry.yaml").is_file()
