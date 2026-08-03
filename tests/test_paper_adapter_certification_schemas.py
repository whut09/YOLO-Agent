from yolo_agent.certification.paper_adapter_factory_schemas import (
    AdapterCertificationIdentity,
    PaperAdapterCertificationReport,
    PaperAdapterCertificationResult,
)


def _identity(component_id: str = "sampling.small_object") -> AdapterCertificationIdentity:
    return AdapterCertificationIdentity(
        component_id=component_id,
        adapter_hash="a" * 64,
        code_commit="commit-one",
        ultralytics_version="8.4.0",
        protocol_hash="protocol-one",
    )


def test_batch_report_round_trip_preserves_adapter_identity(tmp_path) -> None:
    identity = _identity()
    report = PaperAdapterCertificationReport(
        status="passed",
        mode="cpu",
        registry_path=tmp_path / "registry.yaml",
        selected_component_ids=[identity.component_id],
        results=[
            PaperAdapterCertificationResult(
                component_id=identity.component_id,
                identity=identity,
                status="passed",
                initial_maturity="adapter_implemented",
                final_maturity="smoke_passed",
                selection_reason="selected_all_reusable_adapters",
                cpu_report=tmp_path / "component.cpu.yaml",
            )
        ],
    )

    path = tmp_path / "batch.yaml"
    report.to_yaml(path, exclude_none=True, sort_keys=False)
    restored = PaperAdapterCertificationReport.from_yaml(path)

    assert restored == report
    assert restored.report_hash == report.calculate_hash()
    assert len(identity.identity_hash) == 64
