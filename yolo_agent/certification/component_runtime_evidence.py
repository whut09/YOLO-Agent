"""Artifact validation shared by paper component acceptance tracks."""

from __future__ import annotations

from yolo_agent.adapters.ultralytics.plugin_context import PluginRuntimeEvidence
from yolo_agent.certification.runner import BackendRun
from yolo_agent.components.adapters.runtime import AdapterRuntimePayload


def validate_component_runtime_artifacts(
    run: BackendRun,
    *,
    component_id: str,
    protocol_hash: str,
) -> tuple[AdapterRuntimePayload, dict[str, bool | int | str]]:
    """Fail unless runtime artifacts prove the intended plugin actually ran."""
    required_names = {"runtime_payload", "plugin_runtime_evidence"}
    missing_names = sorted(
        name
        for name in required_names
        if name not in run.runtime_artifacts
        or not run.runtime_artifacts[name].is_file()
    )
    if missing_names:
        raise RuntimeError(
            "component runtime artifacts missing: " + ", ".join(missing_names)
        )
    payload = AdapterRuntimePayload.read(
        run.runtime_artifacts["runtime_payload"],
        verify_imports=True,
    )
    evidence = PluginRuntimeEvidence.model_validate_json(
        run.runtime_artifacts["plugin_runtime_evidence"].read_text(
            encoding="utf-8-sig"
        )
    )
    expected_missing: list[str] = []
    for artifact in payload.expected_artifacts:
        if not artifact.required:
            continue
        path = run.runtime_artifacts.get(artifact.name)
        if path is None or not path.is_file():
            expected_missing.append(artifact.name)

    required_hook_calls = 0
    missing_hooks: list[str] = []
    for reference in payload.plugin_references:
        calls = evidence.hook_call_counts.get(reference.reference, {})
        for hook in reference.required_hooks:
            count = int(calls.get(hook, 0))
            required_hook_calls += count
            if count <= 0:
                missing_hooks.append(f"{reference.reference}:{hook}")
    checks: dict[str, bool | int | str] = {
        "payload_component_matched": payload.component_ids == [component_id],
        "payload_protocol_matched": payload.protocol_hash == protocol_hash,
        "runtime_payload_hash_matched": evidence.payload_hash == payload.payload_hash,
        "runtime_protocol_matched": evidence.protocol_hash == protocol_hash,
        "runtime_components_matched": evidence.component_ids == [component_id],
        "runtime_changed_variables_matched": (
            evidence.changed_variables == payload.changed_variables
        ),
        "runtime_compatible": evidence.compatible,
        "runtime_failures_empty": not evidence.failures,
        "required_artifacts_complete": not expected_missing,
        "required_hooks_complete": not missing_hooks,
        "required_hook_calls": required_hook_calls,
    }
    failed = sorted(
        name
        for name, value in checks.items()
        if name != "required_hook_calls" and value is not True
    )
    details = [
        *failed,
        *[f"missing_artifact:{name}" for name in expected_missing],
        *[f"missing_hook:{name}" for name in missing_hooks],
    ]
    if details:
        raise RuntimeError(
            f"{component_id} runtime artifact contract failed: "
            + ", ".join(details)
        )
    return payload, checks


__all__ = ["validate_component_runtime_artifacts"]
