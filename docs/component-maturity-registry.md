# Component Maturity Registry

`ComponentMaturityRegistry` stores machine-local validation and certification evidence
without rewriting the conservative component YAML files under `configs/components/`.
The registry path is explicit so tests, research workspaces, and production machines do
not silently share maturity state.

Each `ComponentEvidenceOverlay` records:

- component ID and adapter source SHA-256;
- source code commit;
- installed Ultralytics version;
- protocol hash;
- immutable maturity artifact contracts and their SHA-256 values.

Registry updates use an OS file lock and atomic replacement. Repeating the same update
does not duplicate an overlay or artifact. Failed and mock artifacts remain available
for audit but cannot promote maturity.

`load_contracts(..., maturity_registry=..., protocol_hash=...)` starts from the source
contract and applies only matching adapter/runtime/protocol evidence. Missing or modified
artifact files are excluded, and promotion stops at the last valid adjacent maturity
stage. The source YAML is never changed. The code commit is retained for provenance;
the protocol hash already binds the code version used by the experiment.

`ComponentValidationBridge` and `apply_certification_report` can persist their outcomes
directly when supplied the same registry. A ResearchSnapshot build can also receive the
registry and target protocol. It copies valid evidence into the research production
artifacts, freezes the effective contracts and maturity summary, and includes the copied
evidence in the snapshot manifest. Later registry updates do not mutate an existing
snapshot.

The registry is evidence storage, not queue authority. `ComponentExecutionBridge`,
compatibility gates, budget gates, matched control, and ASHA remain mandatory.

## Component Certification

Use the advanced component command to create the local artifacts consumed by the
registry:

```powershell
yolo-agent advanced certify-component --component sampling.small_object --cpu
yolo-agent advanced certify-component --component sampling.small_object --gpu --device 0
```

The CPU command runs adapter import, runtime payload generation, hook-signature
validation, unit checks, and isolated local smoke in order. A mock result is retained
for audit but cannot promote the component. A passed CPU report advances only through
`smoke_passed` and is written to
`runs/certification/components/<component-id>/component_certification.cpu.yaml`.

The GPU command is explicit opt-in. It refuses to start unless the registry can load a
valid, hash-matched CPU `smoke_passed` overlay for the same adapter, code, Ultralytics
version, and certification protocol. The adapter must implement its own
`gpu_smoke_test`; the default implementation fails closed. A passed GPU report can
advance to `gpu_certified`, while a failed report remains evidence without promotion.

Both commands default to `runs/component_maturity_registry.yaml`. Use `--registry`
and `--workdir` to isolate a machine or CI workspace. These commands certify the
component runtime path only; they do not create a training node, claim pilot
reproduction, or authorize full COCO.
