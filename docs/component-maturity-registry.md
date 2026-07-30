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
yolo-agent advanced certify-component --component loss.quality.correlation --cpu
yolo-agent advanced certify-component --component loss.calibration.bpc --cpu
yolo-agent advanced certify-component --component loss.quality.pseudo_iou --cpu
yolo-agent advanced certify-component --component distillation.yolo26_teacher_student --cpu
```

The CPU command runs adapter import, runtime payload generation, hook-signature
validation, unit checks, and isolated local smoke in order. A mock result is retained
for audit but cannot promote the component. A passed CPU report advances only through
`smoke_passed` and is written to
`runs/certification/components/<component-id>/component_certification.cpu.yaml`.

`sampling.small_object` has an additional CPU golden fixture. Its `smoke_passed`
artifact is issued only after the real Ultralytics train dataloader hook creates a
protocol-bound sampler manifest and passes DDP, resume, and validation-loader isolation
checks.

The three quality-loss components each run an independent AtomicRecipe golden path.
The fixture sends the same YOLO26 predictions through the native criterion, a
zero-weight runtime payload, and the active runtime payload. Certification requires
exact zero-weight equivalence, a changed active total loss, student backward, and
paper-prior evidence that explicitly sets `exact_reproduction=false`.

The distillation golden path loads real YOLO26n and YOLO26s model graphs without
downloading checkpoints. It requires the distillation total to enter the student loss,
student-only backward, a frozen/eval/no-grad teacher, zero-weight native equivalence,
an unchanged student inference graph, and MethodProfile-only paper attribution.

The GPU command is explicit opt-in. It refuses to start unless the registry can load a
valid, hash-matched CPU `smoke_passed` overlay for the same adapter, code, Ultralytics
version, and certification protocol. The adapter must implement its own
`gpu_smoke_test`; the default implementation fails closed. A passed GPU report can
advance to `gpu_certified`, while a failed report remains evidence without promotion.

Both commands default to `runs/component_maturity_registry.yaml`. Use `--registry`
and `--workdir` to isolate a machine or CI workspace. These commands certify the
component runtime path only; they do not create a training node, claim pilot
reproduction, or authorize full COCO.

Generate the effective machine-local coverage after certification with:

```powershell
python -m yolo_agent.tools.paper_adapter_coverage `
  --registry runs/component_maturity_registry.yaml `
  --local-report runs/paper-adapter-coverage.local.yaml
```

This local report applies only valid adapter-hash and Ultralytics-version overlays.
The committed public coverage report remains based on conservative source contracts.

For small-object sampling, component-level `gpu_certified` still does not claim a
useful detector result. The separate mini COCO certification must pass matched
pilot_3, COCO post-eval, AP_small/FN paired promotion, ASHA, and pilot_10 before this
component can enter the automatic training queue.
