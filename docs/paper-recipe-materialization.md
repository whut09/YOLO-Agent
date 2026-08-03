# Paper Recipe Materialization

Paper records and `RecipePrior` objects are non-executable. A paper-derived method reaches a pilot only through the certified materialization chain:

```text
frozen ResearchSnapshot
-> current-run, current-protocol COCO error facts
-> RecipePrior and canonical component contract
-> local runtime adapter lookup
-> compatibility check
-> runtime dry-run and smoke evidence
-> matched baseline control
-> RecipeCritic and eligibility gate
-> coverage-aware candidate priority and duplicate/cooldown guards
-> ASHA registration
-> RoundExecutionPlan
-> ExecutionQueue
```

## Hard Boundaries

- The adapter must be both `runtime_execution_ready` and artifact-backed `smoke_passed`.
- An adapter class or mock smoke result can prove implementation behavior, but cannot authorize a pilot.
- A runtime payload must be importable, hash-matched, protocol-bound, and invoked through the typed Python entrypoint.
- Missing adapters produce an `implementation_request`; the system does not generate adapter code during training.
- Missing current-node error facts produce evidence recovery only.
- Candidate and matched control keep `imgsz=640` and the same comparison protocol.
- `RecipePrior` and `policy_evaluation.yaml` have no direct queue authority.
- ASHA is the only pilot budget authority; `RoundExecutionPlan` is the only queue source.
- Scalar HPO is disabled by default. When certified paper/component recipes are exhausted, the loop stops explicitly.
- There is no fallback to an ordinary Ultralytics command after adapter preparation.

## Candidate Capacity Priority

After the hard gates pass, candidate capacity is ranked from current error-fact match,
compatible paper coverage, canonical mechanism confidence, verified runtime-hook
availability, implementation/GPU/deployment cost, and dataset-local Policy Memory.
Coverage uses a bounded logarithmic score: one reusable adapter can serve many papers,
but paper count cannot overwhelm diagnosis or confirmed local negative evidence. Paper
year is not part of this execution score.

The mechanism fingerprint includes component IDs, changed-variable names and values,
snapshot, baseline protocol, and coupling reason, but excludes paper IDs. Equivalent
paper sources therefore share one trial, while materially different recipe values stay
distinct. Completed fingerprints and recently attempted component families are deferred
by duplicate and cooldown guards before ASHA capacity is allocated.

Priority is not authorization. A high score cannot bypass adapter lookup, effective
`smoke_passed` maturity, runtime payload identity, compatibility, matched control,
RecipeCritic, eligibility, or ASHA.

## Runtime Maturity Bootstrap

`ComponentValidationBridge` is the non-training path from an implemented adapter to
an executable component contract. It imports the adapter, builds and round-trips the
typed runtime payload, instantiates every declared plugin, verifies its callable hooks,
and runs the adapter's local validation checks without creating an `ExperimentNode`.

The bridge persists three independent, content-hash-bound artifacts:

- `adapter_runtime_payload.<hash>.yaml` for `runtime_integrated`;
- `unit_tested_report.<hash>.yaml` for `unit_tested`;
- `smoke_passed_report.<hash>.yaml` for `smoke_passed`.

`component_validation.yaml` is the recoverable state pointer. Re-running the same
protocol, configuration, and adapter source resumes that state; changing any of those
inputs creates a different validation key. `patch_preview.yaml` remains diagnostic and
is never a maturity artifact.

Smoke provenance is adapter-reported and fail-safe. The default is `mock`; a caller
cannot relabel mock evidence as local evidence. Failed and mock smoke reports are kept
for audit but do not advance maturity. `ComponentExecutionBridge` does not perform this
bootstrap and continues to reject contracts whose non-mock smoke artifact is missing,
deleted, or hash-invalid.

## Audit Identity

For every registered component candidate, terminal events and `decision_ledger.jsonl` record:

- component and adapter IDs;
- adapter class and version where available;
- aggregate patch hash;
- runtime payload hash;
- ASHA assignment and queue authority.

These fields identify what actually ran. Paper claims remain separate prior evidence and never become candidate metrics.

## Failure Outcomes

| Condition | Outcome |
| --- | --- |
| No current protocol error facts | evidence recovery only |
| Metadata-only component | implementation request |
| Adapter import or runtime payload missing | implementation request |
| Runtime/unit/smoke artifact contract missing | candidate rejected; maturity unchanged |
| Smoke, compatibility, 640, or matched-control gate fails | candidate rejected |
| Cohort incomplete | wait for more certified `pilot_3` candidates |
| Certified recipe space exhausted | stop; no scalar HPO fallback |
| Candidate accepted | register with ASHA; ASHA may issue one bounded assignment |

Passing materialization does not claim local improvement. Promotion still requires complete COCO post-evaluation and verified paired evidence.
