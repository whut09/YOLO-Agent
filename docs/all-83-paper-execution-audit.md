# All 83 Paper Execution Audit

Status: offline acceptance only. No GPU training was started by this audit.

## Scope And Boundary

The audit uses the frozen production coverage at
`research/production/paper_method_coverage.yaml`, the paper registry, the
paper-specific mechanism resolver, the recipe registry, and CPU/mock execution
objects. A paper inventory row is provenance and routing evidence. It is not a
training result.

The execution boundary is:

```text
production paper coverage
  -> PaperExecutionInventory (one row per compatible paper)
  -> paper-specific mechanism resolution
  -> recipe/materialization disposition
  -> RoundExecutionPlan (active or deferred node)
  -> ASHA trial identity (execution fingerprint)
  -> matched candidate/control evidence
  -> verified paired result
```

Only the final step can claim an accuracy change. CPU contract checks, recipe
presence, mock ASHA registration, and paper claims cannot produce mAP delta.

## Coverage Layers

| Layer | Count | Meaning |
|---|---:|---|
| Paper inventory coverage | 83 | Every YOLO26-compatible paper has exactly one `PaperExecutionSpec`. |
| Paper-specific implementation coverage | 18 records / 18 distinct resolved mechanism IDs | 18 records have explicit resolver output with paper-specific changed-variable, payload, and evidence identity. The other records retain an unresolved or generic-parent explanation. |
| Recipe coverage | 15 papers | These papers have at least one registered recipe identity. The remaining papers retain `implementation_request`, `evidence_recovery`, or `blocked_runtime`; they are not silently dropped. |
| Runtime-ready coverage | 0 | No production paper currently passes the effective runtime-ready inventory gate. This is intentionally not inferred from a generic adapter or CPU smoke result. |
| ASHA eligible coverage | 0 production papers | No production paper is authorized for real ASHA by this offline audit. Mock-ready fingerprints are tested separately and never persisted as real training evidence. |
| Actual trained coverage | 0 | This audit starts no training and creates no candidate metric artifact. |
| Exact reproduction coverage | 0 | `exact_reproduction_candidates=0` remains an explicit production fact. |

Current inventory dispositions:

| Disposition | Count |
|---|---:|
| `queued` | 0 |
| `runtime_ready` | 0 |
| `already_tested` | 0 |
| `evidence_recovery` | 0 |
| `implementation_request` | 68 |
| `incompatible` | 0 |
| `blocked_runtime` | 15 |
| `deferred_budget` | 0 |

These counts describe the current offline inventory, not a promise that the
15 blocked papers or 68 implementation requests are executable today.

## Paper-Specific Separation

The inventory retains the generic parent component as provenance where it is
present, but generic IDs do not authorize execution:

- 40 papers are associated with `domain_adaptation.general`; they are not
  treated as one implemented domain-adaptation algorithm.
- 32 papers are associated with
  `distillation.yolo26_teacher_student`; they are not treated as one
  implemented distillation algorithm.
- Resolved examples include `feature_distillation`, `relation_distillation`,
  `logits_distillation`, `localization_distillation`,
  `pseudo_label_adaptation`, `source_free_adaptation`, and the independent
  assignment, quality, neck, pyramid, attention, calibration, and inference
  mechanisms.
- Unresolved generic or unknown mechanisms carry an explicit unresolved reason
  and cannot become executable candidates.

## Runtime Guards

The acceptance suite verifies these fail-closed rules with the protocol
registry and mock adapters:

- `inference.sahi_slicing` is an inference candidate and cannot enter training
  ASHA.
- Domain adaptation without explicit source and target data is
  `evidence_recovery` and is blocked from COCO mAP training. COCO train/val
  splits are not accepted as substitutes for the paper domains.
- Distillation without a frozen teacher checkpoint and matching teacher/student
  dataset evidence is `evidence_recovery` and cannot enter training.
- Hard-negative replay without a train-side manifest remains evidence recovery;
  validation predictions cannot be used as training replay evidence.
- Model-graph candidates retain YOLO26 one-to-one head, native DFL-free
  regression, graph identity, and fixed `imgsz=640` constraints.
- A candidate and matched baseline with different protocol identity produce no
  paired delta.

## Execution Identity And ASHA

ASHA deduplicates by execution fingerprint, not paper count. The fingerprint
includes model identity, canonical components, recipe/version, effective
overrides, dataset manifest, baseline protocol, image size, fidelity, seed,
teacher/runtime payload identity, graph identity, and combination identity.
Different paper provenance may share one trial only when that execution
identity is equal. Distinct implementations remain distinct trials.

The mock acceptance path verifies that every unique mock-ready fingerprint is
represented in both `RoundExecutionPlan` and ASHA, while deferred nodes remain
recoverable. It also verifies that one failed trial leaves other paper trials
present.

Evidence-bound coupled recipes retain the four required arms:

1. `baseline`
2. `arm_A`
3. `arm_B`
4. `arm_A_plus_B`

Each non-baseline arm requires a matched baseline control. Assignment
combinations require passed shadow evidence. A blocked component blocks only
that combination and cannot be reported as executable.

## Acceptance Result

The new acceptance suite is CPU/mock-only and checks all 18 requested
boundaries, including inventory uniqueness, generic-method separation,
recipe/disposition retention, plan and ASHA routing, coupled ablation shape,
inference/domain/distillation/hard-negative gates, protocol mismatch, failure
isolation, and silent-drop prevention.

Passing this audit means candidate provenance and eligibility decisions are
auditable. It does **not** mean the 83 papers are exact reproductions, that
all are runtime-ready, that any candidate has been trained, or that the model
improves mAP. A future training result may claim improvement only after both
candidate and matched baseline complete under the same protocol and produce a
verified paired result.
