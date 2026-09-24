# Paper-83 Campaign

This document defines the frozen Paper-83 implementation campaign: what the
three different "83" numbers mean, how campaign membership is frozen and
hashed, what a `PaperImplementationSpec` is, and the hard boundary between
*implementation ready* and *paper reproduced*.

Current per-paper status tables live in CLI-generated documents
([paper-implementation-readiness.md](paper-implementation-readiness.md),
[paper-83-exactness-audit.md](paper-83-exactness-audit.md),
[PRETRAINING_ACCEPTANCE.md](PRETRAINING_ACCEPTANCE.md)) and machine artifacts
(`runs/paper-readiness/paper_implementation_registry.yaml`,
`artifacts/pretraining_acceptance.yaml`,
`artifacts/training_release_v1.yaml`). This document explains definitions,
not current counts — never copy a count from here into a claim.

## Three completely different numbers

### 1. `83/85` — certified-adapter mapping (coverage metric, frozen report)

This is the coverage-acceptance metric `compatible_papers_certified_adapter`
from the frozen report `docs/paper-coverage-acceptance.yaml`
(`report_hash: 797c3b912852717b03e3ce7fc55a3650d8b028f7d1dc9fc2a827c65c5996667c`):

- **Numerator (83)**: YOLO26-compatible papers whose reusable adapter
  candidates include at least one adapter with smoke-level evidence —
  `certified_adapter_ids = reusable_adapter_candidates ∩ smoke_ids`
  (`research/coverage_acceptance.py`).
- **Denominator (85)**: `yolo26_compatible_papers` at the time that report was
  frozen. **Drift note**: the current
  `research/production/coverage_baseline.yaml` denominator is 84; the 85
  exists only inside the frozen historical report, and the campaign manifest
  records `acceptance_lineage_status: historical` accordingly.
- **What it means**: *this paper can reuse a smoke-certified shared adapter.*
  It is a reuse-coverage measurement over the 728-paper catalog (catalog-wide
  it is 83/728), **not** paper implementation and **not** reproduction.

### 2. `83` — the frozen implementation campaign (membership list)

The 83 numerator IDs of that metric became the frozen campaign:
`configs/research/paper_83_manifest.yaml` (`schema_version:
paper_implementation_campaign.v1`, `frozen: true`). The schema pins the count
structurally — `expected_paper_count: Literal[83]`, `paper_count:
Literal[83]`, `papers` constrained to exactly 83 entries — so a campaign of
any other size cannot be constructed. The frozen list binds the acceptance
hash of its source report and the repository commit at freeze time.

### 3. `83/83` — implementation ready (offline readiness audit, generated)

The offline readiness audit result over the frozen 83. Current values are
always read from generated status, never restated here:

- `runs/paper-readiness/paper_implementation_registry.yaml` (per-paper
  `readiness: implementation_ready`),
- `artifacts/pretraining_acceptance.yaml` (`paper_campaign` section),
- `artifacts/training_release_v1.yaml` (`paper_ready_count`,
  `paper_blocked_count`).

`implementation_ready` is defined in the next sections. It unlocks training;
it is not a training outcome.

## Campaign membership and membership hash

- **Membership** = the sorted, deduplicated `numerator_ids` of the
  `compatible_papers_certified_adapter` metric.
  `extract_frozen_paper_ids()` (`research/paper_83_campaign.py`) enforces
  exactly 83 unique sorted IDs, each with a certified-adapter trace, each
  within the report denominator — fail-closed.
- **`membership_hash`** (`research/paper_83_campaign_schemas.py`,
  `calculate_membership_hash`) is the SHA-256 of the canonical JSON
  `{"paper_ids": [sorted ids]}` — and *only* the sorted paper-id membership,
  never mutable paper status. `Paper83Manifest.validate_manifest` recomputes
  it on every load and rejects any mismatch; the readiness evaluator refuses
  any spec whose `manifest_membership_hash` differs.
- Freezing is also fail-closed at build time: `build_paper_83_manifest()`
  asserts the README campaign shape (85/85 profiles, 83/85 adapter mapping)
  and locates the acceptance report by its exact hash before accepting
  membership.

## PaperImplementationSpec

`research/paper_implementation_schemas.py` (`paper_implementation_spec.v1`):
**one independent implementation contract per frozen paper**. Key content:

- identity: `paper_id`, `manifest_membership_hash`, `method_profile_id`;
- mechanism: `paper_specific_mechanisms` (non-generic), `mechanism_summary`,
  `implementation_domain`, `shared_primitives`;
- binding: `component_ids`, `adapter_ids`, `runtime_insertion_points`,
  `runtime_hooks`;
- change surface: ten change buckets (architecture / data / loss / training
  control / graph / assignment / distillation / domain / inference / …),
  `paper_specific_config`, `paper_specific_hyperparameters`;
- verification: unit/smoke/compatibility test references, the four booleans
  (`runtime_implementation_verified`, `unit_tests_passed`,
  `non_mock_smoke_passed`, `compatibility_validation_passed`), and an
  `implementation_fingerprint` — SHA-256 over membership hash + paper +
  profile + mechanisms + components + adapters + config + hyperparameters +
  insertion points + hooks + contract signatures;
- status: `readiness`, `blockers`, `implementation_evidence_class`, and the
  three reproduction booleans (`pilot_reproduced`, `full_reproduced`,
  `confirmed_multi_seed`, all default `False`).

`PaperImplementationRegistry` requires every record to carry the same
membership hash. Built by `yolo-agent papers readiness`.

## Readiness ladder and the reproduction boundary

`PaperImplementationReadiness`
(`research/paper_implementation_schemas.py`):

```text
cataloged → profiled → spec_complete → code_bound → runtime_integrated
  → unit_tested → smoke_passed → implementation_ready
  → pilot_reproduced → full_reproduced → confirmed_multi_seed
```

`implementation_ready` requires all of the following — every one an
offline, static, or synthetic-tensor check: complete non-generic method
profile; non-empty component/adapter/insertion-point/hook bindings; adapter
importable as a `ComponentAdapter`; runtime payload valid; non-mock
`unit_tested` and `smoke_passed` maturity artifacts; YOLO26 compatibility
(imgsz 640, one-to-one head, dfl-free); evidence class `paper_specific`; no
blockers.

**The offline evaluator can never grant reproduction levels.** It hardcodes
`pilot_reproduced=False, full_reproduced=False, confirmed_multi_seed=False`
(`research/paper_implementation_readiness.py`). Those levels are earned only
by real training artifacts:

- `pilot_reproduced` requires verified, promoted paired evidence from real
  `pilot_3` **and** `pilot_10` runs
  (`certification/paper_auto_optimization_maturity.py`; the single legal
  promotion step is `gpu_certified → pilot_reproduced`);
- `full_reproduced` and `confirmed_multi_seed` require full-run and
  multi-seed confirmation reports respectively (`components/maturity.py`,
  `_ARTIFACT_TARGETS`), each hash-bound and non-mock.

Current reproduction evidence in this repository: none —
`docs/paper-adapter-coverage.yaml` has `pilot_reproduced_ids: []`, no
`pilot_reproduced` evidence files exist under
`research/production/component_maturity_evidence/`, and every governance
artifact carries `real_training_executed: false`. That is the expected state
before the first training run.

## How this relates to the component maturity ladder

The per-component ladder (`ComponentMaturity`,
[capability-maturity.md](capability-maturity.md)) and the per-paper readiness
ladder above are parallel systems joined by evidence: a paper reaches
`implementation_ready` only through its components' non-mock
runtime/unit/smoke artifacts, and later reproduction promotions write
component-level evidence back. The readiness pipeline that consumes both is
described in [training-readiness.md](training-readiness.md), and the
hash-pinned snapshot that locks it before training is described in
[training-release.md](training-release.md).
