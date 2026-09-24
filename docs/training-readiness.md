# Training Readiness Pipeline

This document describes the six gates between "papers are implemented" and
"the first real training run". Each section states what the gate **proves**,
what it explicitly does **not** prove, the command that runs it, and the
artifact it writes. Definitions of the campaign itself live in
[paper-83-campaign.md](paper-83-campaign.md); the hash-pinned snapshot taken
after these gates is described in [training-release.md](training-release.md).

Pipeline order:

```text
Paper readiness → Exactness Audit → Runtime Preflight → Non-GPU Tests
  → Pretraining Acceptance → Training Gate → (Training Release) → train
```

Three of the stage documents are CLI-regenerated outputs — do not hand-edit
them: `docs/paper-implementation-readiness.md` (`papers readiness`),
`docs/paper-83-exactness-audit.md` (`papers audit`),
`docs/PRETRAINING_ACCEPTANCE.md` (`papers acceptance`).

## 1. Paper readiness

- **Command / artifact**: `yolo-agent papers readiness` →
  `runs/paper-readiness/paper_implementation_registry.yaml` +
  `docs/paper-implementation-readiness.md`.
- **What it checks**: for each frozen paper, a complete non-generic
  `PaperImplementationSpec` — importable adapters, valid runtime payload,
  non-mock unit/smoke maturity artifacts, test references, YOLO26
  compatibility, `paper_specific` evidence class, no blockers
  (`research/paper_implementation_readiness.py`).
- **Proves**: every paper has a complete, bound, statically verifiable
  implementation contract.
- **Does not prove**: any runtime behavior — no tensor is executed, no
  trainer runs, and the reproduction booleans are hardcoded `False`. The
  module never creates evidence or changes the maturity registry.

## 2. Exactness Audit

- **Command / artifact**: `yolo-agent papers audit` →
  `artifacts/paper_83_exactness_audit.yaml` + `artifacts/paper_83_gap_queue.yaml`
  + `docs/paper-83-exactness-audit.md`.
- **What it checks**: a 17-point evidence inventory per paper (membership,
  method profile validity, mechanism evidence, spec completeness,
  not-generic/alias/metadata/no-op, paper-specific composition, real runtime
  hooks, fingerprints, test presence, rollback path, shared-primitive usage,
  core-mechanism coverage) by joining the six side audits with the
  implementation registry (`research/paper_exactness.py`).
- **Proves**: the evidence surface is complete, correctly categorized, and
  nothing generic or aliased is posing as a paper-specific implementation. A
  result below 83/83 is a legitimate outcome; the audit reports what exists.
- **Does not prove**: runtime executability (that is preflight), and it is
  **not** a comparison of recipe hyperparameters against paper-reported
  numbers, nor a mechanism-fidelity recomputation — it inventories evidence
  only.

## 3. Runtime Preflight

- **Command / artifact**: `yolo-agent papers runtime-preflight` →
  `artifacts/paper_83_runtime_preflight.yaml`.
- **What it checks**: per paper, the real runtime path executes on synthetic
  CPU tensors — forward pass, finite scalar loss, finite gradients, and a
  behavior probe (perturbed inputs/teacher must change the loss by more than
  1e-9, so constant-loss mock implementations fail). Prompt-18G fail-closed:
  any PASS with unknown runtime-hook identity is downgraded to FAIL
  (`research/paper_runtime_preflight.py`).
- **Proves**: on the current commit, all 83 runtime paths actually run and
  respond to inputs, with auditable hook identities — "the code runs".
- **Does not prove**: accuracy, training convergence, behavior on real data,
  or any reproduction. No training loop is started anywhere
  (`real_training_executed: false`).

## 4. Non-GPU Tests

- **Artifact**: `artifacts/non_gpu_test_acceptance.yaml` (committed record
  bound to a git commit; written outside the CLI by the acceptance run).
- **What it checks**: the full CPU test surface — `pytest -q` (fast tier),
  `pytest -q --run-slow` (slow tier), and `ruff check .`, all green.
- **Proves**: the repository's non-GPU code surface — tests and lint — is
  healthy at the pinned commit.
- **Does not prove**: anything about GPU behavior, training correctness, or
  metric outcomes.

## 5. Pretraining Acceptance

- **Command / artifact**: `yolo-agent papers acceptance` →
  `artifacts/pretraining_acceptance.yaml` + `docs/PRETRAINING_ACCEPTANCE.md`.
- **What it checks**: seven sections — `paper_campaign` (frozen manifest
  integrity, implementation-ready counts, no generic/metadata/no-op/mock
  evidence), `optimization_action_space` (required action families covered),
  `autonomous_loop` (multi-round diagnosis → delta → ASHA → rollback →
  stop-policy simulation), `safety` (82/83 blocked, 83/83 allowed,
  fail-closed on manifest loss or hash tampering, full-run consent boundary),
  `tests`, `runtime_preflight` (83/83, zero unknown hooks), and
  `non_gpu_verification`. A pydantic invariant makes
  `training_gate.allowed ≡ all critical checks pass ≡ verdict == "PASS"` —
  a record claiming PASS with a failed section cannot be constructed
  (`research/pretraining_acceptance.py`). Both artifacts are written on PASS
  and FAIL alike.
- **Proves**: the acceptance system itself is consistent across all seven
  surfaces at the evaluated commit.
- **Does not prove**: that training has started or will succeed
  (`real_training_executed: false`), and it is not the release — the release
  adds hash freezing and live re-verification on top.

## 6. Training Gate

- **Where it runs**: the earliest interception in `yolo-agent train`
  (L0), the optimize seam, the executor seam (train commands only), and the
  Ultralytics runtime entrypoint as the last gate (exit code 86 when locked)
  — `research/paper_83_training_gate.py`.
- **What it checks**: fail-closed evaluation of the frozen manifest and the
  exactness audit — manifest present/valid/hash-matched, exactly 83 papers,
  `implementation_ready` as the only counting readiness, ready count at the
  requirement, zero blocked papers, no mock-smoke or generic-only evidence.
  Twelve `GateLockReason` values cover every failure mode.
- **Proves**: the evidence surface unlocks training for the frozen cohort.
- **Does not prove**: that a verified, drift-free release exists. **Gate
  allowed ≠ permission to train** — release verification is a separate,
  subsequent check (see [training-release.md](training-release.md)).

## Training start boundary

**Implementation-ready is not paper reproduced.** Everything above is
offline, static, or synthetic; the first five gates and the gate decision can
be earned without a single training step. The levels that describe training
outcomes — `pilot_reproduced` (verified paired `pilot_3`+`pilot_10` evidence),
`full_reproduced` (full-run reproduction report), `confirmed_multi_seed`
(multi-seed confirmation report) — can only be earned **after** training
starts, from hash-bound, non-mock, verified artifacts written by real runs.
Before that happens, every governance artifact correctly reads
`real_training_executed: false`, and `docs/paper-adapter-coverage.yaml`
reports zero reproduction counts.
