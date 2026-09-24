# Training Release

The training release is the hash-pinned snapshot taken after the readiness
pipeline ([training-readiness.md](training-readiness.md)) passes. It is the
object every real training entrypoint verifies before running. Module:
`research/training_release.py` (`training_release.v1`); built and
immediately re-verified by `yolo-agent papers release`, which writes
`artifacts/training_release_v1.yaml`.

The release exists because gates answer "is the evidence surface ready?" but
not "is *this* checkout the one the evidence was produced on?". The release
pins the whole chain to hashes and re-checks them live.

## What the release pins

`TrainingRelease` fields (as defined in code):

| Field | Content |
|-------|---------|
| `release_id` | `training-release-v1-<git12>-<membership12>` |
| `release_status` | `READY_FOR_FIRST_TRAINING` or `BLOCKED` |
| `git_commit` | the commit the release was built on |
| `paper_membership_hash` | frozen campaign membership hash |
| `paper_ready_count` / `paper_blocked_count` | read from the readiness registry |
| `implementation_registry_hash` | readiness registry payload hash |
| `acceptance_hash` | pretraining acceptance payload hash |
| `allowed_training_modes` | budget profiles unlocked when READY (debug/pilot/pilot_3/pilot_10/baseline_full/baseline_confirm/candidate_full/full) |
| `lock_reasons` | every failed condition when BLOCKED |
| `real_training_executed` | `Literal[False]` — a release never claims training happened |
| `release_hash` | canonical SHA-256 of the body excluding itself |

`ReleaseEvidence` — the pinned hashes:

| Field | Pins |
|-------|------|
| `manifest_membership_hash` / `manifest_file_sha256` | campaign identity + `configs/research/paper_83_manifest.yaml` file bytes |
| `acceptance_file_sha256` / `acceptance_report_hash` | `artifacts/pretraining_acceptance.yaml` file + payload |
| `implementation_registry_file_sha256` / `implementation_registry_report_hash` | readiness registry file + payload |
| `runtime_preflight_file_sha256` | `artifacts/paper_83_runtime_preflight.yaml` (hard gate) |
| `non_gpu_acceptance_file_sha256` | `artifacts/non_gpu_test_acceptance.yaml` (hard gate) |
| `test_result_hashes` | exactness audit + acceptance artifacts |
| `component_runtime_hashes` | paper route files (distillation / domain adaptation) |
| `component_identity_hashes` | per-paper `implementation_fingerprint` from the registry |
| `adapter_source_hashes` | MRO source hash per `<implementation_path>#<adapter_class>` (Prompt-18H) |
| `runtime_dependency_hashes` | non-MRO transitive local dependency modules (Prompt-18H) |
| `git_commit` | release identity repeated inside evidence |

## Build and READY judgment

`build_training_release()` (`research/training_release.py`) produces READY
only when **all** of the following hold: acceptance verdict is `PASS` with
`training_gate.allowed`; ready count meets the requirement and blocked is
zero; all five required acceptance sections passed; the two hard-gate
artifacts (runtime preflight, non-GPU acceptance) re-verify directly;
manifest membership recomputes to the same 83 unique IDs; a **live** re-run
of the training gate agrees with the recorded acceptance; and adapter source
collection resolves with no unresolvable paths. Any failure produces a
BLOCKED release with `lock_reasons` — never a partial release.

## Release verification

`verify_training_release()` re-checks, live, at every training entry:
release self-hash intact; status READY; ready/blocked counts; manifest,
acceptance, and registry file hashes still current; membership hash still
matches; component runtime hashes current; runtime-preflight and non-GPU
hard-gate artifact hashes current; test-result hashes current; per-paper
identity hashes current; **adapter source hashes and runtime dependency
hashes recomputed from the working tree**; and the pinned `git_commit` equal
to or an ancestor of the current HEAD (fail-closed when git is unavailable).

### Source drift

Any edit to an adapter class in the MRO, or to any transitive local
dependency module, changes its source hash. The recomputed
`adapter_source_hashes` / `runtime_dependency_hashes` no longer match the
pinned values and verification fails. This is intentional: the acceptance
evidence was produced by specific source code, and "close enough" source is
not the same source.

### Artifact drift

Any change to the pinned artifacts (manifest, acceptance, preflight,
non-GPU acceptance, registry, exactness audit) after the release was built
fails the corresponding file-hash or payload-hash check. Artifacts must be
regenerated and a new release built rather than patched in place.

### Fail-closed semantics

Every check failure is a hard stop — there is no warning mode, no bypass
flag, and no fallback. Enforcement points, in order:

1. `yolo-agent train` entry (L0.7) — exits 2 with a hint to run
   `yolo-agent papers release`;
2. `yolo-agent optimize` seam (L0.8);
3. executor seam — release re-verification after the paper-83 gate, refusal
   wrapped as `training_release_not_verified:*`;
4. Ultralytics runtime entrypoint `release_guard()` — the last gate, cannot
   be skipped by any caller (`TrainingReleaseMissingError`);
5. candidate eligibility — release fingerprint without drift is one of the
   five hard eligibility conditions for ASHA registration;
6. `papers final-readiness` — read-only check for audits.

## Current release

The committed release artifact is `artifacts/training_release_v1.yaml`. Its
status, counts, and lock reasons are read from that file (or from
`yolo-agent papers final-readiness`) — this document deliberately does not
restate them.
