# Paper Candidate Routing Audit

## Scope

This document is the final offline audit of the paper-candidate routing work. It
describes what the repository can guarantee without running a GPU experiment.
The audited objective is overall COCO `mAP50-95` for YOLO26 with fixed
`imgsz=640`.

The system does **not** guarantee a `+0.02` mAP result. It guarantees that
compatible proposals are accounted for, executable proposals do not silently
disappear between routing stages, and an improvement is reported only from a
valid paired candidate/control comparison.

## Audited Call Chain

The implemented route is:

```text
diagnosis and PaperMethodProfile
  -> PaperRecipePlanner
  -> recipe critic
  -> PaperRecipeMaterializationGate
  -> PaperCandidateCoverageLedger
  -> RoundExecutionPlan
  -> _register_guarded_pilot_trials
  -> ASHAScheduler
  -> candidate plus matched baseline
  -> verified PairedExperimentResult
```

Canonical component resolution happens before planning and materialization.
Unknown identifiers remain unresolved; they are not guessed from similar text.
Execution deduplication uses the effective execution fingerprint, while paper
and method-profile identifiers are merged as provenance.

The persistent run artifact is
`runs/<run-id>/artifacts/paper_candidate_coverage.yaml`. Each proposal has one
current disposition and an append-only stage history covering planner, critic,
materialization, execution-plan, and ASHA registration boundaries.

## Proposal Dispositions

The allowed values are defined by
`yolo_agent/agents/paper_proposal_ledger.py`:

| Disposition | Meaning |
| --- | --- |
| `queued` | The execution fingerprint is eligible and registered or ready for ASHA allocation. |
| `already_tested` | The same fingerprint and protocol have completed, valid paired evidence. |
| `evidence_recovery` | Required local evidence is missing, stale, split-unsafe, or protocol-incompatible; a recovery requirement is retained. |
| `implementation_request` | The proposal is understood, but a required adapter or runtime binding does not exist. |
| `incompatible` | The proposal conflicts with YOLO26, fixed `imgsz=640`, the objective, or an explicit recipe constraint. |
| `blocked_runtime` | An implementation exists, but a checkpoint, resource, readiness check, or runtime condition prevents execution. |
| `deferred_budget` | The candidate remains eligible and recoverable but is delayed by allocation; it has not been discarded. |

`not selected` is not a terminal state. A planner, critic, materializer, or ASHA
boundary that cannot route a proposal must persist one of the dispositions
above with reason codes.

## improve-map-11 Coverage Fixture

The offline overall-mAP fixture drives the same diagnosis categories that were
observed in `improve-map-11`. Its planner inventory includes all eight required
atomic components:

| Canonical component | Primary recipe | Fixture routing | Matched facts |
| --- | --- | --- | ---: |
| `loss.hard_negative_classification` | `yolo26_hard_negative_classification_auxiliary_loss` | selected | 3 |
| `sampling.hard_negative_replay` | `yolo26_hard_negative_replay` | evidence recovery | 3 |
| `loss.quality.correlation` | `yolo26_correlation_auxiliary_loss` | selected | 3 |
| `loss.quality.pseudo_iou` | `yolo26_pseudo_iou_quality_auxiliary_loss` | selected | 3 |
| `assigner.task_aligned` | `yolo26_tood_tal_assignment_shadow` | selected | 3 |
| `assigner.optimal_transport` | `yolo26_ota_assignment_shadow` | selected | 3 |
| `distillation.yolo26_teacher_student` | `yolo26n_distillation` | selected | 3 |
| `neck.rtmdet_large_kernel` | `yolo26_rtmdet_large_kernel_neck` | selected | 4 |

The same fixture also observes these evidence-bound combinations:

| Combination recipe | Fixture routing | Matched facts |
| --- | --- | ---: |
| `yolo26_hard_negative_pair` | evidence recovery | 2 |
| `yolo26_rtmdet_correlation` | selected | 1 |
| `yolo26_rtmdet_pseudo_iou` | selected | 1 |

The replay candidates remain visible even when no split-safe train manifest is
available. Small-object-only proposals are likewise retained in the ledger,
but they cannot become the sole first-cohort source for an overall-mAP goal.

## Component Contracts

| Component | Contract and recipe | Runtime entrypoint | Required evidence and guards |
| --- | --- | --- | --- |
| `loss.hard_negative_classification` | `configs/components/loss/quality_alignment.yaml`; `configs/recipes/yolo26_quality_alignment.yaml` | `yolo_agent.components.adapters.losses.quality_alignment:QualityAlignmentRuntimePlugin`, `compute_loss` | Background false-positive or class-confusion facts; auxiliary-loss payload; matched protocol and primary mAP, latency, size guards. |
| `sampling.hard_negative_replay` | `configs/components/data_pipeline/paper_data_adapters.yaml`; `configs/recipes/yolo26_data_pipeline.yaml` | `yolo_agent.components.adapters.data_pipeline.sampling_plugin:DataSamplingRuntimePlugin`, `build_train_dataloader` | Train-side split-safe hard-negative manifest; dataset, baseline protocol, manifest hashes, and valid sample indices. |
| `loss.quality.correlation` | `configs/components/loss/quality_alignment.yaml`; `configs/recipes/yolo26_quality_alignment.yaml` | `yolo_agent.components.adapters.losses.quality_alignment:QualityAlignmentRuntimePlugin`, `compute_loss` | Localization/confidence mismatch facts; independent changed variable and payload; native YOLO26 DFL-free regression remains intact. |
| `loss.quality.pseudo_iou` | `configs/components/loss/quality_alignment.yaml`; `configs/recipes/yolo26_quality_alignment.yaml` | `yolo_agent.components.adapters.losses.quality_alignment:QualityAlignmentRuntimePlugin`, `compute_loss` | Localization/confidence mismatch facts; independent changed variable and payload; native regression is not replaced. |
| `assigner.task_aligned` | `configs/components/assigner/yolo26_assignment.yaml`; `configs/recipes/yolo26_assignment_shadow.yaml` | `yolo_agent.components.adapters.assigners.yolo26_assignment:YOLO26AssignmentRuntimePlugin`, `compute_loss` | Shadow minimum batches, valid positive assignments, native-loss equivalence, latency/memory checks, then an active matched pilot. |
| `assigner.optimal_transport` | `configs/components/assigner/yolo26_assignment.yaml`; `configs/recipes/yolo26_assignment_shadow.yaml` | `yolo_agent.components.adapters.assigners.yolo26_assignment:YOLO26AssignmentRuntimePlugin`, `compute_loss` | Independent shadow and active state; the task-aligned result cannot consume this trial; matched protocol required. |
| `distillation.yolo26_teacher_student` | `configs/components/distillation/yolo26_teacher_student.yaml`; `configs/recipes/yolo26n_distillation.yaml` | `yolo_agent.components.adapters.distillation.yolo26_distillation:YOLO26DistillationRuntimePlugin`, `compute_loss` | Frozen teacher checkpoint and SHA256; student/teacher dataset and split agreement; protocol and `imgsz=640`; student-only export, latency, and size. |
| `neck.rtmdet_large_kernel` | `configs/components/neck/yolo26_multi_scale.yaml`; `configs/recipes/yolo26_multi_scale_necks.yaml` | `yolo_agent.components.adapters.neck.runtime:YOLO26NeckRuntimePlugin`, `build_model` | Graph identity, shape contract, CPU build/forward smoke, adapter hash and rollback plan; latency, peak VRAM, and model-size guards. |

Abstract `loss.quality.iou_aware_classification` proposals do not substitute for
the independently executable correlation and pseudo-IoU candidates. Without an
explicit implementation binding, the abstract mechanism is recorded as
`implementation_request`.

## Coupled Templates and Attribution

`configs/coupled_recipe_templates.yaml` is the allow-list for coupled
generation. The generator may also use an explicit method-profile coupling
reason or verified local diagnosis, but it does not construct an unconstrained
Cartesian product.

| Template | Allowed role in this audit |
| --- | --- |
| `hard_negative_loss_replay` | Hard-negative classification loss plus split-safe hard-negative replay. |
| `p2_small_object_sampling` | Audited small-object-only combination; not first-cohort for overall mAP. |
| `feature_fusion_quality_loss` | Explicit allow-listed neck plus quality-loss pairs, including RTMDet with correlation or pseudo-IoU. |
| `teacher_student_class_balanced_sampling` | YOLO26 teacher/student plus class-balanced sampling when class-imbalance and capacity evidence exist. |
| `distillation_class_balanced_sampling` | Allow-listed general distillation plus class-balanced sampling. |
| `assignment_quality_alignment` | Task-aligned or optimal-transport assignment plus correlation or pseudo-IoU after the corresponding shadow passes. |
| `slicing_confidence_calibration` | Isolated inference evaluation; it does not claim a training-recipe attribution. |

Every training combination has four attribution arms under the same data and
evaluation protocol:

1. baseline: neither component enabled;
2. A: only component A enabled;
3. B: only component B enabled;
4. A+B: both components enabled with the declared coupling semantics.

Each active arm has a matched baseline control. The ledger records component
IDs, paper provenance, coupling reason, combination ID/fingerprint, and the
internal ablation plan. A missing replay manifest, assignment shadow, teacher,
or coupling fact changes the disposition; it does not silently turn the
combination into an unrelated atomic trial.

## Automatic Training Boundary

The same `yolo-agent train` command performs cached CPU contract, payload,
shape, build, and forward readiness checks during materialization. A readiness
failure blocks only the affected fingerprint; other eligible candidates remain
queued or deferred.

Candidates can proceed automatically when all of the following local state is
present:

- quality and hard-negative auxiliary losses have diagnosis facts, a complete
  runtime payload, fixed `imgsz=640`, and matched-control metadata;
- hard-negative replay has a train-side, split-safe manifest with matching
  dataset and baseline-protocol hashes;
- each assignment candidate has independently passed its shadow evidence gate;
- distillation resolves a frozen teacher checkpoint whose hash, dataset, split,
  and protocol bindings match the student run;
- the RTMDet neck passes cached graph identity, shape, build, and forward smoke
  checks;
- coupled candidates satisfy their allow-listed pair, evidence, and prerequisite
  state.

The following are legitimate non-training outcomes rather than lost proposals:

- missing or stale train-side evidence: `evidence_recovery`;
- missing teacher checkpoint or hash mismatch: `blocked_runtime` or
  `evidence_recovery`, depending on whether recovery is local evidence work;
- missing adapter/runtime binding: `implementation_request`;
- graph, fixed-image-size, YOLO26, or objective conflict: `incompatible`;
- external GPU process or temporarily unavailable resource: `blocked_runtime`,
  resumable without reducing the preserved batch;
- per-candidate contract, shape, forward, hook, or candidate OOM failure:
  `blocked_runtime`/candidate failure with an artifact, without deleting other
  trials;
- eligible cohort beyond the current allocation: `deferred_budget`, retained
  for later ASHA assignment.

No manual certification command is required for a cache miss. Readiness is not
reported as training and a readiness failure is not reported as a completed
search.

## Paired Result Rules

A mAP delta is valid only when both candidate and matched baseline complete and
the pairing check verifies the same dataset manifest, split, protocol,
`imgsz`, fidelity, seed policy, and metric contract. The execution fingerprint
also binds model checkpoint identity, canonical components, effective
overrides, teacher/graph/runtime payload hashes, and coupled-combination ID.

If either arm fails, or if split/protocol identity differs, the result is
`unavailable`: the system may report the individual metric but must say that no
improvement was measured. Old debug output or unmatched evidence does not make
a fingerprint `already_tested`.

ASHA may use a verified paired result for screening and promotion. Paper claims,
paper year, LLM recommendations, readiness smoke results, and shadow assignment
metrics are not mAP improvements.

## Delivery Limits

This audit supports the following promise: compatible proposals remain visible
with a terminal or recoverable disposition, runtime-ready/evidence-complete
candidates reach execution planning and ASHA, and only verified paired evidence
can support an accuracy claim.

It cannot promise that any candidate, paper implementation, or combination
improves COCO by `+0.02` mAP. That target remains an empirical result requiring
successful pilots, promotion, full training, and seed confirmation.

## Offline Verification

The final audit was run on Windows with CUDA explicitly disabled for pytest. No
real GPU training was started.

| Check | Result |
| --- | --- |
| Routing acceptance: `pytest -q tests/test_paper_candidate_routing_acceptance.py tests/test_overall_map_paper_routing_acceptance.py tests/test_paper_candidate_coverage_acceptance.py` | `14 passed in 5.78s` |
| Full suite: `pytest -q` | `2046 passed, 34 skipped in 1571.24s` |
| `ruff check .` | passed |
| `python -m compileall yolo_agent tests` | passed |
| `git diff --check` | passed; Git emitted only the local LF-to-CRLF conversion warning for this Markdown file |

The previously reported assignment-shadow, distillation/quality ASHA, and
orchestrator evidence-recovery regressions are resolved. Missing diagnostic
metrics now retain evidence-domain recovery proposals while continuing to
block training proposals. Low-level ASHA fixtures also model the current
automatic-readiness and execution-fingerprint contracts.

## Repository Artifact Audit

`git ls-files` contains no generated `runs/` tree, runtime log directory,
checkpoint, model weight, ONNX export, TensorRT engine, or safetensors file.
Local experiment output under `runs/` remains outside Git and was not deleted.

The pre-existing untracked paths `.tmp-cert-sampling/` and
`docs/codex-paper-candidate-routing-prompts.md` were not modified, staged, or
committed by this audit.

## Next Training Command

Start a fresh run ID; do not reuse an exhausted or failed search ID:

```powershell
yolo-agent train --model yolo26n.pt --data E:\datatset\coco.yaml --run-id <new-run-id> --goal +2map
```
