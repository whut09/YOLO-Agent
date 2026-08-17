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
