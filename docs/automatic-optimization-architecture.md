# Automatic Optimization Architecture

> Full system architecture: [architecture.md](architecture.md) — this
> document covers the optimization loop only: how the agent autonomously
> improves detection quality on a dataset, round by round, under measured
> evidence.

## What the loop does

YOLO Agent is a diagnosis-driven experiment manager around the Ultralytics
training runtime. A user states one problem, for example:

```text
Small objects are missed too often; improve AP_small without a large latency or
overall mAP regression.
```

The loop builds a trusted baseline, derives error facts from real ground
truth and real predictions, maps the facts to canonical mechanisms,
materializes compatible recipes, runs matched candidate/control experiments,
and records paired deltas. It never promises that a paper will improve a
dataset. The output of every round is a measured decision, including
rejection, evidence-recovery, and "wait for full approval" outcomes.

## The complete decision loop

Each round walks the same chain. Every arrow is a function in the code; the
module is named so each step can be verified.

```mermaid
flowchart TD
    A["Evidence<br/>DetectionErrorProfile built from real GT + predictions<br/>core/detection_error_profile_builder.py"] --> B["ErrorFact<br/>structured facts + ErrorFactStore<br/>core/error_facts.py"]
    B --> C["RootCauseHypothesis<br/>DiagnosisGraph findings (+ advisory LLM drafts)<br/>agents/diagnosis_graph.py"]
    C --> D["ActionFamily<br/>28 canonical families<br/>agents/action_space_schemas.py"]
    D --> E["ActionSpec<br/>parameters + search space + rollback + required_evidence<br/>configs/actions/detection_action_catalog.yaml"]
    E --> F["CandidatePortfolio<br/>CandidateGenerator + CandidatePlan<br/>agents/candidate_generator.py"]
    F --> G["Critic<br/>LLMProposalCritic (deterministic)<br/>agents/llm_proposal_critic.py"]
    G --> H["Compatibility<br/>CompatibilityChecker + assess_matched_control_plan<br/>components/compatibility.py, core/matched_baseline.py"]
    H --> I["Maturity<br/>can_execute: >= smoke_passed + non-mock artifact<br/>components/contracts.py, certification/component_queue_gate.py"]
    I --> J["Runtime Materialization<br/>RecipeMaterializer + ComponentExecutionBridge<br/>recipes/recipe_materializer.py, components/execution_bridge.py"]
    J --> K["Matched Experiment<br/>candidate + control ExperimentNodes, round plan authority<br/>core/round_execution_plan.py"]
    K --> L["ASHA<br/>guarded registration, 4-rung budget ladder<br/>agents/asha_scheduler.py"]
    L --> M["ErrorDelta<br/>DetectionErrorDelta + ErrorDeltaProfile<br/>core/detection_error_delta.py"]
    M --> N["Decision<br/>decide_next_round + promotion verdicts + objective status<br/>agents/loop_decision_engine.py, core/optimization_objective.py"]
    N -->|"next round"| A
```

Notes on two steps where advisory input exists:

- **RootCauseHypothesis**: an LLM may draft hypotheses and explanations
  (`agents/llm_decision_advisor.py`), but the hypothesis set entering
  planning is produced by the rule-based `DiagnosisGraph`; LLM drafts are
  proposal input only.
- **CandidatePortfolio**: LLM-proposed recipes enter the pool only after the
  deterministic `LLMProposalCritic` rejects proposals without target error
  facts, with more than one changed variable, or violating fixed constraints
  (`imgsz=640`); when paper routes are blocked the loop falls back to
  `decision_mode="deterministic_fallback"`
  (`agents/policy_stage_runner.py`).

## Unified optimization space

The loop plans over one action catalog
(`configs/actions/detection_action_catalog.yaml`, 34 actions across all 28
`ActionFamily` values in `agents/action_space_schemas.py`). Every action
declares a `runtime_phase` — `data | train | inference | eval` — and pydantic
validators structurally enforce the phase boundary: an inference-only family
cannot declare a training phase, and a model-graph family cannot declare an
inference phase (`action_space_schemas.py:190-219`).

The single most important rule of this section: **a component existing in the
catalog is not an executable capability.** Execution requires effective
maturity at or above `smoke_passed` with a verifiable, non-mock artifact
(`components/contracts.py`, `can_execute`); the queue gate blocks anything
below it with `effective_maturity_below_smoke_passed`
(`certification/component_queue_gate.py`). Components that are
metadata-only, recipe-idea-only, or legacy cards without adapters are
planning vocabulary — never executable experiments.

### Data-phase actions (`runtime_phase: data`)

These actions produce data-side work items, not training runs (see the next
section for why they never become ASHA trials).

| Dimension | ActionFamily | Catalog actions | Runtime support |
|-----------|--------------|-----------------|-----------------|
| Data selection | `data_selection` | `data.selection.small_object_focus` | sampling adapters (small-object weighted, class-balanced, repeat-factor) |
| Annotation | `annotation` | `data.audit.annotation_repair` | `AnnotationQualityFilterAdapter`; annotation advice artifacts (`advise_labels`) |
| Data cleaning | `data_cleaning` | `data.cleaning.noise_filter` | `AnnotationQualityFilterAdapter` |
| Sampling | `sampling` | `data.sampling.class_aware`, `data.sampling.hard_negative_mining` | sampling adapters; hard-negative replay manifest from the **train** split (`tools/coco_error_mining.py`) |
| Augmentation | `augmentation` | `data.augmentation.small_copy_paste` | copy-paste / crop transform adapters |
| Preprocessing | `preprocessing` | `data.preprocessing.tile_inference_input` | `NormalizationPreprocessingAdapter`, tiling input policy |

### Model-graph actions (`runtime_phase: train`)

| Dimension | ActionFamily | Catalog actions | Runtime support |
|-----------|--------------|-----------------|-----------------|
| Backbone | `backbone` | `model.backbone.pretrained_swap` (pretrained-weight init + freeze schedule) | **No structural backbone adapter exists.** The legacy `backbone.dsconv` card is metadata-only and not executable |
| Neck | `neck` | paper routes (RTMDet large kernel etc.) | 11 neck adapters (weighted FPN, gold-gd, reparameterized conv, deformable aggregation, attention, lightweight neck, …) |
| Feature Fusion | `feature_fusion` | `model.feature_fusion.gather_distribute` | `GoldGatherDistributeAdapter` |
| Head | `head` | `model.head.task_aligned` | `TaskAlignedHeadAdapter`, `P2HeadAdapter` |
| Input policy | `input_resolution` | `model.resolution.train_imgsz`, `model.resolution.inference_sweep` | **Train-time imgsz is pinned to 640 by policy** (`recipes/paper_priors.py`, `core/matched_baseline.py`); only inference-side resolution sweeps are free |

### Training-objective actions (`runtime_phase: train`)

| Dimension | ActionFamily | Catalog actions | Runtime support |
|-----------|--------------|-----------------|-----------------|
| Assignment | `assignment` | `train.assigner.optimal_transport`, `train.assigner.dynamic_smooth_label` | `YOLO26AssignmentAdapter` |
| BBox loss | `bbox_loss` | `train.bbox_loss.quality.correlation`, `train.bbox_loss.calibration.bpc` | quality-alignment aux-loss adapter; legacy bbox-loss cards (CIoU/WIoU/MPDIoU/NWD) are metadata-only |
| Classification loss | `classification_loss` | `train.cls_loss.positive_reweight` (local) | shared training-control primitives |
| Auxiliary loss | `auxiliary_loss` | `train.aux.mutual_supervision`, `train.aux.pseudo_iou` | `MutualSupervisionAdapter`, `QualityAlignmentAuxiliaryLossAdapter` |
| Distillation | `distillation` | `train.distill.decoupled_features` + 35 paper routes | 36 distillation adapters (`components/adapters/distillation/`) |
| Domain adaptation | `domain_adaptation` | `train.domain.2503_23220` + paper routes | 34 domain-adaptation adapters |
| Semi-supervised | `semi_supervised` | `train.semi.pseudo_label_filter` (local only) | shared primitives (`components/pseudo_label_filter.py`, `components/teacher_ema.py`) via domain-adaptation modes; **no dedicated paper adapter — the frozen-83 set contains no semi-supervised paper** |

### Training-control actions (`runtime_phase: train`, config-level)

| Dimension | ActionFamily | Catalog actions | Runtime support |
|-----------|--------------|-----------------|-----------------|
| Training strategy | `training_strategy` | `train.strategy.ema_schedule`, `train.strategy.freeze_schedule` | `components/training_control.py` primitives; config-level, no dedicated epoch-running adapter |
| Optimizer | `optimizer` | `train.optimizer.lr_schedule_shift` | LR/warmup schedule shift only; **no optimizer-swap adapter exists** |
| Regularization | `regularization` | `train.regularization.aug_schedule` | mosaic close-epochs + weight decay, config-level with rollback |

### Inference-phase actions (`runtime_phase: inference | eval`)

| Dimension | ActionFamily | Catalog actions | Runtime support |
|-----------|--------------|-----------------|-----------------|
| Postprocess | `postprocess` | `inference.nms.soft`, `inference.nms.diou` | inference adapters (merge policy, class-aware threshold, tiled, SAHI slicing) — **executable** |
| Inference | `inference` | `inference.tta.multi_scale` | `TestTimeAugmentationAdapter`, `TiledMultiScaleInferenceAdapter` |
| Calibration | `calibration` | `inference.calibration.temperature` (`fit_split: val`) | `ConfidenceCalibrationInferenceAdapter` — score transform only, ranking unchanged; inference-side only |

Inference-only candidates are structurally excluded from training ASHA:
`register_trial` raises `inference-only candidate cannot enter training ASHA`
(`agents/asha_scheduler.py`), and the eval-route HPO path (below) never
authorizes training-graph changes.

## Data actions are not training runs

A round can legitimately conclude that the best next step is not a model
run. The loop has first-class, budget-free mechanisms for this:

- **Decision vocabulary**: `RoundDecisionAction` includes `collect_data` and
  `request_annotation` (`core/error_round_decision.py`). The only actions
  that consume training budget are `BUDGET_ACTIONS = {"promote", "refine"}`;
  every other decision books or defers instead of training. When the
  objective regresses and the dataset is imbalanced, the decision engine
  emits `collect_data`; otherwise `request_annotation`, which adds a
  `missing_labels` problem tag to the next round's diagnosis.
- **Artifact chain, not executors**: the data stages produce work-list
  artifacts and stop there —
  - `advise_labels` → `annotation_advice.json`/`.md` (classes to collect,
    scenes to annotate, boxes to redraw, labeling-tool targets;
    `agents/data_stage_runner.py`),
  - `mine_samples` → `active_learning_plan.json` (low-confidence /
    high-entropy / model-disagreement acquisition; `tools/active_learning.py`),
  - `label_handoff` → `labeling_manifest.json` + `label_handoff.json`
    (`status: ready_for_labeling`, targeting CVAT/Label Studio),
  - `dataset_promote` → `dataset_promotion.json` with a
    `pending_label_review` gate — the stage **never mutates data**; promotion
    happens only after human label review (`agents/active_learning_stage_runner.py`).
- **ASHA isolation**: a data-side action has no `candidate_config.components`
  plus adapter runtime entrypoint, which is exactly what
  `_is_paper_trial` requires (`agents/asha_scheduler.py`). Annotation
  recommendation, hard-negative collection, and data repair therefore can
  never become a `TrainingNode` or consume ASHA budget — they are work items
  for the next diagnosis round, backed by lineage records
  (`evidence.record_lineage`).

## Six coverage numbers

The loop separates these quantities, and each automatic round prints the
funnel and writes it to `artifacts/executable_portfolio.yaml`:

1. Catalog papers: papers known to the offline catalog.
2. Method profiles: papers with local evidence describing their method.
3. Recipe definitions: reusable mechanism recipes, including paper priors.
4. Runtime-ready recipes: recipes whose components have frozen implementation
   identity and valid non-mock maturity artifacts.
5. Diagnosis-matched recipes: recipes bound to the current run's error facts.
6. Executable trials: candidate/control experiments actually registered with
   ASHA.

The first three are knowledge assets. Only the last three describe what can
participate in the current training run.

## Runtime authority

Paper names and titles are priors. Training authorization comes from the
frozen research snapshot and its effective maturity manifest:

```text
PaperMethodProfile -> canonical mechanism -> recipe -> adapter contract
  -> adapter hash + Ultralytics version + protocol hash
  -> non-mock runtime/unit/smoke evidence -> matched pilot/control
```

At train startup, the CLI discovers all reusable training adapters rather
than only a fixed shortlist. It refreshes missing or stale CPU evidence
automatically. A failed adapter is isolated and reported; it cannot silently
become ordinary Ultralytics training. Snapshot identity is checked again
before materialization.

Reviewed local recipe families are merged by the research production
pipeline before the snapshot is hashed. Training reads only the frozen
recipe registry; it never fills gaps from a live paper or recipe registry
during a run. Source paths and parse errors are recorded in the executable
portfolio artifact.

## Why low-fidelity pilots can regress

Three-epoch, ten-percent pilots are screening measurements, not final
claims. The default ASHA policy treats small pilot deltas inside a
`-0.0015` noise band as rankable cohort evidence. Severe regressions,
invalid paired evidence, diagnosis failures, and resource-guard failures are
still eliminated early.

`pilot_10` still requires a verified positive paired delta and target
error-fact improvement. Full training and confirmation seeds remain explicit
`--confirm-full-run` operations. A pilot signal is never reported as exact
paper reproduction.

## Bounded HPO

The repository implements bounded hyperparameter search as a constrained
surface (`agents/bounded_hpo.py`), with these exact rules:

1. **Global, unconstrained HPO is prohibited.** There is no global random
   search (`bounded_hpo.py` module contract). Legacy scalar-HPO recipes are
   disabled by configuration (`configs/loop_policy.yaml`
   `allow_scalar_hpo: false`, `configs/training_recipes.yaml`
   `enable_scalar_hpo: false`) and frozen at schema level
   (`scalar_hpo_enabled: Literal[False]`); the loop skips them with
   `deferred_budget` / `scalar_hpo_disabled` and records
   `budget_authority: "ASHA"` in the rejection event
   (`agents/auto_optimization_loop.py`).
2. **HPO happens only after an eligible action is selected.** The search
   space is derived from the chosen `ActionSpec` alone
   (`search_space_from_action`); nothing outside the action surface is ever
   swept.
3. **Only the ActionSpec/adapter-declared search space is eligible.** Points
   outside the declared parameter set or declared grid are rejected
   structurally — `param_not_in_declared_search_space` /
   `values_outside_declared_grid` raise `HpoRequestRejected`; silent pruning
   of out-of-scope keys is forbidden. Planning is capped (`max_points=8`)
   and filters previously failed combinations through policy memory
   (`core/hpo_candidate_planner.py`).
4. **ASHA remains the budget authority.** Epochs and data fraction come only
   from ASHA rung assignments (`pilot_3 → pilot_10 →
   candidate_full_seed_1 → candidate_full_confirmation`); HPO never allocates
   budget, and inference-only candidates cannot enter training ASHA at all.
5. **Validation only, never tune on test.** Eval-route trials are pinned to
   `split: Literal["val"]`, and threshold/postprocess/calibration requests
   with any other split raise `threshold_tuning_requires_validation_split`;
   the fixed COCO post-evaluation used for all pilot/full fidelity levels
   also evaluates on `val` with `imgsz` locked at 640
   (`adapters/ultralytics/coco_post_eval.py`). Test-split evaluation is a
   reporting step, never a tuning signal.

Honest wiring status: the constraint surface above is implemented and
unit-tested, but `plan_hpo_points` / `request_hpo_points` have no production
caller in the automatic loop yet — the live loop uses
`hpo_scope_for_action` to label each executed action's HPO scope. HPO point
scheduling is therefore bounded-by-construction, not yet an active loop
behavior.

## Reading an error delta

Why a single mAP number is not a decision: the loop compares matched
candidate/parent profiles at the evidence level
(`core/detection_error_delta.py`, `build_detection_error_delta`), section by
section, and only comparable pairs (same `gt_artifact`,
`dataset_manifest_hash`, `split`) produce deltas.

Worked example from one hypothetical round:

```text
Metric level (ErrorDeltaProfile)
  AP_small        +4.2   (0.312 -> 0.354)
  latency         +21%   (deployment guardrail allows <= +5%)
  model size      +2%    (guardrail allows <= +10%)

Evidence level (DetectionErrorDelta)
  small-object false negatives   -15%   (mechanism worked)
  background false positives      +9%   (side effect)
  verdict per section: improved / regressed / unchanged / not_comparable
```

What the loop sees that a flat overall mAP would hide:

- **Overall mAP can stay flat while the mechanism works.** The small-object
  gain (+4.2 AP_small, -15% small FN) can be offset by the background FP
  regression (+9%), so a single headline number would rank this candidate as
  noise. Slice-level deltas show the mechanism did what the paper claimed.
- **The failure slice chooses the next mechanism.** The +9% background FP
  growth points at postprocess/threshold or sampling families for the next
  round — an attribution that an aggregate mAP cannot provide.
- **Deployment constraints are invisible in mAP.** A +21% latency regression
  exceeds the objective guardrail (`max_latency_regression` 0.05); the
  objective status flips to `target_reached_pending_guard_evidence` with
  blocker `latency_regression_exceeds_objective`, and the candidate cannot
  be confirmed regardless of its metric gain
  (`core/optimization_objective.py`).

## Stopping policy

The loop never ends because "no recipe was found" alone; stopping is a
categorised decision persisted to `artifacts/optimization_objective_status.yaml`
(`should_stop`, `stop_reason`, `blockers`) and per round to
`auto_round_summary.yaml`. The real stop reasons in code
(`core/optimization_objective.py`, `core/loop_status.py`,
`agents/auto_optimization_loop.py`), grouped by intent:

| Category | Real stop reason literals | Meaning |
|----------|---------------------------|---------|
| Goal met | `objective_confirmed` | metric target reached **and** ≥ 3 confirmation seeds **and** paired delta CI lower bound ≥ required delta **and** trusted full baseline **and** latency/size guardrails passed |
| Budget exhausted | `gpu_budget_exhausted`, `max_pilot_rounds_reached`, `requested_rounds_completed` | GPU-hour budget (`max_gpu_hours`, default 24), pilot-round budget (default 12), or the safety round cap (`max_rounds_safety: 60`) |
| Candidates exhausted | `no_guarded_candidates`, `no_executable_candidates`, `method_candidates_exhausted`, `family_exhaustion`, `paper_adapter_implementation_required`, `no_certified_paper_components` | the planning funnel is genuinely empty: nothing passed guards, nothing is executable, all method families are tried/cooled down, or the next step needs an implementation request |
| Evidence incomplete | `missing_error_facts`, `asha_evidence_incomplete`, `matched_control_failed`, `assignment_shadow_evidence_invalid` | the loop refuses to plan or conclude without valid paired evidence |
| Repeated failure | `no_improvement_patience_reached`, `training_failed` | no improvement for the configured patience (objective default 4 rounds, diversity policy 5), or training failures exhausted the round |
| Full approval required | `target_reached_pending_full_confirmation`, `target_reached_pending_guard_evidence` | the target is reached but confirmation is blocked: missing seeds, unconfirmed CI, guardrail evidence, or the full-run gate |

Two properties of this policy worth knowing:

- **Round-level reasons are not stops.** `asha_candidates_registered`,
  `diversity_deferred`, `asha_assignment_completed`, and similar round
  summaries end a round while the loop continues; only the run-level set
  above ends the search (`agents/auto_optimization_loop.py`, main-loop break
  set).
- **Full budget is never auto-granted.** ASHA issues no
  `candidate_full`/confirmation assignments without an explicit
  `--confirm-full-run` plus a valid `FullRunConsent` (bound to
  objective/dataset/protocol hashes and the GPU-hours cap). Without it, a
  fully successful search ends in
  `target_reached_pending_full_confirmation`, and
  `full_candidate_recommendations.yaml` records the exact command hint to
  resume with confirmation. On the single-command path the same gate is a
  preflight error before any training starts
  (`agents/optimize_runner.py`).
