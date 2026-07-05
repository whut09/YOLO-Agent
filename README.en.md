# yolo-agent

[中文](README.md) | English

YOLO Agent is an evidence-driven object-detection optimization harness.

It is not a free-form code-generation agent, and it does not blindly generate model code or start training. It runs a controlled, auditable loop:

```text
task + data + errors + constraints
        -> diagnosis
        -> policy proposals
        -> guarded candidates
        -> evidence
        -> next round
```

## Closed Loop

The central design rule is simple: LLMs, humans, and rule engines may propose policies, but only evaluators and evidence gates can turn proposals into experiment candidates.

## Automation Maturity

YOLO Agent is being built as an agent harness, so automation maturity is measured by how much of the optimization loop is explicit, resumable, evidence-gated, and auditable.

Current maturity: **Level 4, with Level 5 foundations in place.** The harness can generate guarded candidates, persist loop state, queue execution, import candidate-level evidence, compare runs, and track lineage. Active-learning and dataset-versioning primitives exist, but a production Level 5 loop still requires operational relabel integrations and dataset evolution policies wired into routine runs.

- **Level 1: schema + metadata**: task specs, scenario configs, component cards, compatibility metadata, and reproducible experiment schemas
- **Level 2: guarded candidate generation**: candidate policies pass through compatibility checks, deployment constraints, smoke guards, and single-variable ablation discipline
- **Level 3: evidence-driven loop**: loop state, stage contracts, evidence gates, decision ledger, artifact manifest, reports, and next-round planning
- **Level 4: queued execution + cross-run learning**: execution queue, executor boundary, candidate/node-level metrics, lineage tracking, forked runs, and cross-run comparison
- **Level 5: active learning + dataset version evolution**: uncertainty mining, relabel worklists, dataset manifest diffs, dataset version promotion, and retraining loop handoff

## Executor Boundary

Execution is modeled explicitly without making training the default:

```text
ExperimentNode -> CommandSpec -> ExecutionResult -> EvidenceStore
```

`loop enqueue` materializes planned `ExperimentNode` objects into a resumable queue before execution:

```text
ExperimentPlan -> ExecutionQueue -> Executor -> ExecutionResult -> EvidenceStore
```

Available executor abstractions:

- `DryRunExecutor`: records what would run without executing the command
- `ShellExecutor`: explicit subprocess execution for controlled commands
- `UltralyticsExecutor`: conservative Ultralytics smoke/draft executor that does not start real training by default
- `UltralyticsTrainExecutor`: explicit training executor for typed `yolo detect train ...` commands, with resume, DDP device strings, multi-GPU device lists, log capture, timeout handling, and result import
- `RuntimeProfiler`: extracts GPU utilization, GPU memory, it/s, epoch time, dataloader wait, batch size, and cache mode from Ultralytics args/results/logs plus optional `nvidia-smi` samples, then writes candidate/node-level evidence
- `DataCachePolicy`: chooses `cache=ram`, `cache=disk`, or conservative no-cache from dataset size, available RAM, and storage kind; when RAM is unsafe but NVMe is available it prefers `cache=disk`
- `BatchTuner`: runs short batch 32/48/64/96 probes before real training, records OOM, it/s, and GPU evidence, and chooses the highest-throughput batch without changing imgsz
- `BenchmarkImporter`: imports external benchmark metrics or Ultralytics run directories into run-level and candidate/node-level evidence

Real training must be selected explicitly:

```bash
yolo-agent loop execute --run runs/exp001 --executor ultralytics-train
```

## What It Optimizes

YOLO Agent treats detection performance as a full-system problem, not just a model-architecture problem.

It can reason about model scale, components, annotation quality, dataset health, augmentation, post-processing, deployment limits, reproducibility, ablation discipline, and evidence quality.

## Loop Harness

The loop orchestrator is a state machine, not a script chain. It persists run context, loop state, event logs, dataset manifests, lineage, execution queues, artifact manifests, decision ledgers, execution results, and stage artifacts under `runs/{run_id}`.

Stage order is defined by `configs/loop_policy.yaml`; the saved `LoopState` is built from that policy rather than a hardcoded Python execution list:

```text
init -> profile_data -> advise_labels -> diagnose_errors -> generate_loop_plan
-> evaluate_policies -> generate_candidates -> ablate -> smoke
-> import_metrics -> report -> next_round
-> mine_samples -> label_handoff -> dataset_promote
```

Stages with missing required evidence become `blocked` so the run can be resumed instead of silently producing untrusted recommendations.

```bash
yolo-agent loop --run runs/exp001 --resume
```

`fork-next` materializes `artifacts/next_round.yaml` into a fresh child run under the same run root. Cross-run lineage is appended to `runs/lineage.jsonl`.

Each stage is governed by an executable contract, not only Python control flow. Stage starts, completions, failures, resume attempts, and contract blocks are appended to `events.jsonl`.

Stage outputs are recorded in `artifacts/artifact_manifest.jsonl` with SHA-256 metadata. Artifact contracts can require current-run manifest entries, valid hashes, and optional Pydantic schemas such as `DatasetReport`, `CandidatePlan`, or `SmokeRunResult`.

## CLI

Initialize a scenario:

```bash
yolo-agent init --scenario infrared_small_target --output task.yaml
```

Run the loop in explicit phases:

```bash
yolo-agent loop init --run-id exp001 --task task.yaml --data data.yaml --training-config configs/training/yolo26_coco_goal.yaml
yolo-agent loop diagnose --run runs/exp001 --errors errors.yaml
yolo-agent loop plan --run runs/exp001
yolo-agent loop enqueue --run runs/exp001
yolo-agent loop execute --run runs/exp001 --executor dry-run
yolo-agent loop smoke --run runs/exp001
yolo-agent loop ingest-metrics --run runs/exp001 --metrics results.csv
yolo-agent loop mine --run runs/exp001 --predictions unlabeled_predictions.json
yolo-agent loop next --run runs/exp001
yolo-agent loop run-stage --run runs/exp001 --stage mine_samples
yolo-agent loop run-stage --run runs/exp001 --stage label_handoff
yolo-agent loop run-stage --run runs/exp001 --stage dataset_promote
yolo-agent loop fork-next --run runs/exp001 --new-run-id exp002
yolo-agent loop lineage --run-root runs --run exp002
yolo-agent loop lineage --run-root runs --best
yolo-agent loop compare --runs runs/exp001 runs/exp002 --out comparison.md
```

Training budget profiles and FastBaselineGate keep quick checks separate from trusted COCO evidence. The default flow is `1 epoch sanity -> 10 epoch pilot -> full baseline -> 3 seed confirmation`:

- `debug`: COCO `fraction=0.01`, `epochs=1`, `val=false`; sanity only.
- `pilot`: COCO `fraction=0.1`, `epochs=10`, fixed `batch=64`; screen candidates before full budget.
- `baseline_full`: full COCO, `epochs=100`, single seed; allowed only after pilot passes.
- `baseline_confirm`: full COCO, `epochs=100`, seeds `1,2,3`; allowed only after full baseline passes.
- `candidate_full`: full COCO, `epochs=100`, seeds `1,2,3`; only for candidates that passed pilot.

BaselineAcceptanceGate blocks `candidate_full` until the baseline is trusted: `map50_95` must exist and be verified, `results.csv` / `best.pt` / `args.yaml` must have sha256 manifest entries, the dataset manifest sha must match, `imgsz` must equal `640`, the profile must be `baseline_full` or `baseline_confirm`, the seed protocol must be satisfied, and any severe runtime bottleneck must be explained. Otherwise it records `baseline_trusted: false` plus `baseline_rejection_reason` and prevents full candidate promotion.

COCO baseline evidence also has its own contract: each `baseline_full` / `baseline_confirm` node must write standardized node-level `map50_95`, `ap_small` / `ap_medium` / `ap_large`, `per_class_ap/*`, `per_class_ar/*`, `latency_ms`, `model_size_mb`, runtime profile metrics, and sha256 artifact manifest entries for `results.csv`, `best.pt`, `args.yaml`, `runtime_profile`, and `coco_eval`. Official COCO `coco_ap50_95` is normalized to harness-standard `map50_95`.

COCO Error Fact Selection chooses the next diagnostic focus from baseline COCO eval facts and writes it into `next_round.yaml`: `top_unresolved_diagnoses` is the ranked unresolved list, `current_round_focus` is the narrow error scope for this round, and `current_round_error_actions` is the allowed action set for proposals. Small-object AP, bottle/person recall, and localization-heavy classes can become pilot-only experiment targets instead of letting the agent generate generic candidates.

CandidatePromotionGate makes pilot-to-full promotion explicit: the same candidate must have passing `debug` evidence and passing `pilot` evidence, pilot error facts must improve at least one target diagnosis, and `latency_ms`, `runtime_avg_it_per_sec`, or `runtime_epoch_time_seconds` must not regress too much against the baseline. Otherwise it records `candidate_full_allowed: false` plus `candidate_promotion_rejection_reason`.

ResourceScheduler checks local resources before an execution queue item actually runs: GPU visibility and idleness, free VRAM against `CommandSpec.resource_requirements`, candidate-level batch tuner evidence, resume checkpoint availability after retries, high-risk candidate deferral, and full COCO budget windows. Items may move to `paused`, `blocked_by_resource`, or `needs_resume`, so the agent does not launch every full COCO experiment at once.

```bash
yolo-agent loop init --run-id exp001 --task task.yaml --data data.yaml --training-config configs/training/yolo26_coco_goal.yaml --training-profile debug
yolo-agent loop init --run-id exp001 --task task.yaml --data data.yaml --training-config configs/training/yolo26_coco_goal.yaml --training-profile pilot
```

Run pending stages until the next block:

```bash
yolo-agent loop auto --run runs/exp001
```

Initialize and run in one command:

```bash
yolo-agent loop auto --task task.yaml --data data.yaml --components configs/components
```

Individual utilities are also available:

```bash
yolo-agent profile-data --data data.yaml --out runs/dataset_report
yolo-agent advise-labels --data data.yaml --predictions predictions.yaml --out runs/annotation_advice
yolo-agent plan --task task.yaml --components configs/components --out runs/plan.yaml
yolo-agent smoke --plan runs/plan.yaml --data data.yaml
yolo-agent ablate-plan --plan runs/plan.yaml --out runs/ablation_plan.yaml
yolo-agent report --run runs/exp001 --out report.md
```

## Evidence Contract

The harness uses an evidence gate before trusted recommendations. Default loop evidence includes `dataset_report`, `label_quality_report`, `smoke_result`, `latency_ms`, `map50`, and `recall`.

Missing required evidence is written to:

```text
runs/{run_id}/artifacts/evidence_status.json
```

Run-level metrics remain supported through `runs/{run_id}/metrics.json`, but candidate comparisons use node-level evidence:

```text
runs/{run_id}/metrics_by_node.jsonl
```

Smoke guards also write candidate-level records: `smoke_passed`, `yaml_generated`, `ultralytics_imported`, and `forward_checked`.

Each metric record is tied to a concrete candidate and experiment node:

```yaml
candidate_id: baseline
node_id: node_baseline
dataset_version: dataset-v3
split: val
metric_name: map50
value: 0.81
source: benchmark_csv
verified: true
validator: official_eval
source_artifact: runs/exp001/results.csv
metric_schema_version: "1.0"
higher_is_better: true
confidence: 0.99
created_at: "2026-07-02T00:00:00Z"
```

Reports show the warning below and suppress best-model recommendations when the evidence gate is not trusted:

```text
No evidence, do not trust this result.
```

## Policy Boundary

YOLO Agent treats all strategy suggestions as proposals:

```text
PolicyProposal -> LoopPolicyEvaluation -> BudgetAllocation -> CandidateConfig -> ExperimentNode
```

The loop policy evaluator decides priority, deployment blockers, missing evidence, single-variable ablation requirements, round budget fit, deferrals, and human-confirmation requirements.

Round budget is configured in `configs/loop_policy.yaml` under `policy_budget`.

## Key Modules

- `yolo_agent/core/task_spec.py`: task and deployment schema
- `yolo_agent/tools/dataset_stats.py`: YOLO dataset profiling and health score
- `yolo_agent/core/label_quality.py`: label quality signals
- `yolo_agent/agents/annotation_advisor.py`: annotation worklists
- `yolo_agent/agents/error_to_action.py`: detection error taxonomy to actions
- `yolo_agent/agents/optimization_recipe.py`: loss/head/assigner/data-check recipes
- `yolo_agent/agents/sampling_policy.py`: data sampling recommendations
- `yolo_agent/agents/augmentation_policy.py`: data-driven augmentation policy
- `yolo_agent/components/postprocess.py`: post-processing strategy registry
- `yolo_agent/agents/error_driven_loop.py`: diagnosis-to-next-round composition
- `yolo_agent/agents/loop_policy_evaluator.py`: proposal-to-experiment gate
- `yolo_agent/agents/orchestrator.py`: state-machine loop runner
- `yolo_agent/core/evidence_contract.py`: evidence requirements and trust gate
- `yolo_agent/core/evidence_store.py`: local reproducibility store
- `yolo_agent/core/stage_contract.py`: executable stage requirements
- `yolo_agent/core/event_log.py`: append-only loop event audit log

## Non-Goals

The first versions intentionally do not:

- start real training by default
- copy unverified third-party loss implementations
- let LLM output directly decide experiments
- recommend a best model without evidence
- hide missing metrics behind invented values

## Development

```bash
python -m pip install -e ".[dev]"
python -m pytest
```

On Windows in this workspace, use:

```bash
py -3.12 -m pytest
```

Bundled scenario templates live under `configs/scenarios/` and validate against `yolo_agent.core.task_spec.TaskSpec`.
