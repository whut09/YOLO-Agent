# yolo-agent

YOLO Agent is a componentized object-detection optimization harness, not a free-form code-generation agent.

The project is intended to help choose YOLO model sizes, network components, losses, training strategies, data policies, post-processing policies, and reproducible experiment plans from task, dataset, detection-error, and deployment constraints.

The harness is evidence-driven. It does not blindly generate model code or start training. It runs a controlled loop:

```text
task + data + detection errors + deployment constraints
        -> diagnosis
        -> action policy
        -> candidate policies
        -> guarded candidates
        -> smoke/evidence
        -> next round
```

## CLI

```bash
yolo-agent --help
yolo-agent init
yolo-agent init --scenario infrared_small_target --output task.yaml
yolo-agent profile-data
yolo-agent plan --task task.yaml --components configs/components --out runs/plan.yaml
yolo-agent check
yolo-agent smoke
yolo-agent search
yolo-agent ablate
yolo-agent benchmark
yolo-agent report
yolo-agent loop init --run-id exp001 --task task.yaml --data data.yaml
yolo-agent loop run-stage --run runs/exp001 --stage profile_data
yolo-agent loop auto --run runs/exp001
yolo-agent loop --run runs/exp001 --resume
```

## Loop Harness

The loop orchestrator is a state machine, not a script chain. It persists:

- `runs/{run_id}/run_context.yaml`
- `runs/{run_id}/loop_state.yaml`
- `runs/{run_id}/artifacts/`

Default stage order is defined in `configs/loop_policy.yaml`:

```text
init -> profile_data -> advise_labels -> diagnose_errors -> generate_loop_plan
-> evaluate_policies -> generate_candidates -> ablate -> smoke
-> import_metrics -> report -> next_round
```

Stages with missing required evidence become `blocked` so the run can be resumed instead of silently producing untrusted recommendations.

## Evidence Contract

The harness uses an evidence gate before trusted recommendations. Default loop evidence includes:

- `dataset_report`
- `label_quality_report`
- `smoke_result`
- `latency_ms`
- `map50`
- `recall`

Missing required evidence is written to `runs/{run_id}/artifacts/evidence_status.json`. Reports show `No evidence, do not trust this result.` and suppress best-model recommendations when the gate is not trusted.

## Development

```bash
python -m pip install -e ".[dev]"
python -m pytest
```

Bundled scenario templates live under `configs/scenarios/` and validate against `yolo_agent.core.task_spec.TaskSpec`.
