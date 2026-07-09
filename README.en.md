# yolo-agent

[Chinese](README.md) | English

YOLO Agent is an evidence-driven YOLO optimization and training harness.

It is not a free-form code-generation agent, and it does not blindly modify model code. It runs a controlled, resumable loop:

```text
preflight -> debug training -> evidence import -> error diagnosis -> next optimization step
```

## Who It Is For

- Users who want automated YOLO training and optimization on COCO or custom YOLO datasets
- Teams who want queue, resume, evidence, and reports instead of manual experiment sprawl
- Engineers who need to compare accuracy, latency, model size, and robustness

## What It Can Do Today

- Check Python, CUDA, Ultralytics, COCO paths, disk space, and run-directory permissions
- Start COCO + YOLO26 debug training and automatically continue to pilot after debug succeeds
- Create run context, dataset manifest, experiment plan, execution queue, and report
- Import `results.csv`, `best.pt`, `args.yaml`, runtime profile, and COCO error facts
- Guard large runs with evidence gates, full-run confirmation, and timeouts

## Install

### Windows / PowerShell

```powershell
cd E:\codex\YOLO-Agent
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -e ".[dev]"
```

### Optional Training Dependencies

```powershell
python -m pip install -e ".[train]"
```

If Ultralytics is already installed, verify it with:

```powershell
python -c "import ultralytics; print(ultralytics.__version__)"
```

### Verify the CLI

```powershell
yolo-agent --help
yolo-agent doctor --data E:\datatset\coco.yaml --model yolo26n.pt
```

## 30-Second Start: COCO + YOLO26

1. Check the environment:

```powershell
yolo-agent doctor --data E:\datatset\coco.yaml --model yolo26n.pt
```

2. Start automated optimization training. It runs debug first and automatically continues to pilot after debug succeeds:

```powershell
yolo-agent train --model yolo26n.pt --data E:\datatset\coco.yaml --goal +2map --run-id coco-yolo26n
```

3. Watch status and next-step guidance:

```powershell
yolo-agent status --run runs/coco-yolo26n
```

4. Full COCO requires a second confirmation:

```powershell
yolo-agent train --model yolo26n.pt --data E:\datatset\coco.yaml --run-id coco-yolo26n --profile baseline_full --confirm-full-run
```

## What These Modes Mean

```text
dry-run = preview only, no training; train runs by default unless --dry-run is passed
debug = tiny real training to verify the pipeline
pilot = small training to see whether the direction is promising
full COCO = full training budget for trusted evidence
```

`train` starts real training by default. Add `--dry-run` when you only want to preview the plan. A successful `debug` run automatically continues to `pilot`, then runs bounded pilot-only optimization rounds by default (30 rounds); before full COCO it stops and requires `--confirm-full-run`. Use `--auto-rounds 0` to stop after pilot.

Detailed Chinese guide: [运行模式说明](docs/training-modes.md).

## Custom YOLO Dataset

```powershell
yolo-agent train --kind custom --model yolo26n.pt --data path\to\data.yaml --run-id custom-yolo26n --profile debug
```

The input must be a standard YOLO `data.yaml`. Start with `debug` to verify paths, classes, and the minimum training flow before moving to `pilot` or a full profile.

## Safety Boundaries

- `train` runs by default; add `--dry-run` for preview-only mode
- `debug` is a small-fraction sanity run; after debug succeeds, the default flow automatically continues to `pilot`
- Add `--no-auto-advance` when you want to stop after the requested profile
- `baseline_full`, `baseline_confirm`, and `candidate_full` also require `--confirm-full-run`
- debug timeout defaults to 3600 seconds; pilot timeout defaults to 43200 seconds
- Auto-advance is bounded: `debug -> pilot -> pilot-only optimization rounds (30 by default)`; full profiles require explicit confirmation, so the loop will not run forever

## Common Commands

```powershell
yolo-agent doctor --data E:\datatset\coco.yaml --model yolo26n.pt
yolo-agent train --model yolo26n.pt --data E:\datatset\coco.yaml --run-id coco-yolo26n
yolo-agent status --run runs/coco-yolo26n
yolo-agent stop --run runs/coco-yolo26n
```

## Documentation

- [Install](docs/install.md)
- [Quickstart](docs/quickstart.md)
- [Training Modes](docs/training-modes.md)
- [COCO + YOLO26 Runbook](docs/coco-yolo26.md)
- [Custom Dataset](docs/custom-dataset.md)
- [Concepts](docs/concepts.md)
- [Loop Engineering](docs/loop-engineering.md)
- [Evidence and Reports](docs/evidence.md)
- [CLI Reference](docs/cli.md)
- [Troubleshooting](docs/troubleshooting.md)

## Development

```powershell
python -m pip install -e ".[dev]"
py -3.12 -m pytest
```

## Project Positioning

YOLO Agent is a componentized object-detection optimization harness, not a free-form code-generation agent.
