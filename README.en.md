# YOLO Agent

[中文](README.md) | English

YOLO Agent is an evidence-driven automatic optimization trainer for YOLO object detection. It connects training, COCO evaluation, error diagnosis, recipe proposals, budget-aware elimination, and reporting in a recoverable and auditable loop. An LLM may propose actions, but it cannot bypass compatibility, evidence, budget, ASHA, or full-run consent gates.

## Install

Python 3.12 and an isolated virtual environment are recommended:

```powershell
cd E:\codex\YOLO-Agent
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -e ".[train]"
```

## Start In Three Steps

New users only need four commands: `setup`, `train`, `status`, and `stop`.

1. Check the environment and create local configuration:

```powershell
yolo-agent setup coco --data E:\datatset\coco.yaml --model yolo26n.pt
```

2. Start automatic training and pilot optimization:

```powershell
yolo-agent train --model yolo26n.pt --data E:\datatset\coco.yaml --run-id coco-yolo26n
```

3. View the aggregated status:

```powershell
yolo-agent status --run runs/coco-yolo26n
```

Stop the run when needed:

```powershell
yolo-agent stop --run runs/coco-yolo26n
```

## What Automatic Optimization Does

The default flow reads trusted baseline and pilot evidence, runs fixed-protocol evaluation and error-fact extraction, and combines rules, policy memory, and a frozen paper snapshot to propose candidates. Deterministic gates then enforce component maturity, YOLO26 compatibility, `imgsz=640`, matched controls, and budget limits. Eligible candidates enter an ASHA-managed pilot queue; their paired metric deltas, latency, and model-size evidence drive elimination, evidence recovery, or continuation.

The default flow never increases image size automatically and does not start full COCO by default. Full candidates require explicit confirmation. `+2 mAP` is an optimization objective, not a guaranteed outcome.

## What The Paper Catalog Is

The project can import [Awesome-object-detection](https://github.com/whut09/Awesome-object-detection) offline and build a frozen `ResearchSnapshot` before training. Training reads only that snapshot and never fetches papers from the network.

The paper catalog is not a training dataset, and paper metrics are not local evidence. Paper records provide diagnostic and recipe priors only:

- `recipe_idea_only` is not an executable recipe.
- A paper record does not imply that an adapter exists.
- An adapter does not imply that smoke tests passed.
- Smoke-passed does not imply pilot-reproduced.
- Pilot-reproduced does not imply full-COCO-confirmed.

## Capabilities Requiring Local Certification

Real adapters require construction, shape, backward, AMP, and smoke tests. Candidate gains also require matched pilots, complete COCO evidence, latency/model-size guards, and multi-seed confirmation. Only local certification artifacts can advance reproduction maturity; paper claims, code presence, or a single-seed gain cannot substitute for certification.

<!-- capability-maturity:start -->
| Capability | Current status | Code present | Automatic execution | Local reproduction | Boundary |
| --- | --- | --- | --- | --- | --- |
| Automatic pilot training | `executable` | yes | yes | depends on local runs | The default training entrypoint can execute debug and pilot runs; success depends on the local environment and data. |
| Automatic basic metric import | `executable` | yes | yes | depends on local runs | Imports results.csv, training artifacts, and basic runtime evidence; missing artifacts still produce an evidence gap. |
| Candidate COCO error facts | `incomplete` | yes | partial | partial | Post-eval, import, and completeness gates exist, but every candidate is not yet guaranteed to produce predictions.json and complete per-class/FN/FP/localization facts. |
| Error-delta next-round decisions | `partial` | yes | partial | partial | Compares parent/current error facts and constrains proposals; incomplete candidate facts fall back to evidence collection or rules. |
| ASHA / successive-halving queue control | `executable` | yes | guarded | not claimed | ASHA assignments feed the authoritative RoundExecutionPlan and queue; full rungs still require explicit confirmation and are not automatic by default. |
| Paper component adapters | `mixed` | yes | mixed | mixed | The registry mixes metadata-only entries, implemented adapters, and executable components; maturity must be checked per component. |
| Three-seed confirmation | `supported, not automatic end-to-end` | yes | explicit confirmation | not claimed | The scheduler and confidence gates support three seeds; candidate_full requires explicit confirmation and the default pilot loop does not run all seeds automatically. |
| Stable +2 mAP improvement | `not guaranteed` | no | no | not claimed | +2 mAP is an objective and acceptance condition, not a project guarantee; it requires a matched baseline, full COCO, three seeds, and confidence intervals. |
<!-- capability-maturity:end -->

## Read Next

- [Commands and advanced entry points](docs/cli.md)
- [Awesome-object-detection offline integration](docs/awesome-object-detection.md)
- [Paper Intelligence](docs/paper-intelligence.md)
- [Capability maturity](docs/capability-maturity.md)
- [GPU Certification](docs/gpu-certification.md)
