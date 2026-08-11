# YOLO Agent

English | [简体中文](README.zh-CN.md)

YOLO Agent is an evidence-driven optimization runner for YOLO object detection. It connects training, COCO evaluation, error diagnosis, paper-informed recipes, matched comparisons, budget control, and reporting in a recoverable workflow.

LLMs may analyze evidence and propose recipes, but deterministic gates control compatibility, experiment budgets, promotion, and full-run consent.

![YOLO Agent architecture](docs/assets/yolo-agent-architecture.svg)

## Tell Agent What's Wrong

You do not need to choose an optimizer, loss, neck, sampling strategy, or paper method. Give YOLO Agent a model and annotated data, then describe the problem in one sentence. The agent builds a baseline, analyzes COCO errors, selects eligible local or paper-informed recipes, runs matched pilots, eliminates weak candidates with ASHA, and reports what actually changed.

Replace the example model and data paths below with files that exist on your machine.

```powershell
# Too many false positives
yolo-agent train --model yolo26n.pt --data E:\dataset\coco.yaml --run-id reduce-fp --target-metric precision --target-delta 0.02 --goal-description "The current model has too many false positives, especially high-confidence ones"

# Performance dropped in a new scene
yolo-agent train --model yolo26n.pt --data E:\dataset\new-scene.yaml --run-id adapt-scene --target-metric map50_95 --target-delta 0.02 --goal-description "Performance dropped after moving to a new scene; diagnose the domain shift and optimize it"

# Small objects are often missed
yolo-agent train --model yolo26n.pt --data E:\dataset\coco.yaml --run-id improve-small --target-metric ap_small --target-delta 0.02 --goal-description "Small-object detection is weak; improve AP_small and reduce false negatives"

# Improve overall mAP
yolo-agent train --model yolo26n.pt --data E:\dataset\coco.yaml --run-id improve-map --goal +2map --goal-description "Improve overall mAP while controlling latency and model size"
```

The sentence guides diagnosis and recipe selection; the metric and delta define the deterministic acceptance target. If no explicit target is supplied, the executable objective defaults to `+2map`. Scene-shift optimization requires representative labeled train/validation data from the new scene. Automatic optimization means automated diagnosis and bounded, evidence-based experiments; it does not guarantee that every dataset or run will improve mAP.

### What is actually searchable

The paper catalog, MethodProfiles, recipe definitions, runtime adapters, and
completed experiments are different measurements. A large paper catalog does not
mean that every paper can be applied to YOLO26. Each automatic round reports the
full search funnel in `artifacts/executable_portfolio.yaml`: catalog papers,
profiles, recipe definitions, frozen runtime-ready recipes, recipes matched to the
current diagnosis, critic-approved recipes, and candidates actually entered into
the ASHA queue.

The command automatically discovers and prepares reusable training adapters. You
do not need to run a separate certification command for ordinary training.
Inference-only policies, unsupported detector-family changes, and methods without
enough local evidence remain separate and are never silently treated as training
recipes.

## Highlights

- One command starts environment checks, debug training, automatic mini-GPU safety certification when needed, and bounded pilot optimization.
- Candidate decisions use matched controls, local evidence, latency, and model-size guards.
- ASHA manages pilot budgets and eliminates weak candidates early.
- Paper Intelligence imports catalogs offline into a frozen `ResearchSnapshot`.
- Component maturity prevents metadata-only or unverified adapters from entering training.
- Paper recipes require a hash-bound runtime adapter and matched control before ASHA can allocate a pilot.
- Every run writes auditable plans, events, evidence, queue state, and reports.

## Install

Python 3.12 and an isolated environment are recommended.

```powershell
git clone https://github.com/whut09/YOLO-Agent.git
cd YOLO-Agent
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -e ".[train]"
```

The `train` extra includes the fixed-protocol COCO evaluator required by automatic
post-evaluation. The separate `certification` extra is only needed by maintainers
running standalone advanced certification commands.

## Quick Start

New users only need four commands.

```powershell
# 1. Validate the environment
yolo-agent setup coco --data E:\dataset\coco.yaml --model yolo26n.pt

# 2. Start bounded automatic pilot optimization
yolo-agent train --model yolo26n.pt --data E:\dataset\coco.yaml --run-id coco-yolo26n

# 3. Inspect the parent run and active child run
yolo-agent status --run runs/coco-yolo26n

# 4. Stop safely
yolo-agent stop --run runs/coco-yolo26n
```

The default budget is automatic and pilot-only. Full COCO training requires explicit confirmation.
Real training preflights a frozen research snapshot before allocating the run; stale paper or adapter maturity state must be rebuilt offline first.
If the local mini-GPU acceptance artifact is missing or stale, `train` rebuilds it
once and continues automatically. A failed safety check stops candidate training and
reports the cause; after fixing the environment, rerun the same `train` command.

`--goal` accepts structured expressions such as `+2map`. Keep natural-language intent
separate when targeting a diagnostic metric:

```powershell
yolo-agent train --model yolo26n.pt --data E:\dataset\coco.yaml --run-id coco-small --target-metric ap_small --target-delta 0.02 --goal-description "Reduce small-object false negatives"
```

## Decision Workflow

```text
trusted baseline and current evidence
-> COCO error facts and diagnosis
-> paper and local recipe proposals
-> compatibility, maturity, and evidence gates
-> matched pilot cohort
-> post-evaluation and paired delta
-> ASHA elimination or promotion
-> report and policy-memory update
```

If required evidence is incomplete, the queue requests evidence recovery instead of promoting another training run. Training keeps `imgsz=640` for YOLO26 comparisons and does not increase it automatically.

## Paper Intelligence

YOLO Agent can import [Awesome-object-detection](https://github.com/whut09/Awesome-object-detection) before training and build a frozen research snapshot. Training never fetches papers from the network.

Paper records are priors, not local results:

- A paper entry does not mean an adapter exists.
- An implemented adapter is not executable until its runtime path and smoke tests pass.
- A single pilot improvement is `possible`, not `confirmed`.
- Paper metrics never count as promotion evidence.

<!-- paper-adapter-coverage:start -->
| Frozen paper records | Implemented component IDs | Unique Python adapter classes | Source runtime components | Pilot reproduced components |
| --- | --- | --- | --- | --- |
| 728 | 55 | 31 | 0 | 0 |

The 55 value counts component IDs backed by 31 distinct Python adapter classes; neither value counts reproduced papers. Artifact-backed machine maturity is reported separately in the acceptance table below.
Audit snapshot: `c606d6c50fefaa7ae0db8bddb39d62057ff09ed5aeae943c81c990971b353e57`.

| Artifact acceptance | Result | Target |
| --- | --- | --- |
| Compatible papers with valid MethodProfile | 85/85 (100.0%) | >=85% |
| Compatible mechanisms with reusable adapter | 20/23 (87.0%) | >=80% |
| Compatible mechanisms runtime integrated | 18/23 (78.3%) | >=70% |
| Compatible mechanisms smoke passed | 18/23 (78.3%) | >=60% |
| Compatible papers reusing a certified adapter | 83/85 (97.6%) | >=70% |

Catalog-wide certified-adapter mapping is 83/728 (11.4%); this is reusable component adaptation, not exact paper reproduction.

Exact reproduction is reported separately: 0; separate detector family: 168; insufficient information: 475.
Acceptance hash: `797c3b912852717b03e3ce7fc55a3650d8b028f7d1dc9fc2a827c65c5996667c`.
<!-- paper-adapter-coverage:end -->

## Capability Boundaries

<!-- capability-maturity:start -->
| Capability | Current status | Code present | Automatic execution | Local reproduction | Boundary |
| --- | --- | --- | --- | --- | --- |
| Automatic pilot training | `executable` | yes | yes | depends on local runs | The default training entrypoint can execute debug and pilot runs; success depends on local environment, data, and evidence gates. |
| Automatic basic metric import | `executable` | yes | yes | depends on local runs | Imports results.csv, training artifacts, and basic runtime evidence; missing artifacts still produce an evidence gap. |
| Candidate COCO error facts | `incomplete` | yes | partial | partial | Post-eval, import, and completeness gates exist; each real dataset run must still verify complete per-class/FN/FP/localization facts. |
| Error-delta next-round decisions | `partial` | yes | partial | partial | Compares parent/current error facts and constrains proposals; incomplete evidence permits evidence recovery only. |
| ASHA / successive-halving queue control | `executable` | yes | guarded | not claimed | ASHA is the training budget authority; full rungs still require explicit confirmation and are not automatic by default. |
| Paper component adapters | `mixed` | yes | guarded | not claimed | Certified components may enter pilots through MethodProfile, maturity, matched-control, and ASHA gates; smoke passed is not pilot reproduced. |
| Three-seed confirmation | `supported, not automatic end-to-end` | yes | explicit confirmation | not claimed | The scheduler and confidence gates support three seeds; candidate_full requires explicit full-run confirmation. |
| Stable +2 mAP improvement | `not guaranteed` | no | no | not claimed | +2 mAP is an objective, not a project guarantee; it requires a matched baseline, full COCO, three seeds, and confidence intervals. |
<!-- capability-maturity:end -->

## Documentation

- [Quick start](docs/quickstart.md)
- [Installation](docs/install.md)
- [CLI and advanced commands](docs/cli.md)
- [Training modes](docs/training-modes.md)
- [Automatic optimization architecture](docs/automatic-optimization-architecture.md)
- [COCO and YOLO26](docs/coco-yolo26.md)
- [Custom datasets](docs/custom-dataset.md)
- [LLM setup](docs/llm-setup.md)
- [Evidence model](docs/evidence.md)
- [Paper Intelligence](docs/paper-intelligence.md)
- [Paper adapter implementation queue](docs/paper-adapter-implementation-queue.md)
- [Paper recipe materialization](docs/paper-recipe-materialization.md)
- [Distillation mechanisms](docs/distillation-mechanisms.md)
- [YOLO26 graph components](docs/yolo26-graph-components.md)
- [Awesome-object-detection integration](docs/awesome-object-detection.md)
- [Capability maturity](docs/capability-maturity.md)
- [GPU certification](docs/gpu-certification.md)
- [SAHI inference certification](docs/sahi-inference-certification.md)
- [Isolated inference policy adapters](docs/inference-policy-adapters.md)
- [Troubleshooting](docs/troubleshooting.md)

## Development

```powershell
python -m pip install -e ".[dev]"
pytest -q
ruff check .
```

The project is licensed under the MIT License.
