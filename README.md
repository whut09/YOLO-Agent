# YOLO Agent

English | [简体中文](README.zh-CN.md)

YOLO Agent is an evidence-driven optimization runner for YOLO object detection. It connects training, COCO evaluation, error diagnosis, paper-informed recipes, matched comparisons, budget control, and reporting in a recoverable workflow.

LLMs may analyze evidence and propose recipes, but deterministic gates control compatibility, experiment budgets, promotion, and full-run consent.

![YOLO Agent architecture](docs/assets/yolo-agent-architecture.svg)

> All shell examples below are single-line PowerShell commands. PowerShell does not support Bash `\` line continuations; multi-line examples are only shown in the docs and are always labeled `bash`.

## Core value

You do not need to choose an optimizer, loss, neck, sampling strategy, or paper method. Give YOLO Agent a model and annotated data, then describe the problem in one sentence. The agent builds a baseline, analyzes COCO errors, selects eligible local or paper-informed recipes, runs matched pilots, eliminates weak candidates with ASHA, and reports what actually changed.

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

The sentence guides diagnosis and recipe selection; the metric and delta define the deterministic acceptance target. If no explicit target is supplied, the executable objective defaults to `+2map`. Automatic optimization means automated diagnosis and bounded, evidence-based experiments; it does not guarantee that every dataset or run will improve mAP.

- One command starts environment checks, debug training, automatic mini-GPU safety certification when needed, and bounded pilot optimization.
- Candidate decisions use matched controls, local evidence, latency, and model-size guards.
- ASHA manages pilot budgets and eliminates weak candidates early.
- Paper Intelligence imports catalogs offline into a frozen `ResearchSnapshot`.
- Component maturity prevents metadata-only or unverified adapters from entering training.
- Paper recipes require a hash-bound runtime adapter and matched control before ASHA can allocate a pilot.
- Every run writes auditable plans, events, evidence, queue state, and reports.

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

The default budget is automatic and pilot-only; full COCO training requires explicit confirmation. After fixing the environment, rerun the same `train` command. See [Quick start](docs/quickstart.md) for the full walkthrough and [CLI and advanced commands](docs/cli.md) for every flag.

## High-level loop

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

## Current verified status

Status is measured in four independent layers. Every number below is read from a machine-readable artifact or config, not hand-maintained; dates and hashes live in the artifacts.

### Research coverage

A large paper catalog does not mean that every paper can be applied to YOLO26; implemented adapters, executable recipes, and reproduced papers are different measurements. Each automatic round reports the full search funnel in `artifacts/executable_portfolio.yaml` (created by the first automatic round). See [Paper Intelligence](docs/paper-intelligence.md).

<!-- paper-adapter-coverage:start -->
| Frozen paper records | Implemented component IDs | Unique Python adapter classes | Source runtime components | Pilot reproduced components |
| --- | --- | --- | --- | --- |
| 728 | 158 | 103 | 0 | 0 |

The 158 value counts component IDs backed by 103 distinct Python adapter classes; neither value counts reproduced papers. Artifact-backed machine maturity is reported separately in the acceptance table below.
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

### Implementation readiness (Paper-83 campaign)

Source: `artifacts/pretraining_acceptance.yaml` (evaluated 2026-09-22); manifest: `configs/research/paper_83_manifest.yaml`.

| Gate | Result |
| --- | --- |
| Papers in campaign manifest | 83 |
| Specification complete | 83/83 |
| Code-bound implementation | 83/83 |
| Runtime integrated | 83/83 |
| Unit tested | 83/83 |
| Non-mock smoke passed | 83/83 |
| Compatibility validated | 83/83 |
| Implementation ready | 83/83 (blocked: 0) |

### Training readiness

Sources: `artifacts/paper_83_runtime_preflight.yaml` (2026-09-21), `artifacts/pretraining_acceptance.yaml` (2026-09-22), `artifacts/training_release_v1.yaml` (2026-09-22).

| Gate | Result |
| --- | --- |
| Runtime preflight | 83/83 passed; unknown runtime hooks: 0 |
| Pretraining acceptance | all gates PASS |
| Training release | `READY_FOR_FIRST_TRAINING`, frozen at git commit `3228dcc6` |

`real_training_executed` is `false` in every artifact: these gates certify readiness, not training results.

### Reproduction evidence

Sources: `docs/paper-adapter-coverage.yaml` (`pilot_reproduced_count`, `maturity_counts`), `docs/paper-coverage-acceptance.yaml` (`exact_reproduction_paper_ids`).

| Evidence level | Count |
| --- | --- |
| Pilot reproduced components | 0 |
| Exact paper reproduction (full) | 0 |
| Multi-seed confirmed | 0 |

Implemented is not reproduced. A single pilot improvement is `possible`, not `confirmed`; paper metrics never count as promotion evidence. As of this snapshot, nothing has been reproduced locally.

## Capability boundaries

<!-- capability-maturity:start -->
| Capability | Current status | Code present | Automatic execution | Local reproduction | Boundary |
| --- | --- | --- | --- | --- | --- |
| Automatic pilot training | `executable` | yes | yes | depends on local runs | The default training entrypoint can execute debug and pilot runs; success depends on local environment, data, and evidence gates. |
| Automatic basic metric import | `executable` | yes | yes | depends on local runs | Imports results.csv, training artifacts, and basic runtime evidence; missing artifacts still produce an evidence gap. |
| Candidate COCO error facts | `incomplete` | yes | partial | partial | Post-eval, import, and completeness gates exist; each real dataset run must still verify complete per-class/FN/FP/localization facts. |
| Error-delta next-round decisions | `partial` | yes | partial | partial | Compares parent/current error facts and constrains proposals; incomplete evidence permits evidence recovery only. |
| ASHA / successive-halving queue control | `executable` | yes | guarded | not claimed | ASHA is the training budget authority; full rungs still require explicit confirmation and are not automatic by default. |
| Paper component adapters | `mixed` | yes | guarded | not claimed | Certified components may enter pilots through MethodProfile, maturity, matched-control, and ASHA gates; smoke passed is not pilot reproduced. |
| Three-seed confirmation | `executable` | yes | explicit confirmation | not claimed | The three-seed confirmation workflow runs end-to-end: pilot winner -> explicit full-run approval (nothing executes without it) -> matched 3-seed runs -> paired 95% CI -> four-way verdict -> confirmation report; interrupted confirmations resume without rerunning finished seeds and approvals are single-use. |
| Stable +2 mAP improvement | `not guaranteed` | no | no | not claimed | +2 mAP is an objective, not a project guarantee; it requires a matched baseline, full COCO, three seeds, and confidence intervals. |
<!-- capability-maturity:end -->

## Documentation map

- [System architecture (canonical)](docs/architecture.md)
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

`pytest -q` is the fast regression suite and leaves large CPU/mock integration tests deselected. Run `pytest -q --run-slow` for the complete non-GPU suite. Real CUDA tests still require the separate explicit `--run-real-gpu` opt-in. Small CPU tensor tests default to one Torch thread; set `YOLO_AGENT_TEST_TORCH_THREADS` to a higher value for local benchmarking.

The project is licensed under the MIT License.
