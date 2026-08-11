# Automatic Optimization Architecture

## What the system does

YOLO Agent is a diagnosis-driven experiment manager around the Ultralytics
training runtime. A user can write one problem statement, for example:

```text
Small objects are missed too often; improve AP_small without a large latency or
overall mAP regression.
```

The agent builds a baseline, imports or computes error facts, maps the facts to
canonical mechanisms, materializes compatible recipes, runs matched
candidate/control pilots, and records paired deltas. It does not promise that
an arbitrary paper will improve an arbitrary dataset. The output is a measured
decision, including rejection and evidence-recovery decisions.

## Six coverage numbers

The repository separates these quantities:

1. Catalog papers: papers known to the offline catalog.
2. Method profiles: papers with local evidence describing their method.
3. Recipe definitions: reusable mechanism recipes, including paper priors.
4. Runtime-ready recipes: recipes whose components have frozen implementation
   identity and valid non-mock maturity artifacts.
5. Diagnosis-matched recipes: recipes bound to the current run's error facts.
6. Executable trials: candidate/control experiments actually registered with
   ASHA.

The first three are knowledge assets. Only the last three describe what can
participate in the current training run. Each automatic round writes
`artifacts/executable_portfolio.yaml` and includes this funnel in the terminal.

## Runtime authority

Paper names and titles are priors. Training authorization comes from the frozen
research snapshot and its effective maturity manifest:

```text
PaperMethodProfile -> canonical mechanism -> recipe -> adapter contract
  -> adapter hash + Ultralytics version + protocol hash
  -> non-mock runtime/unit/smoke evidence -> matched pilot/control
```

At train startup, the CLI discovers all reusable training adapters rather than
only a fixed shortlist. It refreshes missing or stale CPU evidence automatically.
A failed adapter is isolated and reported; it cannot silently become ordinary
Ultralytics training. Snapshot identity is checked again before materialization.

Reviewed local recipe families are merged by the research production pipeline
before the snapshot is hashed. Training reads only the frozen recipe registry;
it never fills gaps from a live paper or recipe registry during a run. Source
paths and parse errors are recorded in the executable portfolio artifact.

## Why low-fidelity pilots can regress

Three-epoch, ten-percent pilots are screening measurements, not final claims.
The default ASHA policy treats small pilot deltas inside a `-0.0015` noise band
as rankable cohort evidence. Severe regressions, invalid paired evidence,
diagnosis failures, and resource-guard failures are still eliminated early.

`pilot_10` still requires a verified positive paired delta and target error-fact
improvement. Full training and confirmation seeds remain explicit
`--confirm-full-run` operations. A pilot signal is never reported as exact paper
reproduction.

## Lessons from current tooling

The public Ultralytics workflow emphasizes a simple `model.train(data=...,
epochs=..., imgsz=640)` entry point, AMP, multi-GPU, checkpoint/resume, and
validation. Roboflow-style dataset workflows emphasize dataset health, class
balance, preprocessing/augmentation, and repeatable evaluation. FiftyOne-style
error analysis emphasizes inspecting hard samples and failure slices. ClearML,
Weights & Biases, and Ray Tune demonstrate the value of immutable experiment
records, tracking, and bounded schedulers such as ASHA. NVIDIA TAO and
deployment-oriented AutoML systems add resource, export, and latency constraints.

YOLO Agent combines these ideas around a local, reproducible contract: dataset
facts and error slices choose the mechanism family; the adapter controls the
framework hook; matched controls measure the effect; ASHA controls cost; and
PolicyMemory supplies a soft prior for future runs. It does not claim to replace
those products or reproduce proprietary internals.

## Mechanism families

The planning taxonomy covers YOLO26-compatible small-object and class-balanced
sampling, hard-negative replay, scale-aware augmentation, quality alignment and
calibration losses, teacher/student distillation, multi-scale fusion and P2
heads, guarded assignment variants, and isolated inference policies such as
slicing and calibration. Detector-family changes, DFL-dependent methods,
transformer query methods, and unsupported operators remain separate or
`implementation_request` entries.
