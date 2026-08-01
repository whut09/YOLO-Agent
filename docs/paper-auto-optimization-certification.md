# Paper Auto-Optimization Acceptance

This opt-in suite certifies the first complete paper-driven optimization path:

```text
fresh ResearchSnapshot -> diagnosis -> PaperMethodProfile
-> gpu_certified sampling.small_object adapter
-> matched pilot_3 control and candidate -> fixed COCO post-eval
-> complete current-node error facts -> paired bootstrap delta
-> ASHA -> matched pilot_10 -> PolicyMemory -> pilot_reproduced
```

The suite is offline during training and fixes `imgsz=640`. Scalar HPO is disabled.
Paper claims remain priors and never become candidate observations.

## Prerequisites

Install the training and certification dependencies and certify the runtime adapter:

```powershell
python -m pip install -e ".[train,certification]"
yolo-agent advanced certify-component `
  --component sampling.small_object `
  --gpu `
  --model E:\path\yolo26n.pt `
  --device 0
```

The maturity registry must contain a valid `gpu_certified` overlay for the current
adapter hash and installed Ultralytics version. The Awesome-object-detection source
must be a local checkout or exported `papers.json`; the suite does not access the
network.

## Run

```powershell
yolo-agent advanced certify-paper-auto `
  --workdir runs/certification/paper-auto `
  --research-root research `
  --source E:\path\Awesome-object-detection `
  --registry runs/component_maturity_registry.yaml `
  --policy-root runs `
  --model E:\path\yolo26n.pt `
  --device 0 `
  --execute-real-gpu
```

Without `--execute-real-gpu`, the command writes a skipped report and does not build
a snapshot or start training.

The terminal prints the selected paper IDs, component ID, adapter hash, maturity,
matched baseline and candidate IDs, AP_small/target recall/FN paired deltas, resource
guards, and the exact recovery or elimination reason. The machine-readable report is:

```text
runs/certification/paper-auto/paper_auto_optimization_report.yaml
```

## Failure Semantics

- A candidate/control mismatch in dataset manifest, subset, seed, epochs, batch
  policy, Ultralytics version, evaluation protocol, objective, or protocol hash fails
  before post-evaluation.
- Missing predictions, COCO evaluation, error facts, or paired bootstrap produces an
  `evidence_recovery` result. No pilot_10 assignment is consumed.
- AP_small, target-class recall, and false-negative count are primary promotion
  conditions. Overall mAP, latency, and model size remain hard guards.
- Only ASHA can issue the pilot_10 assignment. Policy YAML and scalar HPO cannot add
  training work.
- Failed artifacts are retained and do not advance component maturity.

A passed report appends the verified local result to PolicyMemory and advances the
exact adapter identity to `pilot_reproduced`. It does not claim full reproduction,
multi-seed confirmation, or a guaranteed metric gain. Full training and seeds 2/3
remain behind explicit `--confirm-full-run` consent.

## Tests

The default offline suite uses a mock GPU backend and never starts CUDA training:

```powershell
python -m pytest tests/test_paper_auto_optimization_acceptance.py -q
```

Real GPU execution occurs only through the explicit CLI command above.
