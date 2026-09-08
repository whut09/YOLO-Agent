# Real Paper Training Cohort Audit

## Scope

This audit is an offline dry-run acceptance of the current production paper
artifacts. It does not start YOLO, probe CUDA, invoke a subprocess, create
metrics, or modify the production readiness report.

The test reads:

- `runs/coverage-audit/paper_execution_inventory.yaml`
- `runs/coverage-audit/paper_execution_requirements.yaml`
- `runs/paper-readiness/paper_asset_registry.yaml`
- `runs/paper-readiness/paper_readiness_report.yaml`

The denominator is 83 compatible paper identities. The acceptance dynamically
selects rows that have a production readiness decision, a concrete adapter,
changed variables, a runtime payload, real available assets, and a training
route. Inference-only routes, shadow-only assignment routes, distillation,
domain adaptation, and hard-negative replay are excluded from this initial
COCO supervised dry-run cohort until their own prerequisites are complete.

## Dry-Run Trainable Fingerprints

These are executable identities constructed from production paper metadata and
validated through a temporary `RoundExecutionPlan` and `ASHAScheduler`. They
are scheduling identities only, not trained results or production evidence.

| Dry-run fingerprint | Paper | Component | Candidate node | Matched control node | First fidelity |
|---|---|---|---|---|---|
| `dcc96972e07a559274e5917dcd55271b54e943cd233068093a942d4c453fc527` | `arxiv:2104.14082` | `loss.quality.pseudo_iou` | `node-paper_dry_run_00_arxiv_2104.14082__pilot_3` | `node-matched-baseline-dry-run-arxiv_2104.14082` | `pilot_3` |
| `30d981bd90f00d23a9baa9fa86974fc5e5d05605757488b09efb755a2b567529` | `arxiv:2301.01019` | `loss.quality.correlation` | `node-paper_dry_run_01_arxiv_2301.01019__pilot_3` | `node-matched-baseline-dry-run-arxiv_2301.01019` | `pilot_3` |
| `dc6343c96e2e3e3be247016696554b7b3b4dd04f792bfc0210a6cf11e603c238` | `arxiv:2303.14404` | `loss.calibration.bpc` | `node-paper_dry_run_02_arxiv_2303.14404__pilot_3` | `node-matched-baseline-dry-run-arxiv_2303.14404` | `pilot_3` |

For every row, the candidate and control use the same `yolo26n` checkpoint,
dataset manifest, COCO split, `imgsz=640`, `pilot_3` fidelity, seed policy,
and protocol hash. The control plan is present before either side has a
completed result.

## Blocker Isolation

- Missing teacher assets block only distillation routes.
- Missing distinct source and target assets block only domain-adaptation
  routes; a single-domain COCO run is not treated as adaptation evidence.
- Missing train-side replay manifests remain evidence-bootstrap work for
  hard-negative replay only.
- Inference-only routes never enter the training ASHA cohort.
- Assignment routes with `mode=shadow` remain shadow evidence until their
  active route is materialized; they are not counted as trained candidates.
- A failure in one temporary candidate would update only that trial and would
  not remove the other paper identities or trials.

## Result Semantics

The dry-run queue contains both a baseline control and a candidate for every
listed fingerprint, and all three fingerprints are registered in the
temporary ASHA study. The dry-run executor returns `dry_run` for both nodes;
it produces no mAP, latency, model-size, or paired result. A paired delta is
therefore empty until both the baseline and candidate have completed verified
observations under the same matched protocol.

The persisted production report remains unchanged and continues to report
`actual_trained_count=0` for this no-training audit. The three fingerprints
above are not a claim that the corresponding papers have improved mAP.
