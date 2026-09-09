# Real Paper Training Cohort Audit

## Scope

This audit is an offline preparation of the current production paper
artifacts. It does not start YOLO, probe CUDA, invoke a subprocess, or create
metrics. The prepared run stores scheduling artifacts only.

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
validated through the prepared `RoundExecutionPlan` and `ASHAScheduler`. They
are scheduling identities only, not trained results or production evidence.

| Dry-run fingerprint | Paper | Component | Candidate node | Matched control node | First fidelity |
|---|---|---|---|---|---|
| `41571244fbd32d57f0ac08eb69f2a1a7d947d5f6d667a797fb61b18d61dca9ac` | `arxiv:2104.14082` | `loss.quality.pseudo_iou` | `node_paper_cohort_01_arxiv_2104.14082__pilot_3` | `node_matched_baseline_01_arxiv_2104.14082__pilot_3` | `pilot_3` |
| `849bdcd9f8d956f539c93489119ad30ede1e4d8732712cd3155899fabf0a7455` | `arxiv:2303.14404` | `loss.calibration.bpc` | `node_paper_cohort_02_arxiv_2303.14404__pilot_3` | `node_matched_baseline_02_arxiv_2303.14404__pilot_3` | `pilot_3` |
| `d5f2d43200b0f3b4c62f7b9d20815d9110a383bb022405da5e34d97416a6fbc9` | `arxiv:2301.01019` | `loss.quality.correlation` | `node_paper_cohort_03_arxiv_2301.01019__pilot_3` | `node_matched_baseline_03_arxiv_2301.01019__pilot_3` | `pilot_3` |

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

The prepared queue contains both a baseline control and a candidate for every
listed fingerprint, and all three fingerprints are registered in the ASHA
study. The queue is still `queued`; no executor has run. It therefore produces
no mAP, latency, model-size, or paired result. A paired delta remains empty
until both the baseline and candidate have completed verified observations
under the same matched protocol.

The final readiness report records `training_allowed=true` and
`actual_trained_count=0`. The three fingerprints above are scheduling
identities only and are not a claim that the corresponding papers have
improved mAP.
