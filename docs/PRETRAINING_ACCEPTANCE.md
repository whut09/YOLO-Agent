# Pre-Training Acceptance

```text
========================================
YOLO AGENT PRE-TRAINING ACCEPTANCE
========================================
Frozen papers:              83
Implementation ready:       83/83
Blocked:                    0

Paper runtime integrity:    PASS
Non-mock smoke:             PASS
Unified action space:       PASS
Autonomous decision loop:   PASS
Bounded HPO:                PASS
ASHA:                       PASS
Rollback:                   PASS
Runtime preflight (83):     PASS (83/83, unknown_hooks=0)
Non-GPU verification:       PASS (fast=PASS, slow=PASS, ruff=PASS)
Training gate:              UNLOCKED

REAL TRAINING EXECUTED:     NO
========================================
```

## Runtime hook identities (Prompt-18G)

Every preflight record now resolves to an audited
[`RuntimeHookIdentity`](../yolo_agent/research/runtime_hook_identity.py)
instead of the legacy `runtime_hooks: [unknown]` string.  For each of the
83 frozen papers the artifact
(`artifacts/paper_83_runtime_preflight.yaml`) pins:

| Field | Meaning |
| --- | --- |
| `hook_id` | canonical identity, e.g. `hook.assignment.optimal_transport` |
| `phase` | one of `data`, `preprocess`, `model_graph`, `loss`, `assignment`, `optimizer`, `train_step`, `postprocess`, `inference`, `evaluation` |
| `implementation_path` / `class_name` / `method_name` | the resolved importable callable |
| `insertion_point` | where the hook joins the runtime |
| `component_id` / `paper_id` / `paper_binding_id` | ownership; a shared adapter still yields one paper-specific binding per paper |
| `source_sha256` | hash of the backing adapter source; any later edit invalidates the pin |

Hard rules enforced by the acceptance gate:

* a real executable paper must resolve to at least one identity — the
  preflight sweep demotes any PASS record whose hooks are `unknown` or
  unaudited (`unknown_runtime_hooks` must be 0);
* a missing callable or an invalid source path fails the sweep before
  execution;
* data-side surfaces (sampling, augmentation, annotation, postprocess)
  resolve to their true phase (`data`/`preprocess`/`postprocess`) and are
  never forced into a model-forward shape.

The resolver lives in
[`runtime_hook_resolution.py`](../yolo_agent/research/runtime_hook_resolution.py);
scenario pins in `tests/test_paper_18g_hook_identity_scenarios.py` and
`tests/test_paper_18g_real_hook_scenarios.py`.
