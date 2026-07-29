# Paper Recipe Materialization

Paper records and `RecipePrior` objects are non-executable. A paper-derived method reaches a pilot only through the certified materialization chain:

```text
frozen ResearchSnapshot
-> current-run, current-protocol COCO error facts
-> RecipePrior and canonical component contract
-> local runtime adapter lookup
-> compatibility check
-> runtime dry-run and smoke evidence
-> matched baseline control
-> RecipeCritic and eligibility gate
-> ASHA registration
-> RoundExecutionPlan
-> ExecutionQueue
```

## Hard Boundaries

- The adapter must be both `runtime_execution_ready` and `smoke_passed`.
- A runtime payload must be importable, hash-matched, protocol-bound, and invoked through the typed Python entrypoint.
- Missing adapters produce an `implementation_request`; the system does not generate adapter code during training.
- Missing current-node error facts produce evidence recovery only.
- Candidate and matched control keep `imgsz=640` and the same comparison protocol.
- `RecipePrior` and `policy_evaluation.yaml` have no direct queue authority.
- ASHA is the only pilot budget authority; `RoundExecutionPlan` is the only queue source.
- Scalar HPO is disabled by default. When certified paper/component recipes are exhausted, the loop stops explicitly.
- There is no fallback to an ordinary Ultralytics command after adapter preparation.

## Audit Identity

For every registered component candidate, terminal events and `decision_ledger.jsonl` record:

- component and adapter IDs;
- adapter class and version where available;
- aggregate patch hash;
- runtime payload hash;
- ASHA assignment and queue authority.

These fields identify what actually ran. Paper claims remain separate prior evidence and never become candidate metrics.

## Failure Outcomes

| Condition | Outcome |
| --- | --- |
| No current protocol error facts | evidence recovery only |
| Metadata-only component | implementation request |
| Adapter import or runtime payload missing | implementation request |
| Smoke, compatibility, 640, or matched-control gate fails | candidate rejected |
| Cohort incomplete | wait for more certified `pilot_3` candidates |
| Certified recipe space exhausted | stop; no scalar HPO fallback |
| Candidate accepted | register with ASHA; ASHA may issue one bounded assignment |

Passing materialization does not claim local improvement. Promotion still requires complete COCO post-evaluation and verified paired evidence.
