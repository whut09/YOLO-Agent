# Current Executable Coverage Baseline

Numbers below are read from `research/production/coverage_baseline.yaml`
(report hash `882d0e2511fd813f...`, schema `executable_paper_coverage.v1`),
which is generated from the local 728-paper Awesome Object Detection catalog
and the current artifact-backed component maturity registry. If this page and
that file ever disagree, the file wins: regenerate and re-check the numbers
instead of editing them by hand.

| Denominator | Papers |
|---|---:|
| `all_papers` | 728 |
| `yolo26_compatible_papers` | 84 |
| `adaptable_component_papers` | 84 |
| `exact_reproduction_candidates` | 0 |

Additional execution counts (both count papers out of the full 728-entry
catalog, computed in `yolo_agent/research/executable_coverage.py:107-112`):

- `reusable_adapter_paper_count`: 86 — papers with at least one reusable
  adapter candidate.
- `runtime_ready_paper_count`: 84 — papers with at least one currently valid
  runtime-ready adapter.

The runtime-ready count is intentionally strict: it accepts only hash-valid,
non-mock maturity artifacts for the current adapter and Ultralytics identity.
It can differ from the reusable count because a reusable candidate only
becomes runtime-ready once such artifacts exist for the current runtime
identity.

This report does not mean that the remaining compatible papers are impossible
to implement. It means they need a valid adapter or renewed runtime/smoke
certification before they can authorize training. Exact reproduction remains
zero because no component adaptation has been promoted to an exact paper
reproduction claim.

Regenerate the machine-readable and field-level reports with:

```powershell
yolo-agent research coverage-baseline --root research --output runs/coverage_baseline.yaml
```
