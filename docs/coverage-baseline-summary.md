# Current Executable Coverage Baseline

This baseline was generated from the local 728-paper Awesome Object Detection
catalog and the current artifact-backed component maturity registry.

| Denominator | Papers |
|---|---:|
| `all_papers` | 728 |
| `yolo26_compatible_papers` | 80 |
| `adaptable_component_papers` | 80 |
| `exact_reproduction_candidates` | 0 |

Additional execution counts:

- Papers with at least one reusable adapter candidate: 47
- Papers with at least one currently valid runtime-ready adapter: 2

The runtime-ready paper count is intentionally strict. Thirteen reusable
adapter classes are present, but the fresh snapshot accepts only hash-valid,
non-mock maturity artifacts for the current adapter and Ultralytics identity.
At this audit, the valid runtime identities are the three neck adapters; their
paper references overlap, resulting in two unique runtime-ready papers.

This report does not mean that the remaining compatible papers are impossible
to implement. It means they need a valid adapter or renewed runtime/smoke
certification before they can authorize training. Exact reproduction remains
zero because no component adaptation has been promoted to an exact paper
reproduction claim.

Regenerate the machine-readable and field-level reports with:

```powershell
yolo-agent research coverage-baseline `
  --root research `
  --output runs/coverage_baseline.yaml
```

