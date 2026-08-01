# Executable Paper Coverage

YOLO-Agent reports paper coverage with four separate denominators. A single
percentage cannot represent paper metadata, YOLO26 compatibility, adapter
availability, and runtime certification at the same time.

## Denominators

- `all_papers`: every unique paper with one MethodProfile and one implementation
  decision.
- `yolo26_compatible_papers`: papers with an explicitly mapped compatible or
  adapter-required YOLO26 mechanism. `separate_detector_family` is excluded.
- `adaptable_component_papers`: compatible papers that can be scoped to isolated
  components instead of replacing the detector family.
- `exact_reproduction_candidates`: papers with an explicit exact-reproduction
  claim, complete protocol evidence, official code metadata, and runtime-ready
  adapters. Component adaptation never qualifies by itself.

Each paper records:

```text
compatibility_class, adaptation_scope, blocking_fields,
canonical_mechanisms, reusable_adapter_candidates,
runtime_ready_adapters, required_runtime_hooks,
implementation_cost, expected_resource_cost,
exact_reproduction_possible, exclusion_reason
```

Implementation and resource costs are qualitative declarations derived from
local component contracts. Missing values remain `unknown`; paper benchmarks
are never treated as local runtime evidence.

## Generate

Build a fresh frozen ResearchSnapshot before training:

```powershell
yolo-agent research build-snapshot `
  --root research `
  --source awesome_object_detection
```

Audit that frozen snapshot:

```powershell
yolo-agent research coverage-baseline `
  --root research `
  --output runs/coverage_baseline.yaml
```

The command writes `coverage_baseline.yaml` and `coverage_baseline.md`. The YAML
is machine-readable authority; Markdown is a field-level report. The audit reads
the frozen MethodProfiles, contracts, taxonomy, and maturity identities. It does
not read the live paper registry or grant training eligibility.

During snapshot production the same artifacts are frozen as:

```text
executable_coverage_baseline.yaml
executable_coverage_report.md
```

## Interpretation

- A catalog component ID is metadata, not an implementation.
- An importable adapter class is not runtime-ready.
- Runtime-ready requires valid frozen maturity artifacts.
- One paper may reuse multiple adapters.
- One canonical adapter may serve many papers with the same mechanism.
- Paper claims remain separate from local evidence.
- Exact reproduction remains separate from component adaptation.

