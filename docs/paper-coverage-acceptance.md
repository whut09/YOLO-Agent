# Paper Implementation Coverage Acceptance

Status: **passed**

## Acceptance Metrics

| Metric | Result | Target | Status |
| --- | ---: | ---: | --- |
| `compatible_paper_method_profiles` | 85/85 (100.0%) | >=85% | passed |
| `compatible_mechanism_reusable_adapters` | 20/23 (87.0%) | >=80% | passed |
| `compatible_mechanism_runtime_integrated` | 18/23 (78.3%) | >=70% | passed |
| `compatible_mechanism_smoke_passed` | 18/23 (78.3%) | >=60% | passed |
| `compatible_papers_certified_adapter` | 83/85 (97.6%) | >=70% | passed |

## Independent Categories

- All paper traces: 728
- Exact reproduction candidates: 0
- Separate detector family: 168
- Insufficient information: 475

Exact reproduction is not inferred from component adaptation.

## Residual Mechanisms

These mechanisms remain useful follow-up work even when aggregate acceptance thresholds pass.

| Mechanism | Papers | Adapter | Remaining work |
| --- | ---: | --- | --- |
| `detection_head.dynamic` | 2 | - | no reusable adapter mapping; implement or map an evidence-equivalent adapter |
| `attention.deformable` | 1 | - | no reusable adapter mapping; implement or map an evidence-equivalent adapter |
| `attention.spatial` | 1 | `attention.spatial` | adapter exists but lacks valid runtime-integrated artifact identity |
| `detection_head.task_aligned` | 1 | - | no reusable adapter mapping; implement or map an evidence-equivalent adapter |
| `inference.sahi_slicing` | 1 | `inference.sahi_slicing` | adapter exists but lacks valid runtime-integrated artifact identity |

## Traceability

- Method coverage SHA-256: `b486d7dcdb1fdc191ae4e214adb00aee6e3acd35c6a0314595bcc9137aebc57f`
- Executable coverage hash: `b4debecc5829df6492ba809a669e62e89e792104e5948439e0e45ec361470990`
- Maturity registry SHA-256: `575b79051b9089166e1757f40539597f20f0e59084e17b2e1d4d92d3200a9434`
- Acceptance report hash: `797c3b912852717b03e3ce7fc55a3650d8b028f7d1dc9fc2a827c65c5996667c`

The adjacent YAML artifact contains every numerator and denominator ID plus paper, mechanism, adapter, protocol, artifact path, and artifact SHA-256 trace.
