# Paper Mechanism Coverage

YOLO-Agent maps offline paper records through an auditable chain:

```text
paper -> MethodProfile -> cited source term -> canonical mechanism
      -> mechanism cluster -> adapter family -> runtime readiness
```

`PaperMechanismClusterer` groups papers by runtime semantics, not by title or
paper identity. The bundled taxonomy defines 19 reusable families: sampling and
class balancing, hard-example mining, augmentation, assignment, quality
alignment, train-time confidence calibration, localization loss, feature and
logits distillation, multi-scale fusion, small-object heads, attention,
re-parameterized convolution, lightweight neck, feature alignment, domain
adaptation, open vocabulary, slicing inference, and post-processing calibration.

An `exact_match` requires a canonical component that identifies one cluster. A
`semantic_match` records source evidence, source location, confidence, and a
reason. Shared components such as generic distillation remain unresolved until
local text distinguishes feature alignment from logits/output alignment.
Sampling is not merged with hard-example mining, and train-time calibration is
not merged with post-hoc calibration. A paper may map to several independent
clusters when it explicitly changes several mechanisms.

Coverage uses unique canonical mechanisms as its denominator. Paper count is
only reference frequency. The generated `paper_method_coverage.yaml` reports
potentially adaptable, reusable-adapter, and runtime-ready mechanism counts and
ratios separately.

Task and detector-family labels such as `small_object`, `object_detection`, and
`detr` are not executable mechanisms. The offline profiler searches the local
summary, note, harness hints, and official-code metadata for explicit mechanism
evidence. Without that evidence, the paper remains `insufficient_information`.

Complete DETR, open-vocabulary, vision-language, and cross-modal methods use the
`separate_detector_family` track. Reusing an isolated mechanism is component
adaptation, not exact paper reproduction. Alias resolution and paper claims do
not raise adapter maturity or provide local training evidence.

The production pipeline writes `paper_mechanism_clusters.yaml` with every
paper-to-cluster match, parameter differences, limitations, source locations,
semantic conflicts, and adapter-family opportunities. A compact Markdown report
is written beside it. The YAML report and cluster taxonomy are frozen into the
ResearchSnapshot; training reads that frozen artifact and does not recluster the
live paper registry.

The implementation-opportunity ranking answers only which new adapter family
could cover the most source papers. It does not authorize code generation,
training, promotion, or maturity changes. Compatibility, runtime hooks, local
negative evidence, cost, smoke artifacts, matched controls, and ASHA remain
authoritative.

Generate an audit without modifying the registry:

```powershell
python -m yolo_agent.tools.paper_method_coverage `
  --root research `
  --report runs/paper-method-coverage.yaml
```

For paper-level executable coverage and the four explicit denominators, see
[Executable Paper Coverage](executable-paper-coverage.md).
