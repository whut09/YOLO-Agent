# Paper Mechanism Coverage

YOLO-Agent maps offline paper records through an auditable chain:

```text
paper -> MethodProfile -> cited source term -> canonical mechanism
      -> YOLO26 compatibility -> reusable adapter -> runtime readiness
```

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

Generate an audit without modifying the registry:

```powershell
python -m yolo_agent.tools.paper_method_coverage `
  --root research `
  --report runs/paper-method-coverage.yaml
```
