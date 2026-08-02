# Quality and Localization Loss Adapters

YOLO-Agent provides nine independent, additive YOLO26 loss mechanisms:

| Component | Auxiliary mechanism |
| --- | --- |
| `loss.quality.iou_aware_classification` | IoU-aware classification quality |
| `loss.quality.correlation` | Confidence/localization correlation |
| `loss.calibration.bpc` | BPC-style confidence calibration |
| `loss.quality.pseudo_iou` | Pseudo-IoU quality target |
| `loss.quality.localization_aware` | Localization-aware classification |
| `loss.boundary_aware` | Boundary-aware box residual |
| `loss.localization.uncertainty_weighted` | Uncertainty-weighted box residual |
| `loss.hard_negative_classification` | Bounded hard-negative classification |
| `loss.class_balanced_focal` | Class-balanced focal classification |

Each component is an `AtomicRecipe` with one changed variable,
`loss.<mechanism>.weight`, and fixed `imgsz=640`. The runtime adds a weighted
term to the native total loss. It does not replace YOLO26's native assigner,
restore DFL, or change the inference graph. Runtime evidence records the raw
and weighted term, gradient observations, hook calls, payload/protocol hashes,
rank, and checkpoint sidecars.

These implementations are reusable **component adaptations**, not exact paper
reproductions. Paper claims remain `paper_prior`; only matched local evaluation
can become promotion evidence. Source contracts remain `adapter_implemented`
until artifact-backed certification advances the local maturity overlay.

Run isolated CPU certification before any GPU certification:

```powershell
yolo-agent advanced certify-component --component loss.quality.correlation --cpu
yolo-agent advanced certify-component --component loss.quality.correlation --gpu
```

GPU certification is explicit and does not imply `pilot_reproduced`. Automatic
training may materialize a loss recipe only when the frozen ResearchSnapshot
contains a valid `smoke_passed` overlay and matched-control evidence is
available.
