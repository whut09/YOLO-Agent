# YOLO26 Distillation Mechanisms

YOLO Agent provides eight reusable, training-only distillation components:

| Component | Runtime signal | Changed variable |
| --- | --- | --- |
| `distillation.logits` | output distributions | `loss.distillation.logits.weight` |
| `distillation.feature` | validated intermediate features | `loss.distillation.feature.weight` |
| `distillation.localization` | decoded box responses | `loss.distillation.localization.weight` |
| `distillation.relation` | bounded feature relations | `loss.distillation.relation.weight` |
| `distillation.attention` | channel and spatial attention | `loss.distillation.attention.weight` |
| `distillation.masked_feature` | teacher-attention masked features | `loss.distillation.masked_feature.weight` |
| `distillation.quality_aware` | teacher-confidence weighted responses | `loss.distillation.quality_aware.weight` |
| `distillation.teacher_ensemble` | multiple teacher probabilities | `loss.distillation.teacher_ensemble.weight` |

They share `YOLO26DistillationAdapter` and the Ultralytics trainer bridge, but each has an independent component identity, AtomicRecipe, runtime payload, evidence file, resume state, and certification result. CoupledRecipe combinations must retain their internal baseline and single-component ablations.

Teachers are loaded from explicit local `yolo26s.pt` or `yolo26m.pt` checkpoints, remain frozen in `eval` mode, and run under `no_grad`. Teacher and student consume the same preprocessed batch tensor, while the student inference graph remains unchanged. Feature-based mechanisms fail closed unless every configured student and teacher hook fires.

CPU certification verifies a real YOLO26 single batch, AMP, backward, zero-weight equivalence, resume identity, DDP rank isolation, checkpoint hashes, hook locations, and loss contribution artifacts. GPU certification is opt-in and cannot promote beyond `gpu_certified`; matched pilots and local paired evidence are still required for `pilot_reproduced`.

CrossKD, PKD, Localization Distillation, and related papers are MethodProfiles unless the implemented formula and protocol match the paper. Component adaptation never becomes an exact reproduction claim automatically, and paper-reported deltas remain paper prior rather than local evidence.
