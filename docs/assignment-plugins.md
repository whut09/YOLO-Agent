# Guarded Assignment Plugins

YOLO26 assignment extensions use a strict shadow-first lifecycle:

1. Audit the installed `E2ELoss`, both assignment paths, STAL behavior, NMS-free head, and DFL-free regression.
2. Run the native task-aligned assigner as the explicit baseline plugin.
3. Compute a paper method in shadow mode without changing native loss tensors.
4. Record positive count, positive ratio, foreground disagreement, GT conflicts,
   conflict rate, and matching stability for every declared assignment path.
5. Allow an active pilot only when a matching shadow evidence artifact passes the activation gate.

The guarded paper-specific methods are TOOD Task Alignment Learning, OTA, and
DSLA. They remain component adaptations, not exact reproductions of complete paper
systems. TOOD does not add the task-aligned head, OTA retains YOLO26 native losses,
and DSLA adapts FCOS scale priors to YOLO26 point candidates.

Reusable mechanism adapters are also available for task-aligned weighting, dynamic
top-k matching, quality-aware matching, soft-label assignment, dual-path adaptation,
and conflict-aware positive selection. Each mechanism has its own component ID,
changed variable, AtomicRecipe, payload identity, and evidence artifact. A paper can
reuse one only when its MethodProfile contains explicit local text evidence.

Single-path adapters may replace only the declared `one_to_many` training path. The
dual-path adapter is the only mechanism allowed to declare `both`; it must pass
separate `one_to_many` and `one_to_one` shadow aggregates before active replacement.
The native detection head, DFL-free box regression, NMS-free inference graph, and
point representation remain unchanged. Anchor-based assigners are rejected.

The native YOLO26 assigner is an explicit baseline plugin, not a paper candidate.
Generic terms such as `assignment` remain unresolved. A proposal that changes
assignment together with a head or loss must be a `CoupledRecipe` with an internal
ablation plan.

Shadow evidence is local runtime evidence. Paper-reported improvements remain `paper_prior` and cannot promote an active or full candidate.

CPU certification can validate a shadow runtime and advance an effective local
overlay to `smoke_passed`. GPU certification is opt-in. Neither creates an active
pilot or claims reproduction. Active pilots additionally require a same-protocol
shadow artifact, matched control, fixed `imgsz=640`, and ASHA materialization.
