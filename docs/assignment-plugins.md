# Guarded Assignment Plugins

YOLO26 assignment extensions use a strict shadow-first lifecycle:

1. Audit the installed `E2ELoss`, both assignment paths, STAL behavior, NMS-free head, and DFL-free regression.
2. Run the native task-aligned assigner as the explicit baseline plugin.
3. Compute a paper method in shadow mode without changing native loss tensors.
4. Record positive count, positive ratio, foreground disagreement, GT conflicts, and conflict rate.
5. Allow an active pilot only when a matching shadow evidence artifact passes the activation gate.

The first guarded methods are TOOD Task Alignment Learning, OTA, and DSLA. They are independent implementations and remain paper adaptations, not exact reproductions of the complete paper systems. TOOD does not add the task-aligned head, OTA retains YOLO26 native losses, and DSLA adapts FCOS scale priors to YOLO26 point candidates.

Paper assignment adapters may replace only the declared `one_to_many` training path. The native `one_to_one` NMS-free path, detection head, DFL-free box regression, and inference graph remain unchanged. Anchor-based assigners are rejected. A proposal that changes assignment together with a head or loss must be a `CoupledRecipe` with an internal ablation plan.

Shadow evidence is local runtime evidence. Paper-reported improvements remain `paper_prior` and cannot promote an active or full candidate.
