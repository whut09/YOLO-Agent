# Paper Adapter Implementation Queue

`PaperAdapterImplementationPlanner` turns Awesome-object-detection component metadata
into a bounded engineering backlog. It does not generate adapter source code, create a
training node, or treat `direct_adapter_candidate` as executable.

## Inputs

The deterministic planner ranks canonical components using:

- current, local error facts;
- paper year and official-code availability;
- source license;
- audited YOLO26 compatibility and component maturity;
- estimated adapter, latency, and model-size cost;
- verified runtime-hook availability;
- local Policy Memory evidence;
- implementation history fingerprints and family cooldown.

Paper year contributes only a small freshness prior. Repeated or confirmed local
negative evidence has a substantially larger penalty, so a newer paper cannot erase a
local regression.

## Output Tracks

- `ready_to_materialize`: adapter and runtime hook are verified through smoke maturity.
- `implementation_queue`: a canonical adapter is missing; a bounded
  `implementation_request` is generated with acceptance tests.
- `shadow_evaluation_queue`: an implemented high-risk adapter must collect shadow
  evidence before active use.
- `incompatible`: the component violates an audited YOLO26 boundary.
- `separate_detector_family`: DETR, open-vocabulary, open-world, grounded, and
  vision-language detector work stays outside the YOLO26 adapter queue.
- `insufficient_information`: aliases or current diagnosis evidence are incomplete.
- `deferred`: duplicate fingerprints or component-family cooldown postpone otherwise
  actionable work.

For AP_small and false-negative diagnoses, the reviewed priority order is
small-object sampling, P2 head, YOLO26 distillation, and isolated SAHI inference. The
order is still reduced by local negative evidence and deployment cost.

## Safety Boundary

An implementation request contains a canonical component ID, insertion point,
required runtime hook, rollback-oriented acceptance tests, and source paper IDs. It
sets `generated_code_allowed=false`. A later engineering task must implement and test
the adapter explicitly; normal training cannot consume the request.

Equivalent work from multiple papers is collapsed by a fingerprint based on canonical
component, insertion point, runtime hook, and detector track. Family cooldown prevents
repeated adjacent implementation work. The plan can be serialized to YAML/JSON and
recorded in the Decision Ledger with its stable `plan_hash`.
