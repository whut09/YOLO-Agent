# Data Pipeline Paper Adapters

YOLO-Agent provides nine reusable, train-only data mechanisms for YOLO26:

| Component | Changed variable | Runtime hook |
| --- | --- | --- |
| `sampling.small_object_weighted` | `data.small_object_weighted_sampling` | train dataloader |
| `sampling.class_balanced` | `data.class_balanced_sampling` | train dataloader |
| `sampling.repeat_factor` | `data.repeat_factor_sampling` | train dataloader |
| `sampling.hard_negative_replay` | `data.hard_negative_replay` | train dataloader |
| `sampling.false_negative_class_boost` | `data.false_negative_class_boost` | train dataloader |
| `augmentation.copy_paste_rare_classes` | `data.copy_paste_rare_classes` | train dataset |
| `augmentation.scale_aware_crop` | `data.scale_aware_crop` | train dataset |
| `augmentation.object_centric_crop` | `data.object_centric_crop` | train dataset |
| `augmentation.multi_image_sampling_schedule` | `data.multi_image_sampling_schedule` | train dataset |

Each component is an independent `AtomicRecipe`. All recipes fix `imgsz=640`
and leave validation and test datasets unchanged. Sampling uses deterministic
global exposure followed by rank-position DDP sharding. Sampler and transform
state is rank-scoped and restored on resume, including Windows spawn workers.

Every runtime writes an atomic `<mechanism>_manifest.json`. Exposure manifests
contain the dataset identity, seed, rank, class counts, raw and bounded weights,
clipping statistics, and selected sample count. Transform manifests contain the
synchronized image/box parameters. Protocol, runtime payload, and adapter hashes
bind the artifact to the executed implementation.

Hard-negative replay and false-negative class boost require local error
evidence. A paper claim cannot supply that evidence. Zero-strength sampling and
zero-probability transforms return the native Ultralytics loader or dataset
object, providing a baseline-equivalence path.

These are reusable component adaptations. A paper maps to one of them only when
its local summary, note, or cached metadata identifies the same mechanism and
method boundary. Paper-specific parameters and limitations remain in the
`MethodProfile`. The adapter does not claim exact paper reproduction unless an
explicit reproduction profile and protocol authorize that claim.

Source contracts remain `adapter_implemented`. Unit tests, mock runs, or catalog
mapping do not make them executable. Automatic training requires an
artifact-backed maturity overlay through `smoke_passed`, current error facts, a
matched control, and ASHA approval.
