# Awesome-object-detection 离线适配

YOLO Agent 可以把 [whut09/Awesome-object-detection](https://github.com/whut09/Awesome-object-detection) 作为论文 catalog 导入本地 Research Registry。该集成用于建立方法先验和组件索引，不会下载训练数据，也不会在训练时访问网络。

## 它是什么，不是什么

- 它是论文元数据、summary、note、组件线索和来源信息的离线 catalog。
- 它不是 COCO 或其他训练集，不包含可直接替代数据集的图片与标注。
- catalog 中的 AP、延迟和消融结果都是 `paper_claim`，不是本地 evidence。
- catalog 中出现 component ID 不代表 YOLO Agent 已实现对应 adapter。
- `direct_adapter_candidate` 或 `recipe_idea_only` 只是研究优先级，不是 executable 状态。

## 导入本地 Checkout

这些命令属于 advanced 研究流程，应在训练前执行：

```powershell
yolo-agent research import-awesome --source E:\path\Awesome-object-detection
yolo-agent research import-awesome --source E:\path\Awesome-object-detection --dry-run
```

导入器读取本地 catalog，保留 `paper_id`、来源仓库、commit、路径、record hash、原始分类、applicability、harness hints 和 component IDs。重复导入是幂等的；来自其他 source 的 registry 记录不会被删除。

缺失 abstract 时可以使用 summary，但会记录 `abstract_source=summary`。缺失 benchmark 或 license 时保持空值/unknown，不能补写猜测内容。

## 构建冻结 Snapshot

```powershell
yolo-agent research build-snapshot --root research --source awesome_object_detection --maturity-registry runs/component_maturity_registry.yaml
```

有本地缓存的官方代码 README/config 时，可以显式加入离线提取：

```powershell
yolo-agent research build-snapshot --root research --source awesome_object_detection --cached-code-root E:\paper-code-cache
```

cache 目录使用 `<root>/<owner>/<project>/README*.md` 和 YAML/JSON/TOML config。命令不会 clone、fetch 或访问 official code URL；缺失、过大或不可读文件只记录 warning，不阻塞 snapshot。

生产链为：

```text
validate -> import -> deduplicate -> classify -> alias resolve
-> note/hint parse -> MethodProfile/adapter reuse -> recipe priors
-> compatibility review -> snapshot
```

Snapshot 记录 source commit、catalog hash、paper/component/recipe 版本、MethodProfile coverage，以及有效 adapter/Ultralytics/protocol/maturity artifact 身份。catalog、commit、adapter hash 或有效 overlay 变化都会产生新 hash。训练 child run 继承 base run 的 snapshot，训练期间 live registry 的变化不会影响已有 run。

## 成熟度不会被导入提升

导入论文只能增加 `paper_claim`、组件 alias 和 recipe prior。它不会自动把组件提升为 `adapter_implemented`、`runtime_integrated`、`unit_tested`、`smoke_passed`、`gpu_certified`、`pilot_reproduced`、`full_reproduced` 或 `confirmed_multi_seed`。

真正进入训练队列前，组件必须有 ComponentContract、真实 adapter、YOLO26 compatibility 结果，以及 hash-bound runtime、unit-test 和非 mock smoke artifacts。之后仍需 matched pilot、完整 post-eval、paired delta 与预算门禁。full COCO 必须显式确认，`+2 mAP` 不作保证。

当前冻结论文数、已实现 adapter 数、runtime-integrated 数和 pilot-reproduced 数分别记录在 [paper-adapter-coverage.yaml](paper-adapter-coverage.yaml)，不能相互替代。

每篇论文到 canonical component、复用 adapter 或未实现原因的映射记录在 snapshot
中的 `paper_method_coverage.yaml`。多篇论文描述同一机制时只复用一个 adapter，
论文特有参数与限制保留在 MethodProfile 中。

字段级方法证据和不足原因记录在 `paper_method_evidence.jsonl` 与
`paper_method_evidence_coverage.yaml`。标题命中只能作为低置信度 prior；明确本地
summary/note/cache 证据才能降低 `insufficient_information`。MethodProfile 补全不等于
adapter 实现，不能直接进入训练队列。

成熟度定义见 [能力成熟度](capability-maturity.md)，完整决策边界见 [Paper Intelligence](paper-intelligence.md)。
