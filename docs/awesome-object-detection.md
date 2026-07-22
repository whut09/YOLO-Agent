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
yolo-agent research build-snapshot --root research --source awesome_object_detection
```

生产链为：

```text
validate -> import -> deduplicate -> classify -> alias resolve
-> note/hint parse -> recipe priors -> compatibility review -> snapshot
```

Snapshot 记录 source commit、catalog hash、paper/component/recipe 版本和生成时间。相同 catalog 与 commit 产生稳定 hash；内容或 commit 变化会产生新 hash。训练 child run 继承 base run 的 snapshot，训练期间 live registry 的变化不会影响已有 run。

## 成熟度不会被导入提升

导入论文只能增加 `paper_claim`、组件 alias 和 recipe prior。它不会自动把组件提升为 `adapter_implemented`、`smoke_passed`、`pilot_reproduced` 或 `full_reproduced`。

真正进入训练队列前，组件必须有 ComponentContract、真实 adapter、YOLO26 compatibility 结果和 smoke evidence。之后仍需 matched pilot、完整 post-eval、paired delta 与预算门禁。full COCO 必须显式确认，`+2 mAP` 不作保证。

成熟度定义见 [能力成熟度](capability-maturity.md)，完整决策边界见 [Paper Intelligence](paper-intelligence.md)。
