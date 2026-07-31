# Paper MethodProfile 与 Adapter 复用

`PaperMethodProfile` 保存论文特有的方法名、参数、限制、协议条件和
`source_location`。它是 `paper_claim`/`paper_prior` 元数据，不是本地训练
evidence，也不会直接创建 recipe 或训练队列。

每篇论文只进入一个 `PaperImplementationDecision` 分类：

- `reuse_existing_adapter`：canonical mechanism 已有可导入 adapter，复用同一实现。
- `new_method_profile`：机制已知，但论文信息不足以定义 runtime contract。
- `new_component_adapter`：机制和接入点明确，但还没有已验证 adapter。
- `coupled_recipe`：论文方法明确涉及多个 canonical component，必须保留内部消融。
- `separate_detector_family`：论文或组件与 YOLO26 执行语义不兼容。
- `insufficient_information`：组件 alias、方法边界或来源信息不足。

Alias resolution 只确定 canonical mechanism，不会提升组件成熟度。同一个
canonical component 被多篇论文引用时，`adapter_to_papers` 会把论文映射到同一个
adapter ID；论文特有差异保留在 MethodProfile，不复制 adapter 类。

`exact_reproduction_claim` 和 `component_adaptation` 是互斥字段。默认接入是
component adaptation；只有存在独立、严格匹配论文协议的认证证据时，才能在其他
本地 reproduction artifact 中声明 exact reproduction。MethodProfile 本身不能完成
这项提升。

离线 ResearchSnapshot 构建会自动生成并冻结：

```text
paper_method_coverage.yaml
```

报告包含 paper-to-adapter 映射、六类决策计数、稳定 decision hash，以及每个未实现
component 的原因。维护者也可以在不修改 registry 的情况下单独生成报告：

```powershell
python -m yolo_agent.tools.paper_method_coverage `
  --root research `
  --report runs/paper-method-coverage.yaml
```

使用冻结快照时传入 `--snapshot <snapshot-directory>`；工具会先验证 snapshot
artifact hash。训练期间不会运行这个生成器，也不会联网读取论文。
