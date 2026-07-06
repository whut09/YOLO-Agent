# LLM 配置

YOLO Agent 默认会让 LLM 参与“诊断、建议和策略 proposal”，但不会让 LLM 直接批准实验、绕过 evidence gate，或在没有证据时声称最佳模型。

没有 `OPENAI_API_KEY` 时，命令不会失败；系统会跳过 LLM proposal，回退到规则策略。

## 1. 设置 API Key

当前 PowerShell 会话临时设置：

```powershell
$env:OPENAI_API_KEY="..."
```

如果想长期使用，可以写入本地 `.env.local` 或自己的 PowerShell profile。不要把真实 key 写进会提交的文件。

## 2. 检查 LLM 配置

只检查 LLM：

```powershell
yolo-agent doctor --llm
```

和训练环境一起检查：

```powershell
yolo-agent doctor --data E:\dataset\coco.yaml --model yolo26n.pt --llm
```

常见输出：

```text
llm status=ready
llm enabled=true
llm provider=openai
llm model=gpt-5.5
llm executable_decisions_allowed=false
```

如果没有 key，会看到：

```text
llm status=missing_key
llm fallback=rule_engine
```

这不是错误。它只表示本轮不会调用 LLM，系统会继续使用规则策略、EvidenceGate、CompatibilityChecker 和 UtilityScorer。

## 3. 配置文件在哪里

提交到仓库的是脱敏模板：

```text
configs/llm_decision.example.yaml
```

本地可用配置放在：

```text
configs/local/llm_decision.local.yaml
```

`configs/local/` 已经被 `.gitignore` 忽略，可以放真实 provider、model、api_key_env 等本地信息。

## 4. LLM 能做什么，不能做什么

LLM 可以做：

- 生成诊断摘要
- 提出 policy proposals
- 提醒缺少哪些 evidence
- 起草 doctor-style 决策报告

LLM 不能做：

- 直接启动训练
- 跳过 full-run 二次确认
- 绕过 compatibility / evidence / budget gate
- 没有 metrics 时宣称模型更好

也就是说，LLM 是策略生成器，不是最终裁判。最终是否执行，仍由 harness 的状态机、证据契约和执行队列决定。
