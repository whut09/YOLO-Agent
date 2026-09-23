# YOLO-Agent Documentation Audit

- 审计日期：2026-09-23
- 审计方式：只读核对。代码为第一 source of truth，machine-readable config/artifact 为第二，文档仅描述真实能力。
- 核对手段：`git ls-files`、源码符号检索（Grep）、CLI parser 源码（`yolo_agent/cli.py` `build_parser()`）、`configs/*.yaml`、`artifacts/*.yaml`、tests 命中。
- 范围：`README.md`、`README.zh-CN.md`、`docs/**/*.md`（56 个）、`configs/**/*.yaml`、`artifacts/*.yaml`、`pyproject.toml`、`yolo_agent/**/*.py`。
- 本审计只新增本文件与 `artifacts/documentation_audit.json`，未修改任何 runtime 行为或既有文档。
- 机器可读版本：`artifacts/documentation_audit.json`。

## Summary

| 指标 | 数量 |
| --- | --- |
| P0（误导性事实错误，须立即修） | 1 |
| P1（过期/含混/双语漂移，应尽快修） | 4 |
| P2（改进项/记录在案） | 5 |
| 验证为 correct 的关键 claims | 12 |
| 命令示例 vs parser mismatch | 1（叙述性，非 flag 错误） |
| Bash `\` 续行符（PowerShell 语境失效） | 0 |

总体结论：**文档质量高于预期**。命令示例与 CLI parser 高度一致（唯一问题是 quickstart 的一处过时叙述）；README 覆盖统计数字与 yaml artifact 完全可对上；maturity 表格由单一 config 生成、三处呈现一致；"exact reproduction=0 / implementation-ready=83/83 / real training executed=NO" 三个概念在文档与 artifact 中区分清晰。主要问题集中在一处过时叙述、全库路径 typo、双语目录漂移和 imgsz=640 的"硬 invariant vs action space"表述张力。

## P0 Issues

### P0-1 quickstart 声称 setup 生成"下一条 `optimize` 命令"——已过时，与代码矛盾
- **source**: `docs/quickstart.md:25`
- **claim**: setup 会生成"下一条 `optimize` 命令"。
- **actual code evidence**:
  - `yolo_agent/tools/setup_wizard.py:96,202` — `next_command=_train_command(...)`；
  - `setup_wizard.py:213-219` — `_train_command` 拼接的是 `yolo-agent train --model ... --data ... --run-id ...`，不含任何 "optimize"；
  - `setup_wizard.py:275-276` — 终端输出 `next: {result.next_command}`；
  - `docs/cli.md:66` 自己也写明 "`Next:` 只应提示继续使用 `yolo-agent train ...`"，与 quickstart 互相矛盾；
  - 全库检索 `下一条|optimize 命令`：仅 quickstart.md:25 一处残留；代码中不存在生成 optimize 命令的路径。
- **status**: wrong (stale narrative)
- **recommended owner document**: `docs/quickstart.md`
- **recommendation**: 将"下一条 `optimize` 命令"改为"下一条 `train` 命令"。

## P1 Issues

### P1-1 示例路径 typo：`E:\datatset`（应为 `E:\dataset`），27 处 / 10 个文件
- **source**（全部命中位置）:
  - `docs/quickstart.md:22,31,36,46,58,61,64,67,108`（9 处）
  - `docs/troubleshooting.md:49,58,108`（3 处）
  - `docs/training-modes.md:30,48,111,131`（4 处）
  - `docs/coco-yolo26.md:10,31,48,62`（4 处）
  - `docs/cli.md:12,20,28`（3 处）
  - `docs/llm-setup.md:39`、`docs/install.md:52`、`docs/paper-candidate-routing-audit.md:242`、`docs/all-83-real-training-readiness.md:70`（各 1 处）
- **claim**: 示例数据集路径 `E:\datatset\coco.yaml`。
- **actual evidence**: 两份 README（`README.md:19-28`、`README.zh-CN.md:19-28`）统一使用 `E:\dataset\coco.yaml`；docs 子目录却系统性使用 `E:\datatset`。"datatset" 非任何真实约定路径，属拼写漂移。
- **status**: wrong (typo)
- **recommended owner document**: 上述 10 个文件（quickstart/troubleshooting/training-modes/coco-yolo26/cli 为主要 owner）
- **recommendation**: 全库 `datatset` → `dataset`（示例路径统一为 `E:\dataset\coco.yaml`，与 README 一致）。

### P1-2 README 双语漂移：中文"文档"目录缺 4 个条目
- **source**: `README.md:170-191`（英文目录 20 项）vs `README.zh-CN.md:149-166`（中文目录 16 项）
- **claim**: 两侧均为项目文档导航。
- **actual evidence**: 中文版缺失以下 4 个英文版已列条目：
  1. `docs/automatic-optimization-architecture.md`（Automatic optimization architecture）
  2. `docs/paper-adapter-implementation-queue.md`（Paper adapter implementation queue）
  3. `docs/sahi-inference-certification.md`（SAHI inference certification）
  4. `docs/inference-policy-adapters.md`（Isolated inference policy adapters）
- **status**: stale / bilingual drift
- **recommended owner document**: `README.zh-CN.md`
- **recommendation**: 中文"文档"章节补齐 4 条目（保留现有命名风格，如"自动优化架构"、"SAHI 推理认证"）。

### P1-3 能力成熟度审查日期过期：`reviewed_at: 2026-08-03`
- **source**: `configs/capability_maturity.yaml`（文件级 `reviewed_at: 2026-08-03`）；呈现在 `README.md:157-168`、`README.zh-CN.md:136-147`、`docs/capability-maturity.md:6`
- **claim**: capability-maturity 表格代表当前能力边界。
- **actual evidence**: 2026-08-03 之后仓库新增了大量高阶能力 artifact：`artifacts/pretraining_acceptance.yaml`（evaluated_at 2026-09-22T06:59:26Z）、`artifacts/training_release_v1.yaml`（2026-09-22，release_status READY_FOR_FIRST_TRAINING）、`artifacts/paper_83_runtime_preflight.yaml`（2026-09-21，83/83 passed）。各 capability 的 status 字段本身仍与代码一致（见 verified-claims），但"最近审计日期"已滞后约 7 周，且未反映 paper-83 预检/发布等新门禁能力。
- **status**: stale (date), content correct
- **recommended owner document**: `configs/capability_maturity.yaml`（唯一修改点），再由 `python -m yolo_agent.tools.capability_matrix` 重新生成三处呈现。
- **recommendation**: 运行一次重新审查并更新 `reviewed_at`；可考虑为 "Paper-83 preflight/release 门禁" 增补一行 capability 条目。

### P1-4 `imgsz=640` 的表述张力：action space 声明了 resolution action，recipe 层硬拒绝
- **source**:
  - 文档侧（统一声称固定）：`docs/concepts.md:22,101,192`、`docs/coco-yolo26.md:67,81,85`、`docs/training-modes.md`、`README.md:120`、`README.zh-CN.md:99` 等全库 30+ 处
  - 代码侧（两套真相）：`configs/actions/detection_action_catalog.yaml:227-246`（`model.resolution.inference_sweep`，imgsz 640→1280，search_space [768,960,1280]）、`:248-267`（`model.resolution.train_imgsz`，640→960）；但 `yolo_agent/recipes/schemas.py:79-108` 硬校验 `RecipeValidationError("Recipe input size is fixed at imgsz=640")`，另有 `recipes/paper_priors.py:90-92`、`recipes/coupled_library.py:419`、`components/contracts.py:165-166`
- **claim**: 文档把 `imgsz=640` 描述为硬 invariant（"固定/never increased"）。
- **actual evidence**: 当前代码行为是**分层**的：recipe/组件插件层把 640 锁死（validate 阶段直接抛错），action space 却仍声明 resolution action（含回滚值 `inference.imgsz: 640`）。两者不冲突但语义不同——文档"硬 invariant"的说法没有说明这是 recipe 层约束而非 action space 层缺失。
- **status**: ambiguous
- **recommended owner document**: `docs/concepts.md`（统一动作空间描述处）+ `docs/coco-yolo26.md`
- **recommendation**: 在 concepts.md 增加一句澄清："resolution action 存在于统一动作空间（主要用于 inference sweep），但 YOLO26 公平对比的 recipe 层将训练 imgsz 锁定为 640，任何改 imgsz 的 recipe 会在校验阶段被拒绝"。**不得**为让文档自洽而删除 action catalog 条目或放宽 recipe 校验。

## P2 Issues

### P2-1 runtime preflight 与 training release 缺少正式用户文档
- **source**: `yolo_agent/cli.py:390-414`（`papers runtime-preflight` 命令）、`yolo_agent/core/executor.py:2159-2175`（`training_release_not_verified` 强制门禁）、`yolo_agent/research/training_release.py`
- **actual evidence**: `docs/` 下仅 `docs/PRETRAINING_ACCEPTANCE.md:18,62` 提到 runtime preflight 与 training release（验收摘要视角）；`docs/cli.md` 未收录 `papers runtime-preflight`；用户视角的"训练发布门禁如何工作、何时需要重建"没有 owner 文档。
- **status**: missing_from_docs
- **recommended owner document**: `docs/cli.md`（命令收录）或新增 `docs/training-release.md`

### P2-2 bounded HPO 无专述文档
- **source**: `yolo_agent/agents/bounded_hpo.py:1,23,60,84`（`HpoScope = asha_train_time|eval_route|forbidden`）、`configs/training_recipes.yaml:2-4`（`enable_scalar_hpo: false`）、`configs/loop_policy.yaml:4`
- **actual evidence**: `docs/capability-maturity.md:49` 仅把 `bounded_hpo.py` 列为证据源文件；`docs/paper-auto-optimization-certification.md:22` 提到 "disables scalar HPO"。当前真实能力是"存在 bounded HPO（仅限 ActionSpec 声明的 search space），scalar HPO 显式禁用"，这一区分没有用户文档解释。
- **status**: missing_from_docs
- **recommended owner document**: `docs/training-modes.md` 或 `docs/concepts.md`

### P2-3 `doctor` 命令可用但对用户隐藏
- **source**: `yolo_agent/cli.py:159-164`（`USER_COMMANDS = (setup, train, status, stop)`）、`:180`（metavar 只显示这 4 个）、`:1083-1109`（doctor parser 实际注册且可调用）
- **actual evidence**: `docs/coco-yolo26.md:31` 使用 `yolo-agent doctor --data ... --model ...`，命令真实有效；但 `--help` 的 metavar 不显示 doctor，新用户从 help 无法发现它。
- **status**: ambiguous（文档用法有效，但发现性差）
- **recommended owner document**: `docs/cli.md`（如属有意隐藏，加一句说明 doctor 属内部/兼容入口）

### P2-4 README 中 `artifacts/executable_portfolio.yaml` 为运行时产物，首次运行前不存在
- **source**: `README.md:38`（"Each automatic round reports the full search funnel in `artifacts/executable_portfolio.yaml`"）
- **actual evidence**: 写入路径存在于代码（`yolo_agent/agents/auto_optimization_loop.py`、读取 `yolo_agent/cli.py:4001`），本地尚未产生该文件（截至审计日无完成轮次）。claim 本身正确（runtime behavior），仅提示读者不要在运行前寻找该文件。
- **status**: correct（附说明即可，无需改）
- **recommended owner document**: `README.md`（可选：加 "created by the first automatic round"）

### P2-5 能力成熟度表的三处呈现为"单一源生成"，非重复真相（记录在案）
- **source**: `configs/capability_maturity.yaml` → `yolo_agent/tools/capability_matrix.py:36-39,58-120` → `README.md:157-168`、`README.zh-CN.md:136-147`、`docs/capability-maturity.md:10-19`
- **actual evidence**: 三处表格逐项一致，且由同一 manifest 生成；`docs/capability-maturity.md:3-4` 明确禁止手编表格。审计确认无 drift。
- **status**: correct（single source of truth，机制健康）

## Bilingual Drift（EN vs ZH）

| # | 位置 | 差异 | 严重度 |
| --- | --- | --- | --- |
| 1 | README 文档目录（EN :170-191 vs ZH :149-166） | ZH 缺 4 条目（见 P1-2） | P1 |
| 2 | README "Documentation" 之外的正文 | 逐段核对：命令示例、coverage 表、maturity 表、边界语句均语义一致；ZH :145 的四档裁决枚举比 EN :166 更详细（补充性，非冲突） | 无问题 |

## Command Mismatches（文档命令 vs CLI parser）

核对范围：`docs/cli.md`（:12,20,28,39-42,55,63,73-75,81,103-119,125,155,163）、`docs/quickstart.md`（:22,36,46,58,61,64,67,96,108）、`README.md`（:19-28,82-91,104）、`README.zh-CN.md`（:19-28,64-73,83）、`docs/coco-yolo26.md:31,48,62`、`docs/training-modes.md`、`docs/troubleshooting.md`、`docs/install.md`。

| # | 文档位置 | 问题 | 判定 |
| --- | --- | --- | --- |
| 1 | `docs/quickstart.md:25` | "optimize 命令"叙述过时（见 P0-1） | wrong |
| 2 | 其余全部命令示例 | 子命令/flag/取值与 `yolo_agent/cli.py` parser 一致（`--target-metric` 合法值 `map50_95,map50,ap_small,ap_medium,ap_large,precision,recall,f1` 来自 `core/optimization_objective.py:28-37`；`--goal` 正则 `optimization_objective.py:135` 接受 `+2map`/`+0.02map50_95`/`+2ppmap50`/`+2%map`；install extras 与 `pyproject.toml:20-37` 一致） | correct |
| 3 | Bash `\` 续行符 | 全文检索 0 命中，所有示例为单行 PowerShell 写法 | 无问题 |

## Architecture Drift 与混淆项核查（任务清单 A-L）

| 项 | 结论 | 关键证据 |
| --- | --- | --- |
| A. README paper/component 统计 | **correct**：728/158/103/0/0 与 `docs/paper-adapter-coverage.yaml`（snapshot_hash `c606d6c5...`）完全一致；85/23/83 与 `docs/paper-coverage-acceptance.yaml`（report_hash `797c3b91...`，separate_detector_family 实数 168、insufficient_information 实数 475）一致 | `README.md:133-153`；生成器 `yolo_agent/tools/capability_matrix.py:38` |
| B. 83/85 certified adapter vs Paper-83 campaign | **未混淆**：README 明示 "83/728 (11.4%); reusable component adaptation, not exact paper reproduction"；`all-83-*` 系列文档语境为 83-paper manifest（`configs/research/paper_83_manifest.yaml`），两个 "83" 语境独立 | `README.md:147-151` |
| C. exact reproduction vs implementation-ready | **区分清晰**：exact reproduction=0（README :151）；implementation-ready=83/83 且 `REAL TRAINING EXECUTED: NO`（`docs/PRETRAINING_ACCEPTANCE.md`）；`artifacts/paper_83_runtime_preflight.yaml` `real_training_executed: false`；`artifacts/training_release_v1.yaml` `READY_FOR_FIRST_TRAINING`（未冒充已复现） | artifacts 三件套 |
| D. capability maturity 日期/状态 | 状态与代码一致；日期过期（见 P1-3） | `configs/capability_maturity.yaml` |
| E. error facts / error-delta 真实能力 | **implemented**：`core/detection_error_profile.py:161`、`core/detection_error_delta.py:91,138-140`、消费链 `core/error_round_decision.py:92` 等；README 标 `incomplete`/`partial` 与"gate 存在但逐 run 验证"的边界描述相符 | core/* 模块族 |
| F. bounded HPO 是否存在 | **存在**（`agents/bounded_hpo.py`），scalar HPO disabled（configs 双处 + `auto_optimization_loop.py:3982-3999`）；文档无冲突，仅缺专述（见 P2-2） | 代码+configs |
| G. imgsz=640 invariant vs resolution action | 张力存在，见 P1-4 | catalog :227-267 vs recipes/schemas.py:79-108 |
| H. 3-seed confirmation 实现状态 | **implemented & executable**：`agents/seed_confirmation.py`（seeds 42/43/44）、`agents/confirmation_statistics.py`（配对 95% Student-t CI）、`agents/promotion_rule.py:36`（CONFIRMED/POSSIBLE/REJECTED/INCONCLUSIVE 四档）；README/docs 标 `executable`+`需显式确认` 与代码相符 | agents/* 模块族 |
| I. preflight/acceptance/release 正式文档 | 部分缺失，见 P2-1 | `docs/PRETRAINING_ACCEPTANCE.md` 仅覆盖验收摘要 |
| J. CLI 文档 vs parser | 除 P0-1 叙述外全部一致 | `yolo_agent/cli.py:159-180,520-560,1083-1218` |
| K. README EN/ZH 一致性 | 目录缺 4 条目（P1-2），正文一致 | 见上表 |
| L. PowerShell 示例有效性 | 全部单行、flag 真实；无 `\` 续行问题 | 检索 0 命中 |
| 已知疑点 `E:\datatset` | **确认是 typo**（27 处/10 文件，README 用 `E:\dataset`），见 P1-1 | 检索命中列表 |
| 已知疑点 "setup 生成下一条 optimize 命令" | **确认过时**，见 P0-1 | `setup_wizard.py:96,202,213-219` |

## Verified-Correct Claims（抽样清单，机器可读版含完整字段）

1. 四个用户命令 `setup/train/status/stop` 及其 flags 与 parser 一致（`cli.py:159-164,1111-1233`）。
2. `--goal` 语法与文档四种写法全部合法（`optimization_objective.py:135-139`）。
3. `--target-metric` 取值表与文档一致（`optimization_objective.py:28-37`）。
4. 隐藏 advanced/research/papers 命名空间下的 `certify-*`、`import-awesome`、`build-snapshot`、`runtime-preflight` 命令均真实存在（`cli.py:520-560,390-414,7030-7418`）。
5. TaskSpec（`core/task_spec.py:76`）、DetectionErrorProfile（`core/detection_error_profile.py:161`）、DetectionErrorDelta（`core/detection_error_delta.py:91`）、ErrorDecisionTrace（`core/error_decision_trace.py:88`）、ActionSpec（`agents/action_space_schemas.py:149`）全部实现并被 runtime 消费——`docs/concepts.md`、`docs/evidence.md` 对应描述成立。
6. ASHA 队列控制 `executable`（`agents/asha_scheduler.py`，四 rung、confirmation_seeds [42,43,44]）。
7. PolicyMemory（`core/policy_memory.py:288` PolicyMemoryStore，写 `runs/policy_memory.jsonl`）——README 决策流程 "policy-memory update" 成立。
8. 推理隔离策略 5 组件契约 + adapter（`configs/components/inference/isolated_policies.yaml`、`components/adapters/inference/plugin.py`）——`docs/inference-policy-adapters.md` 成立。
9. Paper Intelligence 离线导入/冻结快照（`cli.py:520-560`，训练不联网）——README 描述成立。
10. 成熟度链 10 级（`docs/capability-maturity.md:25`）与 `components/maturity.py` 机制一致。
11. 安装文档 extras（train/certification/dev）与 `pyproject.toml:20-37` 一致。
12. 文档目录链接无死链（所有被引用 docs 文件均存在，反向：EN 目录 20 项、ZH 16 项全部可点开）。

## Method Note

- 所有结论基于工作区实读（代码 `yolo_agent/`、configs、artifacts、`yolo_agent/cli.py` parser 定义），未根据旧文档推断。
- 本审计未修改任何 runtime 代码、configs、既有文档；新增文件仅 `docs/DOCUMENTATION_AUDIT.md` 与 `artifacts/documentation_audit.json`。
- 未运行真实训练；artifacts 的状态字段（如 `real_training_executed: false`）按原样引用。
