# Grilling M1 & M2

快问快答格式记录 2026-07 对 M1（数据准备）和 M2（baseline testing）的 grill 过程与结论。
grill 的目的不是找错，是**把心智模型 explicit 化并跟事实对齐**——面试和 blog 都要靠这些论证。

---

## 元问题

### Q: 这次 grill 具体走了哪几支？

**9 支都走完了**，其中 3 支落地为代码改动、6 支保持现状但记录了推理：

| # | 主题 | 结论 | 落地 |
|---|---|---|---|
| Q1 | **数据切分比例** | 50/50 → **50 train / 114 eval** | 代码改动 |
| Q2 | **`max_new_tokens=256` 够不够** | → **512** | 代码改动 |
| Q3 | **base Llama-2 到底在做什么** | 观察结果写入 `docs/baseline_behavior.md` | 文档新增 |
| Q4 | **STOP_SEQUENCES 过截风险** | 保持现状（实证过截率为 0） | 无 |
| Q5 | **pad_token_id = eos_token_id** | 保持现状（单条生成时无害） | 无 |
| Q6 | **单条 vs Batched generation** | 保持单条 | 无 |
| Q7 | **Base vs Chat 版做 baseline** | 保持 base | 无 |
| Q8 | **随机种子 42 的具体 shuffle 是否有偏** | 推荐做长度分布检查，**推后**——如 M4 结果反常再回查 | 无（暂缓） |
| Q9 | **能否跳过 baseline 完整重跑** | 不跳，完整重跑 | 无 |

### Q: 这次 grill 落到哪一个 commit？

`81833c6 M2: 扩容评估集到 114 + max_new_tokens=512 + 忽略 .env`

（原本是两个 commit：`2b5a21f` 含泄露 `.env`、`5baff87` 删除 .env。GitHub push protection 拦截，用 `git reset --soft` 合并成一个干净 commit，dangling 老 commits 在 reflog 里 90 天后 GC。）

---

## Q1 · 数据切分：50/50 → 50/114

### Q: 原本的 50/50 切分是怎么来的？

`prepare_data.py` 里 `TRAIN_SIZE = 50, EVAL_SIZE = 50`。README Decision 3 论证过训练侧选 50 是够的（"够 credible、够 VRAM budget、迭代快"），但**评估侧选 50 从没被单独论证过**——是个默认对称值。

### Q: 50/50 的问题在哪？

**评估集统计功效不足**。pass@1 是二项分布的估计量，标准误 `SE = √(p(1-p)/N)`。N=50、p=0.13 时 SE ≈ 4.8pp，**95% CI 半宽约 ±10pp**。这意味着 baseline 13% 观测值的真实值 95% 概率落在 [3%, 23%]。M4 fine-tuned 观测 22%，两个 CI 大幅重叠，**不能宣称微调有效**。

术语（pp、SE、CI）查 `docs/glossary.md` 统计部分。

### Q: 为什么最终改成 50/114 而不是其他配比？

**HumanEval 一共 164 题**，四方案对比：

| 方案 | Train | Eval | 严谨度 | 工作量 |
|---|---|---|---|---|
| A. 现状 | 50 | 50 | 低 | 已完成 |
| **B. 选择这个** | **50** | **114**（所有非训练题） | 中 | 改一个数字 |
| C. 跨数据集 | MBPP 374 | HumanEval 164 | 高（学术标准） | 数据 pipeline 重写 |
| D. Magicoder / HumanEval | OSS-Instruct 75K | HumanEval 164 | 最高 | 更大数据 + 更多 GPU 时间 |

选 B 的理由：训练侧不变（微调稳定性不受影响）、评估侧 SE 从 4.8pp → 2.9pp、改动最小、MVP 定位不追 SOTA。

### Q: 传统学术做法是什么？为什么不做？

**主流做法（Codex、CodeLlama、DeepSeek-Coder 等论文）**：HumanEval 全 164 题当**held-out eval**，训练用其他数据集（MBPP、CodeAlpaca-20k、magicoder-oss-instruct 等）。

不做原因：**learn-by-building 项目，跨数据集 pipeline 工作量太大**，且 MVP 目标就是"展示微调机制"而不是"追 SOTA 分数"。会在 blog 里主动 disclose 局限性。

### Q: 换到 50/114 之后，微调收益需要多大才能"跳出噪声"？

`SE_diff = √(SE_A² + SE_B²)`。N=114、假设 baseline p=0.13、finetuned p=0.22：
- SE_diff ≈ 4.9pp
- **95% CI of diff ≈ ±9.8pp**

观测到的 9pp 提升**刚好在显著性边缘**——比 N=50 时（需要 >15pp 才算显著）严格多了。M4 要冲更大 gap 才安全。

---

## Q2 · `max_new_tokens`：256 → 512

### Q: 256 是怎么来的？

`eval/generate_completions.py`（当时叫 `generate_baseline.py`）里 `MAX_NEW_TOKENS = 256`，没经过实证支撑，是常见默认值。

### Q: 有什么问题？

`eval_50.jsonl` canonical solution 长度分布（`chars/4` 粗估 tokens）：

| 分位 | chars | lines | ~tokens |
|---|---|---|---|
| p50 | 179 | 7 | ~44 |
| p90 | 338 | 15 | ~84 |
| p95 | 516 | 17 | ~129 |
| **max** | **864** | **32** | **~216** |

最长的 `HumanEval/81` 估算 ~216 tokens，跟 256 上限只差 40 tokens 缓冲。`chars/4` 是乐观估计——**大概率触顶被截**。

### Q: 为什么最终选 512 而不是 384 / 1024？

- 384（p95 3x）：max 案例仍可能触边
- **512**：max 之上有 100+ tokens 余量，HumanEval 论文里普遍用 512-1024
- 1024：极安全但生成时间翻倍无收益

生成时间约 2x（10-20 min on T4），Colab 免费档完全够。

---

## Q3 · Base Llama-2 到底在做什么

### Q: 上一次 Colab 跑出来什么样的 completion？

- `HumanEval/83`：硬编码 `n ∈ {1..6}` 查表
- `HumanEval/69`：变量叫 `max_freq` 但算的是最大元素——语义误读
- `HumanEval/131`：硬编码错误常数

50 条 completion **全部非空、结构上都是"函数体样式的代码"**。

### Q: 这个观察证明了什么？

**Base Llama-2-7B 在 HumanEval 上不是"不会写代码"，是"写但常错"**。它知道函数体的**形状**，但不理解任务的**语义**。这**不是 bug，是 base 模型典型行为**。

### Q: 对 M4 的故事意味着什么？

**微调收益预期明确**：50 条 canonical 都是真实算法，微调后模型应减少硬编码、更忠实读 docstring、更干净地结束函数。

Blog 的核心叙事："**Baseline 不是不会写代码，是写得浅。微调教的不是新语法，是怎么读 spec**"。

### Q: 这个观察落到哪份文档？

`docs/baseline_behavior.md`——完整样例、机制解释、blog 素材、免责事项。

---

## Q4 · STOP_SEQUENCES 过截风险

### Q: 现有 STOP_SEQUENCES 是什么？

`["\ndef ", "\nclass ", "\nif __name__", "\nprint(", "\n#", "\n@"]`——post-hoc 后处理裁剪，找到第一个匹配的 stop 序列，把它及之后的内容全部丢掉。

### Q: 过截风险实证结果？

对 114 条 canonical solution 扫描每个 stop 序列，**全部命中数为 0**。原因：canonical 都是**函数体**（每行以 4 空格缩进开头），而 stop 序列要求 `\n` 后紧跟 column-0 字符——**几何上不可能匹配到 canonical 内部**。

### Q: 反过来看，欠截风险呢？

模型可能在**当前缩进层级内**一直 ramble，`\n    # unused case`（有前导空格）不会触发 `\n#`。可能后果：多数情况下追加代码是死代码（第一个 return 已退出）→ 无害；少数情况可能语法错。

### Q: 为什么保持现状？

- 过截风险实证为 0
- 欠截风险大概率无害（Python 死代码不影响 return path）
- 贴近社区实践（Codex 官方 evaluation 也是这些 pattern）
- 改动的边际收益低、风险高
- 真跑一次 baseline 之后再针对性调整（更强证据支撑）

---

## Q5 · `pad_token_id = eos_token_id` 正确吗？

### Q: 为什么这么写？

Llama tokenizer 默认没有 `pad_token`，这是消 transformers warning 的常见 workaround。

### Q: 为什么无害？

**单条生成时（当前）根本没有 padding 需要**，这个设置完全是形式代码。真要做 batched generation 时，需要显式加 `attention_mask` 让 padding 位不影响 logits——但那不是现在的情况（见 Q6）。

---

## Q6 · 单条 vs Batched Generation

### Q: 现在慢在哪？

for 循环单条生成，T4 上一题约 5-10s，114 题总时长 10-20 分钟。

### Q: Batched 能省多少？

Batch=4 理论 3-4x 加速，省约 10 分钟。

### Q: 为什么不做？

代价太高：padding 开销、VRAM × 4、写 50 行代码 + 边界情况验证（tokenizer padding、attention mask、per-item stop detection）。**省 10 分钟不值得**。M5 消融阶段如果要跑很多次再回来。

---

## Q7 · Base 模型 vs Chat 版做 baseline

### Q: 为什么不用 chat 版？

| 维度 | Base（当前） | Chat |
|---|---|---|
| 预期 pass@1 | ~13% | ~20% |
| 需要 prompt format | 原样 | wrap 成 chat template |
| 微调空间 | **大** | 小 |
| 故事对比度 | **强**（浅 → 深） | 弱（好 → 更好） |

保持 base：**base 模型的行为差距正是 blog 故事的核心——不是"更强"，是"更懂 spec"**。

---

## Q8 · 随机种子 42 的具体 shuffle 是否有偏？

### Q: 风险是什么？

seed=42 的具体切分**可能碰巧**把简单题都塞进 train / 难题留给 eval，导致：训练太容易 + baseline 虚低 + M4 增益虚高。HumanEval 难度长尾（1 行 lambda ~ 30 行算法），N=50 抽样时**单一 seed 的偏差可能 ±3-5pp**。

### Q: 严谨做法有哪些？

1. **实测 canonical 长度分布**在 train/eval 两边是否对齐（免费）
2. **多 seed 平均**：跑 3-5 个 seed 的完整 pipeline，报平均（GPU 时间 5x，贵）
3. **Stratified split**：按难度分层——需要难度标签，HumanEval 官方没给

### Q: 为什么推后而不是立即做方案 1？

推后到 M4 结果观察阶段：如果 M4 → M2 gap 反常（过大或过小），回来做分布检查作为诊断。如果 gap 落在合理范围（+10-20pp），单一 seed 偏差影响可控，**在 blog 里主动 disclose 就行**。

**Debt 项**：M4 完成后如果反常，用 `docs/scripts_tmp/check_lens.py` 类似的脚本检查 train/eval 的 canonical 长度分布是否对齐。

---

## Q9 · 能否跳过 baseline 完整重跑？

### Q: 为什么想跳？

上次已经跑过一次（旧配置 50/50 + 256 tokens），要省 20 分钟 Colab 时间。

### Q: 三条思路对比

- **完全跳过**：不行——**没有 baseline 数字，微调再厉害也讲不出"改进"的故事**
- **只补跑新增的 64 题**：旧配置 completion 是 256 tokens 生成，配置不一致不能拼；且旧配置本身可能有截断问题
- **完整重跑**：20 分钟，Colab 免费额度绰绰有余，**推荐**

### Q: 结论

完整重跑。**省 20 分钟不值得引入解释歧义**。

---

## Grill 之后仍留的债务项

1. **Q8 的分布检查**：M4 结果反常时的诊断手段
2. **欠截失败案例分析**：Baseline 重跑后，挑 fail 的题人工看 completion，判断是模型能力 vs stop 规则不够——可以进 `docs/baseline_behavior.md` 作为 blog 素材
3. **难度对齐 vs 抽样质量**：整个 seed 单点问题的更根本方案是多 seed 平均，但 GPU 时间预算不允许

---

## 这次 grill 改动的项目状态

**代码 / 配置改动（`81833c6`）**：
- `data/scripts/prepare_data.py`：`EVAL_SIZE=50→114`、输出文件名 `eval_50→eval_114`
- `eval/generate_completions.py`（当时叫 `generate_baseline.py`）：`MAX_NEW_TOKENS=256→512`、`--eval-set` 默认路径
- `eval/tests/test_scoring.py`：`EVAL_SET_PATH` 同步
- `notebooks/run_baseline_colab.ipynb`：source cells 里 5 处 pattern
- `CLAUDE.md`：Status 段更新 + working conventions 加"文档用中文"
- `.gitignore`：加 `.env` 系列（防 secret 泄露）

**新增数据**：`data/processed/eval_114.jsonl`（114 题）

**删除数据**：`eval_50.jsonl`、`baseline_generations.jsonl`（旧配置产物作废）

**新增 docs**：`docs/glossary.md`、`docs/baseline_behavior.md`、`docs/grilling_m1_m2.md`（本文档）

**验证**：`pytest eval/tests/` 3/3 pass，包括 114 canonical 全部通过自测

**待 push**：本地 clean，`81833c6` 的父提交就是 remote 的 `e0537c4`——**普通 fast-forward `git push origin dev/t`**，不用 force。

---

## 交叉引用
- 术语：`docs/glossary.md`
- Baseline 观察：`docs/baseline_behavior.md`
- 设计决策：`README.md`
- 项目状态：`CLAUDE.md`
