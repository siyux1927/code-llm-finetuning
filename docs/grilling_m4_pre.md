# Grilling M4 Pre-Development

快问快答格式记录 **M4（Fine-tuned Evaluation）开发前**的 8 支决策与理由。跟 M3 pre-grill 一样是**事前**——写代码前拍板。

M4 的核心洞察：它不是一个新模块，是 **"M2 generation + adapter 加载" 的参数化组合**。所以 grill 的重点是**契约**（如何最小改动让 M2 脚本兼职 M4）而不是新算法。

---

## 元问题

### Q: 这份 grill 决定了什么？

| # | 主题 | 决定 |
|---|---|---|
| Q1 | 生成脚本改造契约 | 重命名 `generate_baseline.py` → `generate_completions.py`，加 `--adapter` 参数 |
| Q2 | Adapter 加载策略 | `PeftModel.from_pretrained`，**不 merge** |
| Q3 | 生成参数 | 跟 M2 完全一致（greedy / 512 tokens / 同 STOP_SEQUENCES） |
| Q4 | 补 scoring CLI | 写 `eval/score.py` |
| Q5 | 对比分析脚本 | **现在写** `eval/compare_results.py` |
| Q6 | Colab notebook | 分开：新写 `run_finetuned_colab.ipynb` |
| Q7 | Pass@1 在 notebook 里算掉 | 是（M4 + 回补 M2） |
| Q8 | Adapter 缺失的 defensive check | 加 |

---

## Q1 · 生成脚本改造契约

### Q: 为什么改造 M2 脚本而不是新写？

- 生成逻辑 90% 一致（同 base 模型、同 quantization、同 decoding、同 stops）
- 两份代码要同步维护——任何一处不一致都可能引入 M2 vs M4 对比的 confound
- 单一脚本用参数分支更符合"M4 是 M2 的扩展"这个概念模型

### Q: 契约细节？

| 项目 | 值 |
|---|---|
| 新脚本名 | `eval/generate_completions.py`（原 `generate_baseline.py` 重命名） |
| 新增参数 | `--adapter <path>`（可选） |
| 语义 | 不传 → baseline (M2)；传 → fine-tuned (M4) |
| Output 默认命名 | 不传 → `baseline_generations.jsonl`；传 → `finetuned_generations.jsonl` |
| Output 覆盖 | `--output <path>` 明确指定 |

### Q: 为什么"中性"命名而不是 `generate_baseline` 或 `generate_finetuned`？

单一脚本承担两种角色——命名要反映实际行为（生成 completions），角色由参数决定。类比 `git commit` 而不是 `git commit-new` / `git commit-amend`。

---

## Q2 · Adapter 加载策略

### Q: 用什么 API？

`PeftModel.from_pretrained(base_model, adapter_path)`。标准 PEFT API。base_model 保持 4-bit 量化。

### Q: 要不要 `merge_and_unload()`？

**不要**。

| 选项 | 推理速度 | VRAM peak | 复杂度 |
|---|---|---|---|
| **不 merge** | 每 forward 多 ~0.1ms | 略省 | +0 行 |
| Merge | 略快 | 短暂 peak | +1 步、多点风险 |

理由：114 题总差异微乎其微；merge 后无法回退调试；MVP 保持最少步骤；M5 部署阶段再考虑 merge。

### Q: 加载完要做什么？

`model.eval()` 切到 evaluation 模式（disable dropout）。跟 M2 baseline 一致。

---

## Q3 · 生成参数（Non-negotiable）

### Q: 用什么参数？

跟 M2 baseline **一模一样**：
- 4-bit NF4 + fp16 compute（BitsAndBytesConfig 完全复用）
- Greedy decoding (`do_sample=False`)
- `max_new_tokens=512`
- STOP_SEQUENCES 完全相同

### Q: 为什么这一支没得选？

任何一处不同都会让 M4 vs M2 的差异**掺杂 confound**——你没法回答"这 5pp 提升是微调带来的还是解码方式变了带来的"。**统一是唯一正确答案**。

---

## Q4 · Scoring CLI

### Q: 为什么现在补？

`eval/scoring.py` 目前只是 library。M2 grill 就发现了这个坑但没修——原因是"手工 REPL 也能跑"。**M4 阶段这个决定该翻案**：

- M4 每次跑完都要算 pass@1（+ 立刻看 baseline 对比）
- Compare 脚本（见 Q5）需要 import scoring
- 复现实验 / blog demo 都需要一行命令

### Q: 输入输出契约？

```
python eval/score.py \
    --generations <path-to-generations.jsonl> \
    --eval-set data/processed/eval_114.jsonl

# 输出：
# pass@1 = 0.1316 (15/114)
# 95% CI: ±6.2pp (SE 3.15pp, N=114)
```

顺便报**置信区间**——用户看到数字时就直接看到不确定性范围，避免"12% → 13% 就是提升"这种误读。

### Q: 可选输出？

`--per-problem-output <path>` 把每题的 pass/fail + error 写成 jsonl，供后续错误分析用（不是必须的，但成本几乎为 0）。

---

## Q5 · 对比分析（M4 vs M2）

### Q: 为什么现在建脚手架？

Blog 需要的**不只是**两个数字：
- 哪些题从 fail → pass（"微调真的教会了什么"）
- 哪些题从 pass → fail（**回归**——微调伤了原本能力的信号）
- Side-by-side 样例（blog 素材）
- 统计显著性判断（观察差异是不是脱离噪声）

拿到结果再手工分析可以做，但**打断心流**——先分析 → 发现需要个工具 → 停下来写工具 → 回来继续。**现在写更省事**。

### Q: 输入输出契约？

```
python eval/compare_results.py \
    --baseline data/processed/baseline_generations.jsonl \
    --finetuned data/processed/finetuned_generations.jsonl \
    --eval-set data/processed/eval_114.jsonl

# 输出（示意）：
# =====================================================
# Pass@1 comparison
# =====================================================
# Baseline:    13.16%  (SE ±3.15pp, N=114)
# Fine-tuned:  24.56%  (SE ±4.03pp, N=114)
# Delta:      +11.40pp  (95% CI ±10.04pp)
# Significant at 95%? YES
#
# =====================================================
# Flipped problems
# =====================================================
# Base FAIL → Tuned PASS: 15 problems
# Base PASS → Tuned FAIL: 2 problems
#
# --- Sample of 5 'fail → pass' ---
# [HumanEval/xx]
#   Baseline completion (first 200 chars):
#     ...if n == 1: return 1 / elif n == 2: return 2...
#   Fine-tuned completion (first 200 chars):
#     ...def helper(x): return sum(range(x)) / return helper(n)...
```

### Q: 显著性怎么判？

`SE_diff = √(SE_baseline² + SE_finetuned²)`，95% CI half-width = 1.96 × SE_diff。观测 delta 超过 CI half-width → 显著。跟 M1/M2 grill Q1 里的公式一致。

### Q: Side-by-side 样例数量？

**默认 5 组**（`--n-samples 5`），可覆盖。太少不够素材；太多打印太长看不完。

---

## Q6 · Colab notebook：分开

### Q: 为什么分开而不合并？

| 选项 | 优 | 劣 |
|---|---|---|
| **分开**：`run_baseline_colab.ipynb` (M2) + `run_finetuned_colab.ipynb` (M4) | 各自故事线清晰、checklist / 时间预估不同 | 有点模板重复 |
| 合并：一个 notebook 参数切换 | DRY | UI 混乱——用户"我这次要跑 M2 还是 M4？" |

**教学项目/blog demo 场景下清晰 > DRY**。

---

## Q7 · Pass@1 直接在 Colab 里算

### Q: 加什么 cell？

```
!python eval/score.py \
    --generations data/processed/finetuned_generations.jsonl \
    --eval-set data/processed/eval_114.jsonl
```

顺手加一行——用户在 Colab 里立即看到数字，不必等下载。

### Q: 为什么也回补 M2 notebook？

M2 notebook 之前没算 pass@1 是因为当时没有 scoring CLI。**现在有了**，把这个坑填掉。不然用户跑完 M2 baseline 还是不知道数字。

对称性让**两个 notebook 都是"跑完就直接看到结果"**——比"M2 要下载到本地才能算、M4 直接算"友好。

### Q: M4 notebook 还多算什么？

除了 pass@1，还跑一次 `compare_results.py`——用户直接看到 M2 vs M4 对比 + 显著性判断 + flipped 样例。**一个 notebook 结束时就是完整的 blog Part 2 素材**。

---

## Q8 · Defensive check：adapter 找不到

### Q: 为什么加这个 check？

用户容易踩的坑：
- 忘了先跑 M3 就跑 M4
- clone 时 `models/lora_adapter/` 没拉全
- 路径写错

**默默让 PEFT 报一个不清楚的错**很糟——用户看到 "config.json not found" 一头雾水。**主动检查 + 友好消息**成本几乎为 0。

### Q: 检查什么？

```python
if args.adapter and not (args.adapter / "adapter_config.json").exists():
    raise SystemExit(
        f"[m4] Adapter not found at {args.adapter}. "
        f"Did M3 training finish and models/lora_adapter/ get pulled?"
    )
```

只查 `adapter_config.json`——它是 PEFT 保存的核心元数据文件，adapter_model.safetensors 之类的会一起存在。

---

## M4 完成后的下一步

- **成功路径**：M4 pass@1 - M2 pass@1 > 9pp（显著）→ 微调有效 → 写 blog Part 2
- **不显著路径**：Delta 落在噪声区间 → 检查 M3 train loss 是否收敛、检查 M8 债务的 seed 偏差
- **回归严重路径**：Base PASS → Tuned FAIL 数 > 3-5 → 微调伤了基础能力 → 减 epochs 重训 M3
- **进入 Phase 2**：M4 完成 = MVP 完成 → M5（quantization）/ M6（inference）/ M7（cost）

---

## 交叉引用
- 术语：`docs/glossary.md`
- M1/M2 grill：`docs/grilling_m1_m2.md`
- M3 pre-grill：`docs/grilling_m3_pre.md`
- Baseline 行为：`docs/baseline_behavior.md`
- 设计决策：`README.md`
- 项目状态：`CLAUDE.md`
