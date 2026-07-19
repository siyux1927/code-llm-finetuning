# Llama-2-7B 在 HumanEval 上的 Baseline 行为

快问快答格式的观察实录。写在 M4 之前，作为解读 M2→M4 差异的参考，也作为 blog 的原料。

---

## Q: 我们用什么模型 + 什么设置跑 baseline？

首次 Colab 运行（50 题、`max_new_tokens=256`）的完整设置：

| 项目 | 值 |
|---|---|
| 模型 | `meta-llama/Llama-2-7b-hf`（**base**，非 chat/instruct） |
| 量化 | 4-bit NF4，fp16 compute（bitsandbytes） |
| 解码 | Greedy（`do_sample=False`），符合 pass@1 协议 |
| max_new_tokens | 256（首次）→ 512（`81833c6` 之后） |
| Stop 规则 | `\ndef `、`\nclass `、`\nif __name__`、`\nprint(`、`\n#`、`\n@` |
| Prompt 格式 | 原始 HumanEval prompt（函数签名 + docstring），不做 instruction wrapping |

术语查 `docs/glossary.md`。

## Q: 为什么这个设置对故事很重要？

因为 M2 和 M4 **共享同一个 base capability 起点** —— 一开始就用 base 模型（不是 chat/instruct）+ 4-bit 量化，M4 vs M2 对比就能把"微调有没有改变行为"跟"量化是不是伤了性能"、"instruction tuning 是不是有帮助"**隔离开**。差异纯粹归因于**微调**。

## Q: base Llama-2 拿到 raw HumanEval prompt 会做什么？

它不是**"配合"**——它是**在猜下一个 token**。理论上可能：继续写更多 docstring、开始下一个 `def`、输出 `>>> example`、复读 prompt……

**经验事实**：首次 Colab 运行 50 条 completion **全部非空、结构上都是"函数体样式的代码"**。这证明"函数签名 + docstring → 函数体"在预训练语料里是极强的 pattern，模型自然会往这个坑跳。

代价：**结构对了，语义常错**。以下是两类失败模式。

---

## Q: 观察到的第一种失败模式是什么？（**硬编码**）

模型输出 `if x == a: return v1 / elif x == b: return v2 / ...`——只对少数具体输入枚举，背后**没有通用算法**。

**例子——`HumanEval/83`**：
```python
if n == 1:
    return 1
elif n == 2:
    return 2
elif n == 3:
    return 3
elif n == 4:
    return 4
elif n == 5:
    return 5
elif n == 6:
    return 6
```
纯查表，`n ∈ {1..6}` 之外 fall through 返回 `None`。测试用大 n 会 0 分。

**例子——`HumanEval/131`**：同样形状，对 `n ∈ {0, 负数, 1, 2, 3, 4}` 硬编码错误的常数（`n==3 → 6`、`n==4 → 0`）。

## Q: 为什么模型会硬编码？

预训练语料（StackOverflow、教程、不完整 gist）里包含大量**枚举 case 的玩具代码**。模型对"通用形式"没把握时，会 fall back 到**统计上邻近的模式**——对短函数来说，通常就是 case 枚举。

**不是 bug，是 base 模型典型行为**。

---

## Q: 第二种失败模式是什么？（**语义误读**）

模型读了 docstring，但把变量/操作绑定到**错误的含义**上。**语法层面像模像样，语义层面是错的**。

**例子——`HumanEval/69`**（任务：返回大于 0 的最大整数，且其**出现次数**不少于其**值**）：
```python
if len(lst) == 0:
    return -1
max_freq = 0
for i in lst:
    if i > max_freq:
        max_freq = i
if max_freq == 0:
    return -1
return max_freq
```

变量叫 `max_freq`，逻辑算的却是**列表里的最大元素**，不是**出现频率**。模型识别出了"找一个最大"的形状，但绑定到了错误的量上。

## Q: 为什么会语义误读？

base LLM 的**语法 pattern-matching 还不错，但短规格上的语义推理很浅**。docstring 用领域术语（frequency、count、occurrence）时，模型常锚定在第一个类"max"的短语上，跳过后面的限定。

---

## Q: 还有哪些预期但首次未观察到的模式？

- **Docstring 续写**：继续写更多 docstring 而不是函数体
- **跑题**：写 `>>> example` 交互式例子（多半会被 `STOP_SEQUENCES` 截住）
- **Prompt 回声**：重复函数签名 / 复读 docstring 一部分

都是 base 模型 completion 任务的已知失败模式。是否出现取决于 shuffle 选中了哪 114 道题——扩容重跑后再补录。

---

## Q: 微调应该改变什么？

50 条训练样本都是 canonical solutions（真实的算法实现）。微调应该把模型推向：

1. **偏好通用算法而非枚举**——看到 50 个"签名 + docstring → 递归/迭代/列表推导 函数体"的正例后，模型下一 token 的分布会远离 `if n==1 elif n==2 ...` 模板
2. **更忠实地绑定 docstring 名词**——不是学 Python（早会），是内化"这类任务里 docstring 的歧义词按严格 CS 含义读"
3. **干净地终止函数体**——减少需要 `STOP_SEQUENCES` 救场的跑飞生成

## Q: Blog 里想讲出来的核心一句话？

> Baseline Llama-2-7B 在 HumanEval 上不是"坏"，是"浅"。它知道 Python 函数的形状，但不理解任务的含义。基于 50 条 canonical solution 做微调，教的不是新语法或新算法，而是**这个 benchmark 的作者到底希望规格被怎么读**。这个 shift 体现在 pass@1 从 9.65%（baseline, 11/114）→ 16.67%（M4, 19/114）。

## Q: 这个提升在统计上站得住脚吗？

**不完全站得住**，要如实说清楚：

- Delta = +7.02pp，95% CI half-width = ±8.73pp（`SE_diff = √(2.77² + 3.49²) ≈ 4.46pp`，×1.96）。观测差异**小于**置信区间半宽，所以**不能在 95% 置信度下断言这是真实提升**——不排除是 N=114 抽样噪声。
- 但定性信号是一致的：**10 题 fail→pass，只有 2 题 pass→fail**（回归），5:1 偏向有利；抽查的 flip 样例（`HumanEval/4`、`/162`、`/21`、`/122`、`/46`）里，微调后的解法系统性地更像"通用算法"而不是本文档前面描述的硬编码/语义误读模式——跟"微调应该改变什么"那一节的假说方向一致。
- **没有去做**`docs/grilling_m1_m2.md`/`grilling_m4_pre.md` 里预留的 seed 偏差诊断（检查 50/114 切分是否恰好把难题都留给了 eval）——权衡过后，决定不做额外排查，直接把这个不确定性如实写进 blog，而不是为了讲一个更干净的故事去挖掘到"显著"为止。这跟本文档下面"解读上的坑"一节的立场一致：**50 个样本、N=114 eval 的这个规模，本来就该预期置信区间盖住大部分点估计的提升**。

结论：把这次结果讲成"一个统计上不确定、但定性上一致的微调效果信号"，而不是"微调让 pass@1 提升了 X 个百分点"。

---

## Q: 有哪些解读上的坑要主动 disclose？

- **train/eval 同分布**（都在 HumanEval）：微调改进部分反映的是"分布对齐"，不完全是"能力增益"。更严谨的做法是 `MBPP-train → HumanEval-eval`——Phase 2 的扩展，不是 MVP 阶段该做的。
- **Baseline 是 4-bit 量化**，不是 fp16。社区里 fp16 的数字（Llama-2 论文报 ~12-14%）可能比我们观察到的高 1-2pp。M2 和 M4 用同样 4-bit 设置，所以**对比公平**，但**绝对 baseline 数字**跟公开 benchmark **不直接可比**。
- **N=114 eval → 95% CI ≈ ±6pp**：任何小于 6pp 的微调收益都在噪声区间里。按置信区间讲改进，不要只报点估计。
- **50 个样本对微调来说极小**：QLoRA 能在小数据收敛，但可达上限有限。别夸大——这是**微调机制的 demo**，不是 SOTA 尝试。

---

## 交叉引用
- 术语：`docs/glossary.md`
- Grilling 过程：`docs/grilling_m1_m2.md`
- 设计决策：`README.md` Decisions 1-10
- 项目状态：`CLAUDE.md`
