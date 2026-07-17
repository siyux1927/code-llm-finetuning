# 术语速查表 (Glossary)

快问快答格式的术语参考。项目里出现的概念、生成/量化/微调相关词汇，回头查这里，不必翻 blog 或 README 长文。

---

## 模型 & 数据

**Q: `Llama-2-7b-hf` 是什么？**
Meta 2023 年发布的 Llama 2 系列、7B 参数、**base 版本**——只做过预训练，没做过指令微调、没做过 RLHF。`-hf` = HuggingFace 转换后的 checkpoint 格式。

**Q: Base / Chat / CodeLlama 三种变体的区别？**

| 变体 | 训练做了什么 | 面向场景 |
|---|---|---|
| `Llama-2-7b-hf`（**我们用的**） | 只预训练 | 通用文本续写 |
| `Llama-2-7b-chat-hf` | 预训练 + SFT + RLHF | 对话，能听懂"请帮我..." |
| `CodeLlama-7b` | Llama 2 上继续用 5000 亿 tokens 代码训练 | 代码生成，已经很强 |

**Q: Llama 2 训练用了什么数据？**
约 2 万亿 tokens 公开在线数据：Common Crawl 网页 + 书籍 + arXiv 论文 + GitHub 代码 + Wikipedia 等。Meta 没完全披露细节。**知识截止 ~2022 年 9 月**。

**Q: Llama 现在最新是什么？**
截至 2026 年——**Llama 4**（2025.04 发布，MoE 架构，Scout / Maverick / Behemoth 三档）。中间还有 Llama 3、3.1、3.2、3.3。

**Q: 项目为什么不用最新的 Llama 4？**
见 `README.md` Decision 2——**故事讲不清**。Llama 2 base 弱、微调空间大、社区 benchmark 多；用最新模型会让"微调效果"被"底模已经很强"稀释。

---

## 量化 (Quantization)

**Q: 什么是量化？**
把模型权重从高精度 → 低精度存储，**省显存**，代价是计算精度小幅损失。

**Q: fp32 / fp16 / 4-bit 存 7B 模型分别多大？**

| 精度 | 每参数占用 | 7B 模型总大小 | Colab T4 (15GB) |
|---|---|---|---|
| fp32 | 4 字节 | 28 GB | ❌ |
| fp16 | 2 字节 | 14 GB | 勉强，很危险 |
| **4-bit** | 0.5 字节 | **~4 GB** | ✅ 宽裕 |

**Q: 什么是 NF4？**
**Normal Float 4-bit**——一种为"神经网络权重近似服从正态分布"这个特性专门设计的 4-bit 编码。比朴素均匀 4-bit 精度损失小得多。QLoRA 论文提出的。

**Q: "fp16 compute" 是什么意思？**
权重存 4-bit，但每次矩阵乘法时**临时解压到 fp16 再算**。存储省、计算准，代价是运算时多一步解压。类比：图片存 JPEG，看的时候解成全彩渲染。

**Q: bitsandbytes 是什么？**
实现 4-bit / 8-bit 量化的开源库，两三行代码接入 HuggingFace transformers。**CUDA-only**（Mac/MPS 不支持——这就是为什么 M2/M3 只能跑 Colab）。

---

## 解码 (Decoding)

**Q: 什么是解码？**
模型每一步输出"下一个 token 的概率分布"（词表几万个 token 每个都有概率）。**解码策略决定怎么从这个分布里挑一个 token**。

**Q: Greedy 是什么？**
永远选**概率最高**的 token。**确定性、无随机**——同一个 prompt 永远得同一个输出。

**Q: 其他解码策略？**
- **Sampling (temperature)**：按概率随机抽，多样但不确定
- **Beam search**：同时保留 top-k 候选路径，接近全局最优但慢
- **Top-p / top-k**：只在最高概率的一部分 token 里抽

**Q: `do_sample=False` 什么意思？**
HuggingFace transformers 里 **greedy 模式**的开关。设为 `False` = greedy。

**Q: 为什么 Greedy 适合 pass@1？**
pass@1 要求"每题一次尝试"。Greedy 是确定性的——结果**可复现**、**公平对比**（M2 vs M4 的差异纯粹来自模型本身，不是采样噪声）。

---

## 生成参数

**Q: `max_new_tokens` 是什么？**
模型在 prompt 之后**最多能生成多少个 token**（不含 prompt）。到上限强制停止。项目里首次跑 256，后来发现太紧改成 512。

**Q: 什么是 STOP_SEQUENCES / stop 规则？**
**生成完之后的后处理裁剪**。找到第一个匹配的 stop 序列，把它及后面的内容全部丢掉。用途：base 模型不知道"函数写完了该停"，会继续写下一个函数、注释、demo 代码，需要人工兜底截断。

**Q: 我们用的 STOP_SEQUENCES 有哪些？**

| Stop 序列 | 意图 |
|---|---|
| `\ndef ` | 模型开始写下一个函数 |
| `\nclass ` | 开始写下一个类 |
| `\nif __name__` | 主入口块 |
| `\nprint(` | 演示代码 |
| `\n#` | 开始写注释 |
| `\n@` | 开始写装饰器 |

---

## Python & HumanEval 术语

**Q: docstring 是什么？**
Python 函数/类/模块的**第一条字符串字面量**（通常用三引号 `"""..."""`）。Python 特殊对待：可用 `func.__doc__` 访问、`help(func)` 会显示、IDE 用它做 hover 提示。

**在 HumanEval 里，docstring 就是任务说明**——模型读它推断要实现什么。

```python
def has_close_elements(numbers, threshold):
    """Check if any two numbers are closer than threshold."""   # ← docstring
    ...
```

**Q: canonical solution 中文叫什么？**
**标准答案** / **官方参考解** / **规范解答**。"canonical" 意思是"官方权威规范"。

**Q: canonical solution 在项目里做什么用？**
1. **训练数据**（M3）：`train_50.jsonl` 的 `completion` 字段就是 canonical solution
2. **判分器 sanity check**（`eval/tests/test_scoring.py`）：验证 114 个标准答案都能过自己的测试——保护判分代码本身正确

**Q: pass@1 是什么？**
每题让模型生成 **1 个**解答，跑测试判 pass/fail。分数 = 通过题数 / 总题数。是 HumanEval 的**官方指标**。

**Q: pass@k（k > 1）呢？**
每题生成 k 个候选，只要有一个通过就算通过。数字更好看，但不代表单次调用的能力。我们选 pass@1（见 README Decision 6）。

---

## 微调 & LoRA

**Q: 什么是 fine-tuning（微调）？**
在已经预训练好的 base 模型上，用特定任务的数据**继续训练**一小段，让模型偏向这个任务。

**Q: LoRA 是什么？**
**Low-Rank Adaptation**——微调时**不改原模型权重**，而是**加两个小矩阵**（rank-r 分解）叠在上面。只训练那两个小矩阵，可训练参数量 <1%。省显存、可插拔、可以随时切换任务。

**Q: QLoRA 是什么？**
**Q**uantized **LoRA**——base 模型先量化到 4-bit（省显存），然后在上面做 LoRA。显存需求进一步降低，Colab Free 的 T4 就能微调 7B。见 README Decision 5。

**Q: Completion tuning vs Instruction tuning？**
- **Completion**（我们用的）：数据格式 `prompt → 直接续写`。贴近 base 模型的自然行为、实现最简单。Decision 4。
- **Instruction**：数据格式 `"请做 XXX" → 答案`。需要 **loss masking**（只在答案部分算 loss，不在指令部分算）。更贴近对话使用场景，但预处理更复杂。

---

## 统计

**Q: pp 是什么？**
**Percentage points（百分点）**——**绝对差**。10% → 20% 是 +10pp（也是相对 +100%）。用 pp 避免"10%到底是绝对还是相对"的歧义。

**Q: SE 是什么？**
**Standard Error（标准误）**——观测值抖动的幅度。对 pass@1：`SE = √(p(1-p)/N)`。N 越大 SE 越小（1/√N 定律）。

**Q: CI 是什么？**
**Confidence Interval（置信区间）**——以观测点为中心、宽度 **±2·SE** 的区间，真值有约 95% 概率落这里面。

**Q: 为什么我们的 pass@1 CI 是 ±6pp？**
N=114、假设 p=0.13 时：
`SE = √(0.13 × 0.87 / 114) ≈ 2.9pp`
`95% CI ≈ ±2 × 2.9 = ±5.8pp`

**Q: 为什么从 N=50 换到 N=114？**
把 SE 从 4.8pp 降到 2.9pp，让 M4 vs M2 的差异更容易脱离噪声区间。见 `docs/grilling_m1_m2.md`（待写）里 Q1 的完整推导。

---

## 交叉引用
- 项目状态：`CLAUDE.md`
- 设计决策：`README.md` Decisions 1-10
- Baseline 行为观察：`docs/baseline_behavior.md`
- Grilling 过程与结论：`docs/grilling_m1_m2.md`（待写）
