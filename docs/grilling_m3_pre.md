# Grilling M3 Pre-Development

快问快答格式记录 **M3 QLoRA 微调开发前**的 8 个设计决策与理由。写代码之前先拍板——训练一次 20-30 min GPU，改成本高，选错代价大。

M1/M2 那份 grill 是**事后回看**改现有代码；这份是**事前预 grill** 定超参空间。两者互补。

---

## 元问题

### Q: 为什么先 pre-grill 再写代码？

M3 涉及 **7 类超参**（LoRA config × 3 + 训练 hyperparams × 6 + 数据预处理 × 2）。写完再改一次训练不成本很高（20-30 min T4 时间）。**pre-grill 是廉价的决策**，跑训练是昂贵的验证。

### Q: 这份 grill 的 8 支各自决定了什么？

| # | 主题 | 决定 |
|---|---|---|
| Q1 | 库选择 | **PEFT + `transformers.Trainer`** |
| Q2 | LoRA config | **r=16, α=32, dropout=0.05, target=所有 linear 层** |
| Q3 | 训练超参 | **5 epochs / batch 1 × grad_accum 4 / LR 2e-4 / cosine / paged_adamw_8bit** |
| Q4 | Loss masking | **completion-only**（prompt 部分 labels 置 -100） |
| Q5 | Prompt format | **`prompt + completion + EOS`，无分隔符** |
| Q6 | 训练时评估 | **只记 train loss**，不做中间 pass@1 |
| Q7 | 保存格式 | **只存 LoRA adapter**（~40-80MB），不 merge |
| Q8 | 环境 | **Colab notebook 启动器 + `train/train_lora.py` 主脚本** |

---

## Q1 · 用什么库？

### Q: 三个选项各是什么？

- **A. PEFT + `transformers.Trainer`**：官方最标准组合，PEFT 负责 LoRA 逻辑，Trainer 负责训练循环
- **B. TRL `SFTTrainer`**：更高层封装，一站式处理 tokenize + completion-only loss + Trainer
- **C. 手写 loop**：完全自定义，最大控制

### Q: 为什么选 A 而不是 B？

- 教学项目要**透明**——每步做什么都能追进去
- SFTTrainer 的 magic（自动 collator、自动 loss mask）**出问题难查**
- PEFT + Trainer 的**每个环节可控**：数据预处理、collator、loss、优化器都是显式写的

### Q: 为什么不 C？

重复造轮子，多写 200 行代码，没有 pedagogical 收益。Trainer 已经处理了 gradient accumulation、mixed precision、logging、checkpoint 保存——这些不是我们要展示的东西。

---

## Q2 · LoRA config

### Q: rank (r) 选多少？

**16**。QLoRA 论文给的默认 r=64 是为大规模指令数据集（Alpaca 52K）设的。我们只有 **50 样本**——r=64 会给出 ~160M 可训练参数，过拟合风险大。r=16 → ~40M 可训练参数（约总参 0.6%），够容量、不过量。

### Q: alpha (α) 选多少？

**32**（= 2r）。LoRA 论文的经验法则是 `α = 2r`。effective learning rate = base_lr × (α/r) = base_lr × 2——相当于给 LoRA 部分一个 1x-2x 的隐性 LR 放大。

### Q: dropout 选多少？

**0.05**。QLoRA 论文默认。小数据仍需正则化防过拟合，但太大（>0.1）在 50 样本上会稀释信号。

### Q: target_modules 选哪些？

**所有 linear 层**：`q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`（7 个）。

**为什么不只调 q_proj + v_proj（LoRA 原论文的最小设置）？** 那是为大数据集节省参数的策略。我们参数预算没压力，反而怕**欠拟合**——50 样本要让模型学到"通用算法而非硬编码"的 shift，需要多个层级都能调整。

### Q: 训练 bias 吗？

**不训**（`bias="none"`）。标准 LoRA 做法。

---

## Q3 · 训练超参

### Q: 多少 epochs？

**5**。50 样本 × 5 epochs / (batch 1 × grad_accum 4) = **约 62 步**。
- 3 epochs（~37 步）可能欠拟合
- 10 epochs（~125 步）小数据容易背题
- 5 是常见小数据甜蜜点

### Q: batch size 怎么选？

**per-device batch = 1，grad_accum = 4，有效 batch = 4**。T4 只有 15GB VRAM，7B 模型 4-bit + LoRA 需要激活 + 梯度，per-device > 1 会 OOM。grad_accum 补偿：在多个 forward 之间累积梯度，等效大 batch 但内存不变。

### Q: LR 选多少？

**2e-4**。QLoRA 论文默认。这个数**比全模型微调（通常 5e-5 到 1e-4）高 2-4x** 是因为：
- LoRA 只训练小部分参数，需要更大 LR 才能有效学习
- 4-bit 量化本身对梯度有正则化效应，容许更激进的 LR

### Q: LR scheduler？

**cosine**。经典 warmup + cosine decay 曲线。末期 LR 会平滑降到 0，防止训练末期梯度大幅摆动。

### Q: warmup 怎么设？

**5 步固定 warmup**。数据太少无法用 `warmup_ratio`（3% × 62 步 = 1.86 步太短）。5 步是"至少一个有效 epoch 的 warmup"的粗粒度做法。

### Q: 用什么 optimizer？

**`paged_adamw_8bit`**（bitsandbytes 提供）。标准 AdamW 需要保留 fp32 的 momentum + variance，占用 8 字节/参数。8-bit 版本用 2 字节/参数——**微调 7B 模型能省几个 GB**，是 QLoRA 能在 T4 上跑起来的关键之一。"Paged" 是 NVIDIA CUDA 的分页机制，防 optimizer state 突发 OOM。

### Q: weight decay?

**0**。LoRA 惯例。LoRA 参数本身规模小，weight decay 的正则化收益微弱，反而可能阻碍学习。

### Q: max_grad_norm?

**0.3**。QLoRA 论文推荐值。防止训练早期梯度爆炸。

---

## Q4 · Loss masking：completion-only vs 全序列

### Q: 为什么选 completion-only？

- **Prompt 是输入**，模型不该被训练去"复述输入"——那既浪费容量又可能让模型学会 prompt 的 specific 表面模式
- 50 样本很小，**全序列 loss 会让 prompt 部分过拟合**（模型开始记住 docstring 的确切措辞）
- HumanEval 标准做法就是 completion-only
- 从 M4 视角，我们要评估的是"给一个新 prompt 能否写对"——训练也应该只在"生成"部分算 loss

### Q: 实现细节？

在 data collator 里手动构造 `labels`：
- Prompt 部分的 token → labels = **-100**（PyTorch cross-entropy 忽略这个值）
- Completion + EOS 部分 → labels = 原 token id

---

## Q5 · Prompt Format：怎么拼？

### Q: 数据源结构？

`train_50.jsonl` 每行：
```json
{"task_id": "HumanEval/xx", "prompt": "def foo():\n    \"\"\"...\"\"\"\n", "completion": "    return ..."}
```

### Q: 拼接策略？

```
{prompt}{completion}{EOS}
```
- **不加分隔符**：prompt 结尾已经有 `\n`，completion 已有正确缩进
- **加 EOS**：教模型"函数写完了到这里停"——**这是让模型学会自然停止、减少 STOP_SEQUENCES 依赖的关键**
- **BOS 由 tokenizer 自动加**

### Q: Tokenize 策略上有什么坑？

Llama tokenizer 是 SentencePiece，跨字符串边界的 BPE 可能合并。为了 loss mask 边界精确：
- 分开 tokenize prompt 和 completion（各自 `add_special_tokens=False`）
- 手动拼：`[BOS] + prompt_ids + completion_ids + [EOS]`
- Labels：`[-100]*(1 + len(prompt_ids)) + completion_ids + [EOS]`

避免"整段一起 tokenize 后找 prompt 长度"这种脆弱做法。

---

## Q6 · 训练时评估策略

### Q: 为什么不做中间 pass@1 评估？

**训练总步数只有 62 步；单次 pass@1 评估需要 20 分钟生成**。做 1 次中间 eval 就相当于把训练时间翻倍，得不偿失。

### Q: 那怎么监控？

**只记 train loss**。每步都记（`logging_steps=1`）。62 步很短，肉眼看曲线足够。

### Q: 怎么判断过拟合？

Train loss 的经验值：
- 初始：~2-3
- 健康收敛：稳定降到 0.5-1.0
- **警戒**：<0.5（可能开始背题）
- **确认过拟合**：<0.1（模型基本记住了训练答案）

### Q: 那 M4 结果反常怎么办？

有两种反常情况：
- **M4 pass@1 反常低**：可能是过拟合（模型只在训练 prompt 上好，泛化失败）→ 减 epochs 重跑
- **M4 pass@1 反常高**：可能是训练/评估集难度不对齐（见 M1/M2 grill Q8 债务）

---

## Q7 · 保存格式

### Q: 三个选项？

| 选项 | 大小 | 用途 |
|---|---|---|
| **A. LoRA adapter only** | ~40-80 MB | M4 需要 base + adapter |
| B. Merged fp16 model | ~14 GB | 独立运行、部署简单 |
| C. Merged + 4-bit 量化 | ~4 GB | M5 才需要 |

### Q: 为什么选 A？

- **便携**：adapter 本身就是小文件，可以进 git（<100MB GitHub 限制）
- **组合灵活**：base 模型可以不变，多个 adapter 可以切换/对比
- **M4 workflow 简单**：加载 base + `PeftModel.from_pretrained(adapter_path)`
- **M5 才做 merge**：MVP 阶段没必要提前 merge，会浪费磁盘和 Colab 时间

### Q: 具体保存到哪？

- 训练中间产物：`train/output/`（gitignore，Trainer 的 output_dir）
- 最终 adapter：`models/lora_adapter/`（提交进 git，供 M4 加载）

---

## Q8 · 环境

### Q: 为什么还是 Colab？

跟 M2 一样：base 模型 4-bit 量化需要 CUDA（bitsandbytes on Mac/MPS 不 work）。T4 15GB 够 7B + LoRA。

### Q: 代码组织？

- **主脚本**：`train/train_lora.py`（跟 `eval/generate_completions.py` 平级）
- **Colab 启动器**：`notebooks/run_train_lora_colab.ipynb`（跟 M2 的 notebook 平级）
- **训练输出**：`train/output/`（gitignore）
- **最终 adapter**：`models/lora_adapter/`（commit）

### Q: `requirements-colab.txt` 要加什么？

只加 `peft`。其他（transformers、accelerate、bitsandbytes）都已经在了。

---

## 训练完之后的下一步（M4 视角）

- 加载 base 模型（同 M2 的 4-bit 量化设置）
- `PeftModel.from_pretrained(base_model, "models/lora_adapter/")`
- 复用 `eval/generate_completions.py` 的生成 + `eval/scoring.py` 的判分
- **对比数字**：`M4_pass@1 - M2_pass@1` 是核心成果

---

## 待观察的债务项

1. **Train loss 曲线的形状**：如果 62 步内 loss 没降到 0.5 以下，说明训练不够——考虑加 epochs
2. **过拟合信号**：如果 loss 迅速降到 <0.1，考虑减 epochs 或加 dropout
3. **M4 pass@1 vs M2 pass@1 的差距**：见 M1/M2 grill Q1 结论——需要 >9pp 才算 95% 显著

---

## 交叉引用
- 术语：`docs/glossary.md`
- M1/M2 grill：`docs/grilling_m1_m2.md`
- Baseline 行为：`docs/baseline_behavior.md`
- 设计决策：`README.md`
- 项目状态：`CLAUDE.md`
