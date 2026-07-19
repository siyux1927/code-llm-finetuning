# Grilling M5 Pre-Development

快问快答格式记录 **M5（Model Quantization）开发前**的决策与理由。跟 M3/M4 pre-grill 一样是**事前**——写代码前拍板。

M5 的核心洞察：M3/M4 用 bitsandbytes 4-bit 是**训练时的技巧**（QLoRA 为了在 T4 上塞得下 7B 才量化），跟"量化一次、产出可部署 checkpoint"是两件事。M5 要做的是**训练后量化（PTQ）**——把 M3 微调好的模型压缩成一个独立、可反复部署的小 checkpoint，为 M6（vLLM 部署）做准备。

---

## 元问题

### Q: 这份 grill 决定了什么？

| # | 主题 | 决定 |
|---|---|---|
| Q1 | 量化对象 | 只量化 M3 微调后的模型（base + adapter merge），不重复量化 base model |
| Q2 | 量化方法 | GPTQ（transformers 原生 `GPTQConfig` + `optimum`/`gptqmodel` 后端），不是 bitsandbytes |
| Q3 | 量化档位 | 只做 4-bit，不做 8-bit（原 README Decision 7 的 8-bit 档已去掉，见该处更新） |
| Q4 | 校准数据 | 复用 `data/processed/train_50.jsonl`，不引入外部语料 |
| Q5 | 成功标准 | pass@1 + 模型体积两项；不测推理延迟/显存峰值（那是 M6 的事） |
| Q6 | 脚本结构 | 新写 `train/quantize_gptq.py` + `eval/generate_quantized.py`，不改 `generate_completions.py` |
| Q7 | 产物去留 | 量化 checkpoint 只留 Colab 本地，不进 git；生成结果 jsonl 进 git |

---

## Q1 · 量化对象

### Q: 为什么只量化微调后的模型，不重复量化 base？

M5 的定位是"把 M4 的成果部署化"，不是重新验证 M2 的 baseline 故事——base model 量化前后的行为已经在 M2（bnb 4-bit）里看过一次了。两组都做会让工作量翻倍，且不会带来新的结论。

### Q: 具体怎么拿到"微调后的模型"？

`PeftModel.from_pretrained(base, adapter_path)` 之后调用 `merge_and_unload()`，得到一个不依赖 PEFT 的、独立的 fp16 模型（内存中，不落盘）。这是 `docs/grilling_m4_pre.md` Q2 当初"M4 不 merge"的原因——merge 推迟到部署阶段，现在就是那个阶段。

---

## Q2 · 量化方法：GPTQ，不是 bitsandbytes

### Q: bitsandbytes 不是已经很好用了吗，为什么还要换？

bnb 4-bit（NF4）是**训练时**的技巧：不需要校准数据，按权重统计分布现场量化，目的是省显存让训练跑得起来，没有配套的高性能推理 kernel，也不是"量化一次存下来复用"的工作流。

GPTQ 是**训练后量化**：用校准数据做逐层 Hessian-based 权重取整优化，量化完存一个独立、更小的 checkpoint，配套 ExLlama/Marlin 之类的推理 kernel。**vLLM（M6）对 GPTQ 的原生支持远好于 bnb**——这是 M5 选 GPTQ 而不是继续用 bnb 的核心原因：故事线要接得上 M6。

### Q: 具体用什么库/API？

`auto-gptq` 这几年维护变弱，现在 transformers 生态走的是原生集成：

```python
from transformers import GPTQConfig
gptq_config = GPTQConfig(bits=4, group_size=128, dataset=calibration_texts, tokenizer=tokenizer)
model = AutoModelForCausalLM.from_pretrained(merged_model_path_or_obj, quantization_config=gptq_config, device_map="auto")
model.save_pretrained(output_dir)  # 存下来是完整量化 checkpoint，以后 from_pretrained(output_dir) 直接加载，不用再传 config
```

后端是 `optimum` + `gptqmodel`（替代 `auto-gptq` 的新维护分支），需要加进 `requirements-colab.txt`。

---

## Q3 · 量化档位：只做 4-bit

### Q: 原计划是 8-bit + 4-bit 两档，为什么砍掉 8-bit？

- 8-bit 那次要重新跑一遍量化 + 生成 114 题 + 打分，时间和显存压力翻倍。
- 风险敞口也翻倍：`gptqmodel` 库版本兼容性、Colab 显存这些坑，多跑一次就多一次踩坑机会。
- 8-bit 主要贡献的是"质量/体积曲线更细的中间点"，不是不同的结论——MVP 阶段一个数据点（fp16 merged vs GPTQ-4bit）已经能讲清"压缩了多少体积、掉了多少精度"这个故事。
- 如果以后觉得这条曲线值得细化，8-bit 可以作为 Phase 2 扩展项补上。

同步更新了 `README.md` Decision 7。

---

## Q4 · 校准数据：复用 train_50.jsonl

### Q: GPTQ 校准数据一般用什么？我们为什么不跟着用？

社区标准做法常用 wikitext2/c4 这类通用语料，校准量级也常见 128 条起。我们只用 50 条自己的 `train_50.jsonl`：

- 不引入新的数据依赖，跟项目"自包自维"的风格一致
- 校准分布跟微调目标分布一致（都是 HumanEval canonical solutions），量化误差的方向跟任务本身对齐
- 50 条比常见的 128 条少，量化质量可能不是最优——**这是一个已知取舍，不是没注意到**。如果 pass@1 掉得离谱，这是第一个要怀疑的点。

---

## Q5 · 成功标准：pass@1 + 体积，不测延迟

### Q: 为什么不顺便把推理延迟/显存峰值也测了？

那是 M6（vLLM 部署）明确要回答的问题——现在测了也只是走个形式，容易注意力分散、拖长 M5 范围。M5 只回答"压缩了多少、精度掉了多少"这一个问题：

```
变体            pass@1      体积
fp16 merged     ??%         ~14GB
GPTQ-4bit       ??%         ~4GB
```

### Q: fp16 merged 这一档的 pass@1，会跟 M4 报的 16.67% 一样吗？

**不会，大概率不一样，这个差异要显式写进结论里**：M4 跑的是"4-bit 量化 base + LoRA adapter 叠加"（`bnb_4bit_compute_dtype=torch.float16`），M5 这里是"merge 之后的纯 fp16 模型"——精度环境更高，数字大概率比 16.67% 高一点。两个数字不矛盾，是两个不同的推理配置。

---

## Q6 · 脚本结构：新写，不改 generate_completions.py

### Q: 为什么不在 generate_completions.py 上加 --quant 参数？

`generate_completions.py` 的契约已经被 M2/M4 固定死（bnb 4-bit 加载，`--adapter` 可选）。M5 需要 fp16 / GPTQ 两种完全不同的加载路径，硬塞进去会让这个脚本变得又大又杂，偏离"这是 M2/M4 的最小改动扩展"这个概念模型。

### Q: 新脚本怎么保证解码参数跟 M2/M4 一致？

`eval/generate_quantized.py` 直接 `from eval.generate_completions import STOP_SEQUENCES` （以及 `max_new_tokens=512`、greedy），不重新定义一遍——跟 `grilling_m4_pre.md` Q3 强调的"生成参数不能变、否则引入 confound"是同一个原则。

`train/quantize_gptq.py` 负责 merge + 4-bit GPTQ 量化，产出 checkpoint 目录，命名对应 `train_lora.py` 的风格。

---

## Q7 · 产物去留：checkpoint 不进 git

### Q: 量化后的模型要不要提交到仓库？

**不要**。GPTQ-4bit checkpoint ≈ 4GB，merged fp16 model ≈14GB，两个都远超仓库能装的范围——这跟 base model 本身从来没进过 git 是一个道理。真正提交的是：

- `docs/grilling_m5_pre.md`（本文档）
- `train/quantize_gptq.py`、`eval/generate_quantized.py`、`notebooks/run_quantization_colab.ipynb`
- `data/processed/quant_fp16_generations.jsonl`、`quant_gptq4bit_generations.jsonl`（生成结果，几十 KB，脚本可从 checkpoint 重新生成，checkpoint 本身可从 adapter 重新量化——**都是可复现的中间产物，不是不可再生的数据**）

---

## 风险/待观察项

- **量化时显存**：merge 后 fp16 7B ≈14GB，T4 只有 15GB，比较紧。`GPTQConfig` 走 transformers 原生路径是逐层量化 + CPU offload，标准场景下够用，但真跑的时候要盯 `nvidia-smi`；OOM 就考虑加显式 CPU offload 参数。
- **`gptqmodel` 库兼容性**：相对新的后端，Colab 环境第一次装可能有版本兼容坑——脚本里加个清楚的报错提示（参照 M4 Q8 adapter 缺失检查的思路）。
- **50 条校准数据偏少**：见 Q4，已知取舍，pass@1 反常时第一个排查这里。

---

## M5 完成后的下一步

- **正常路径**：GPTQ-4bit pass@1 相比 fp16 merged 掉得在可接受范围内（比如 <5pp），体积压缩到 ~1/3.5 → 写进 blog Part 3 的量化素材，进 M6（vLLM 部署）
- **精度掉得离谱**：先怀疑校准数据量（Q4）——50 条可能不够，考虑扩到 128+（可以混 eval_114 的一部分，但要小心别把 eval 集"泄漏"进校准，只能用 train 侧数据或独立生成的样本）
- **量化本身跑不动（OOM/库报错）**：见风险项，先查显存和库版本

---

## 交叉引用
- 术语：`docs/glossary.md`
- M3 pre-grill（LoRA 训练 + "不 merge"决策）：`docs/grilling_m3_pre.md`
- M4 pre-grill（生成脚本契约 + merge 权衡）：`docs/grilling_m4_pre.md`
- 设计决策：`README.md` Decision 7
- 项目状态：`CLAUDE.md`
