# Grilling M6 Pre-Development

快问快答格式记录 **M6（Inference Optimization / vLLM）开发前**的决策与理由。跟 M3/M4/M5 pre-grill 一样是**事前**——写代码前拍板。

M6 的核心洞察：M2/M4/M5 一路用的都是"裸 `model.generate()` 循环"，这是最简单但**不是生产环境会用的服务方式**。M6 要回答的问题是——**换成专门的推理服务框架（vLLM）到底值不值**，用真实数字量化这个提升，而不是空谈"vLLM 更快"。

---

## 元问题

### Q: 这份 grill 决定了什么？

| # | 主题 | 决定 |
|---|---|---|
| Q1 | 对比范围 | 只比较 serving 方式（vLLM vs 裸 HF generate()），都用 M5 的 GPTQ-4bit checkpoint，不重复测 fp16 |
| Q2 | 测速度维度 | 单请求延迟 + 114 题 batch 吞吐量都测——只测单请求看不出 vLLM continuous batching 的核心优势 |
| Q3 | 正确性校验 | vLLM 生成结果要重新跑一次 pass@1，对照 M5 的 GPTQ-4bit 数字确认换框架没有偷偷改变模型质量 |
| Q4 | 脚本结构 | 新写 `eval/generate_vllm.py`；HF 那侧（`eval/generate_quantized.py`）不改代码，用 notebook `%%time` 包一层计时 |
| Q5 | Notebook 归属 | 合并进 `notebooks/run_quantization_colab.ipynb`（M5 的 notebook），不新建——理由见下 |
| Q6 | 风险应对 | 所有已知风险 + 对应的轻量冒烟测试都写进 notebook（不只是文档里提一句），跑的人能现场看到"这步在测什么、为什么测" |

---

## Q1 · 对比范围：只测 serving 方式，模型固定为 GPTQ-4bit

### Q: 为什么不顺便也测 fp16 merged 在 vLLM 里跑多快？

fp16 vs GPTQ-4bit 的对比是 M5 已经回答过的问题（质量/体积权衡）。M6 的问题是另一个维度——**同一个模型，换一种 serving 方式，速度差多少**。两个维度混在一起测，矩阵变成 2×2，工作量翻倍，但故事线反而更难讲清楚（"是量化更快，还是 vLLM 更快，还是两个一起更快"分不清）。M6 只固定用 GPTQ-4bit（本来就是准备部署的那个模型），只变 serving 方式这一个变量。

### Q: 那"裸 HF generate()"这边是不是要重新写代码？

不用。`eval/generate_quantized.py` 已经是这条路径了，M5 就用它跑过 GPTQ-4bit 的生成（为了算 pass@1）。M6 只是**给这次调用包一层计时**，不改生成逻辑本身。

---

## Q2 · 测速度维度：单请求延迟 + batch 吞吐量都测

### Q: 为什么不能只测单请求延迟，跟 HF 那边一样"一个一个跑"？

vLLM 的核心价值就是 PagedAttention + continuous batching 带来的并发处理能力——只测单请求延迟，两边其实是在测"谁的单次 forward 更快"，测不出 vLLM 真正强在哪。114 题一次性提交给 vLLM 的 `LLM.generate()`（vLLM 内部自动做 batching），用总耗时算出的吞吐量，才是 vLLM 相对裸 generate() 循环的完整优势。

### Q: 具体测哪几个数字？

- **单请求延迟**：vLLM 对 1 个 prompt 单独计时（batch=1），可以直接跟 HF 逐题循环的"每题平均耗时"比——这个对比反映的是 vLLM 优化过的 CUDA kernel 本身带来的提升，跟 batching 无关
- **batch 吞吐量**：vLLM 一次性提交全部 114 个 prompt，总耗时 → tokens/sec，这个数字反映 continuous batching 的提升

三个数字（HF 逐题总耗时 / vLLM 单请求耗时 / vLLM batch-114 总耗时）放在一张表里，故事线清楚：换 kernel 提升多少、加上 batching 又提升多少。

---

## Q3 · 正确性校验：重新跑一次 pass@1

### Q: 为什么不直接信任模型没变，跳过这一步？

vLLM 的推理 kernel 实现（PagedAttention、算子融合等）跟 HF 原生 `generate()` 不是同一套代码路径，即使都是 greedy（`temperature=0`），浮点计算顺序不同可能导致极少数 token 的输出有细微差异。这一步成本很低（多跑一次 `eval/score.py`），但能确认"换 serving 框架没有偷偷改变模型质量"，避免 M6 的速度数字建立在一个"其实模型输出变了"的错误前提上。

---

## Q4 · 脚本结构：新写 generate_vllm.py

### Q: `eval/generate_vllm.py` 的契约是什么？

```
python eval/generate_vllm.py \
    --model-path models/gptq_4bit \
    --eval-set data/processed/eval_114.jsonl \
    --output data/processed/quant_gptq4bit_vllm_generations.jsonl
```

内部：
1. `vllm.LLM(model=args.model_path, quantization="gptq")` 加载
2. `SamplingParams(temperature=0, max_tokens=512, stop=STOP_SEQUENCES)`——`STOP_SEQUENCES`/`max_tokens` 从 `generate_completions.py` import，保证解码参数跟 HF 那边完全一致（不然速度差异可能掺杂"生成了不同长度内容"这个 confound）
3. 先对第 1 个 prompt 单独调用一次 `llm.generate([prompt], params)` 并计时 → 单请求延迟
4. 再对全部 114 个 prompt 一次性调用 `llm.generate(all_prompts, params)` 并计时 → batch 吞吐量
5. 输出 jsonl，命名跟 HF 那份区分开（`quant_gptq4bit_vllm_generations.jsonl` vs `quant_gptq4bit_generations.jsonl`），避免互相覆盖，也方便日后同时保留两份生成结果做人工比对

### Q: vLLM 的 `stop` 参数跟 HF 那边 `truncate_at_stop_sequence` 的事后截断是一回事吗？

效果等价但实现不同：vLLM 原生 `stop` 是在生成循环内部检测到就提前停止（本身也是速度优势的一部分，不用生成到 512 tokens 再截断），HF 那边是生成完整 512 tokens 后再截断字符串。只要 greedy 解码保证两边在截断点之前生成的 token 一致，语义上等价。这个差异不需要用户拍板，是个实现细节。

---

## Q5 · Notebook 归属：合并进 M5 的 notebook

### Q: 为什么不跟 M2/M3/M4 一样分开？

`grilling_m4_pre.md` Q6 当初选分开，是因为 M3 的产物（LoRA adapter）**进了 git**——M4 notebook 可以直接 `git clone` 拿到 adapter，不用重跑 M3。

M5 的产物（`models/gptq_4bit/`，GB 级）**不进 git**。如果 M6 是单独 notebook，等于全新 Colab runtime、空硬盘——M6 打开时 `models/gptq_4bit/` 根本不存在，得把 M5 的 merge+量化整套重跑一遍才能拿到 checkpoint，白白浪费一次 GPU 时间。合并成一个 notebook，量化完在同一个 session 里直接接着测 vLLM，不用重跑。

---

## Q6 · 风险应对：所有风险 + 轻量冒烟测试都写进 notebook

### Q: 为什么不只在文档里写风险，要求都体现在 notebook 里？

`docs/grilling_m5_pre.md` 之前的风险项写在文档里，但实际跑的人在 Colab 里看不到——真正跑起来才发现问题，已经花了 15-30 分钟。这次要求：**每个已知风险，在 notebook 里对应位置都有一段 markdown 说明 + 一个轻量冒烟测试 cell**，用几十 MB 的小模型/小样本跑一遍同样的 API 调用路径，几秒钟内验证库版本、显存、格式兼容性，而不是在 7B 模型上跑到一半才炸。

### Q: 风险 ↔ 冒烟测试对照表？

| 风险 | 什么时候可能炸 | 轻量测试 | 位置 |
|---|---|---|---|
| M5: merge 时显存不够（fp16 7B ≈14GB，T4 只有 15GB） | merge 阶段 OOM，白等 15+ 分钟 | `torch.cuda.mem_get_info()` 查看可用显存，<13GB 提前警告 | M5 Step 4 之前 |
| M5: `gptqmodel`/`optimum` 库版本或 API 不兼容 | 量化跑到一半报错，白等 20+ 分钟 | 用 `hf-internal-testing/tiny-random-LlamaForCausalLM` 跑一遍完整 `GPTQConfig → from_pretrained → save_pretrained` 流程 | M5 Step 4 之前 |
| M6: vLLM 在这个 Colab runtime 装不上/跑不动（CUDA/驱动版本挑剔） | 加载 7B checkpoint 时才发现 vLLM 引擎本身有问题 | 用 `facebook/opt-125m` 跑 `LLM(...).generate()` 冒烟测试 | M6 vLLM 步骤之前 |
| M6: vLLM 读不懂我们存的 GPTQ checkpoint 格式 | 只能用真实 checkpoint 测，没法用小模型模拟 | 靠顺序天然防住：单请求测试本身就是小规模真实测试，notebook 里显式 `assert`／失败即停，不再往下跑 batch-114 | M6 单请求测试之后 |
| M5: 50 条校准数据可能不够 | 量化能跑通，但 pass@1 掉得多 | 测不了"会不会失败"，只能事后从分数判断，保留成已知取舍 | 不适用（记录在 grilling_m5_pre.md） |

---

## M6 完成后的下一步

- **正常路径**：vLLM 单请求 + batch 吞吐量都明显优于裸 HF generate()，pass@1 对照一致 → 写进 blog Part 3（量化+推理加速的组合故事），进 M7（成本分析，用这里的吞吐量数字算单请求成本）
- **vLLM 没有明显更快**：先怀疑 `gpu_memory_utilization` 或 batch 没有真正一次性提交（检查是不是不小心退化成逐题调用）
- **vLLM 装不上/跑不动**：见风险表，Colab 换更高档 GPU（如果有）或换驱动兼容的 vLLM 版本

---

## 交叉引用
- 术语：`docs/glossary.md`
- M4 pre-grill（notebook 分开 vs 合并的原始决策）：`docs/grilling_m4_pre.md`
- M5 pre-grill（量化对象/方法/校准数据 + 风险表雏形）：`docs/grilling_m5_pre.md`
- 设计决策：`README.md` Decision 8
- 项目状态：`CLAUDE.md`
