# Code Generation Fine-tuning on LLMs

A comprehensive study on fine-tuning Llama 2-7B for code generation using LoRA and QLoRA techniques, with production optimization strategies including quantization, inference acceleration, and cost analysis.

## Project Overview

This project explores the full pipeline from model fine-tuning to production deployment, targeting the **LLMOps** engineering role at large companies. The goal is to understand:
1. How to fine-tune models for specific tasks (code generation)
2. How to optimize for production constraints (VRAM, latency, cost)
3. How to measure and communicate improvements

### Target Model
- **Base Model**: Llama 2-7B
- **Task**: Code generation (HumanEval)
- **Fine-tuning Method**: QLoRA (4-bit quantized LoRA)

---

## Project Phases

### Phase 1: MVP (Quick Validation)
- **M1**: Data Preparation
- **M2**: Baseline Testing
- **M3**: LoRA Fine-tuning
- **M4**: Fine-tuned Evaluation

### Phase 2: Iteration (Production Optimization)
- **M5**: Model Quantization
- **M6**: Inference Optimization (vLLM)
- **M7**: Cost Analysis

---

## Design Decisions

### **Decision 1: Why HumanEval + Code Generation?**

**Problem**: Need a task that is:
- Practically valuable (not toy problems)
- Objectively measurable (not fuzzy)
- Suitable for LLMOps interviews

**Options**:
- A. Text classification (too simple)
- B. Instruction-following (hard to quantify results)
- C. RAG system with retrieval-augmented generation (complex, production-relevant)

**Choice**: C (RAG system) → then scoped down to **Code Generation** for faster MVP

**Reasoning**:
- HumanEval is the **standard benchmark** in industry (all interviewers recognize it)
- Pass@1 metric is objective (code either passes tests or doesn't)
- Easy to show "improvement": "Model went from X% to Y% pass rate"
- Directly relevant to LLMOps: most companies optimize code-related models

---

### **Decision 2: Llama 2-7B vs CodeLlama vs Others?**

**Problem**: 7B model size offers good balance, but which variant?

**Options**:
- A. CodeLlama-7B (already heavily optimized for code)
- B. Llama 2-7B (general-purpose, weaker at code, more room for improvement)
- C. Mistral-7B (efficient, but less community resources)
- D. Qwen-7B (strong for Chinese, international CV less recognizable)

**Choice**: B (Llama 2-7B)

**Reasoning**:
- **Clearest story**: "I improved a general-purpose model's code capability from ~25% to 50%+"
- **More impactful**: Shows what fine-tuning can do, not just polish on already-strong model
- **Best baseline**: Most community benchmarks and comparisons available
- **Industry standard**: Widely used in production, LLMOps teams familiar with it

---

### **Decision 3: Data Volume for Fine-tuning?**

**Problem**: HumanEval has 164 problems. How many to use?

**Options**:
- A. Full 164 problems (most complete, but VRAM/time intensive)
- B. Sample 50 problems (balanced, manageable, credible)
- C. Sample 20 problems (quick, but feels insufficient)

**Choice**: B (50 problems)

**Reasoning**:
- Enough to demonstrate "meaningful improvement" from fine-tuning
- Fits in Colab Free Tier VRAM budget
- 50 samples is credible for research (not cherry-picked)
- Reduces iteration time, more room for experimentation in Phase 2

---

### **Decision 4: Completion vs Instruction-tuning?**

**Problem**: How should training data be structured?

**Options**:
- A. **Completion**: `def func(args): ...` → complete the function body directly
  - Simplest implementation, most direct task alignment
  - Closest to actual HumanEval evaluation
  
- B. **Instruction-tuning**: `"Write a function that..." + signature` → function body
  - More realistic LLMOps scenario (user gives instruction → model executes)
  - Requires loss masking (only compute loss on output, not input)
  - More complex preprocessing

**Choice**: A (Completion)

**Reasoning**:
- MVP phase prioritizes speed + simplicity
- Direct alignment with HumanEval's evaluation protocol
- Easier to debug and verify correctness
- Can revisit instruction-tuning in Phase 2 if needed

---

### **Decision 5: LoRA vs QLoRA?**

**Problem**: Fine-tuning method affects VRAM constraints and optimization opportunities

**Options**:
- A. **LoRA**: Standard approach, keeps base model in full precision (float16)
  - More VRAM (potentially OOM on Colab Free)
  - Simpler, more familiar
  
- B. **QLoRA**: Quantizes base model to 4-bit, then LoRA on top
  - Significantly lower VRAM usage
  - One-line library integration (`bitsandbytes`)
  - Same quality, smaller footprint
  - Already a production optimization technique

**Choice**: B (QLoRA)

**Reasoning**:
- **Practical**: Ensures stable training on Colab without VRAM surprises
- **Low overhead**: `bitsandbytes` makes it a few lines of code
- **Educational value**: Introduces quantization early, which connects to Phase 2 (M5)
- **LLMOps alignment**: Resource-constrained optimization is core to the role

---

### **Decision 6: Pass@1 vs Pass@k Evaluation?**

**Problem**: How to measure improvement? Single attempt or multiple?

**Options**:
- A. **Pass@1**: One generation attempt per problem
  - Simpler, more conservative
  - Official HumanEval metric
  - Numbers look lower (e.g., 30% → 45%)
  
- B. **Pass@k**: Generate k solutions, take best
  - More generous (looks better)
  - Requires k× computation
  - Better represents "true capability"

**Choice**: A (Pass@1)

**Reasoning**:
- **Fair comparison**: Matches official HumanEval protocol
- **Clearest story**: Direct apples-to-apples improvement
- **Simpler engineering**: No extra sampling logic
- **Credibility**: No need to explain "why k=10 instead of k=5"

---

### **Decision 7: Quantization Method in M5?**

**Problem**: How to bridge from fine-tuning to production optimization?

**Options**:
- A. Direct 4-bit quantization only (single compression point)
- B. Progressive quantization (8-bit first, then 4-bit) — shows a quality/size curve
- C. Compare multiple methods (GPTQ vs AWQ vs bitsandbytes)

**Choice**: A (GPTQ, 4-bit only)
- Phase 2 Extension: 8-bit variant, AWQ, and bitsandbytes-for-inference can be added later if the tradeoff curve becomes interesting to flesh out

**Reasoning**:
- GPTQ is most mature and widely used in production, and (unlike bitsandbytes, which M3's QLoRA already used as a *training-time* trick) produces a standalone deployable checkpoint with inference kernels vLLM (M6) natively supports well
- Quantizes the **M3 fine-tuned model only** (base + LoRA adapter merged via `merge_and_unload()`), not the base model — M5 is about deploying the M4 result, not re-deriving the M2 story
- Dropped the 8-bit variant originally planned here: it doubles the Colab run (quantize + generate + score again) and the risk surface (one more shot at hitting a `gptqmodel` version/VRAM issue) for a data point that mainly adds resolution to the quality/size curve, not a different conclusion. One point (fp16 merged vs. GPTQ-4bit) is enough to tell the MVP story: "compressed to size X, cost Y pp of pass@1." Revisit 8-bit in Phase 2 if that finer curve turns out to matter.
- Realistic scope for M5

---

### **Decision 8: Inference Framework in M6?**

**Problem**: How to deploy and accelerate the quantized models?

**Options**:
- A. vLLM (industry standard, PagedAttention, mature ecosystem)
- B. TensorRT-LLM (fastest, NVIDIA official, steeper learning curve)
- C. Ollama (simplest, lightweight, less performant)

**Choice**: A (vLLM)

**Reasoning**:
- Standard in large-scale LLMOps deployments
- Natural progression from MVP inference
- Excellent support for quantized models
- Most directly relevant to job interview discussions

---

### **Decision 9: Cost Analysis Scope in M7?**

**Problem**: What's the financial value of this approach?

**Options**:
- A. **Local inference vs API calls** (single request cost + annual TCO)
- B. Different fine-tuning strategies cost (7B vs 3B vs no fine-tuning)
- C. Full lifecycle TCO (development + ops + hardware)

**Choice**: A (Local vs API cost modeling)

**Reasoning**:
- Most direct business case for LLMOps hiring managers
- Easiest to understand and extend
- Directly answers: "When is self-hosted cheaper than API?"
- Scalable argument: single request cost × QPS × days = annual cost

---

### **Decision 10: API Models for Cost Comparison?**

**Problem**: Which LLM APIs to benchmark against?

**Options**: (Selected for coverage across price tiers, regional focus, and availability)

**Choice**: Comprehensive comparison table including:
1. **Self-hosted baseline**: Llama 2-7B + vLLM (quantized)
2. **Premium US**: GPT-4, Claude 3.5 Sonnet
3. **Mid-tier**: Deepseek, Qwen-3.7
4. **Cost-optimized**: Minimax-3, GLM-4

**Time Horizon**: Annual cost (industry standard for TCO decisions)

**Reasoning**:
- Full spectrum: from $0.0001/request (self-hosted) to $0.03 (GPT-4)
- Regional coverage: US, China, globally available options
- Fair comparison: apples-to-apples on code generation quality + cost
- Future-proof: easy to swap API keys as they become available

---

## Implementation Roadmap

(To be updated as each module completes)

- [x] **M1: Data Preparation** — 164-problem HumanEval split into disjoint 50/50 train/eval sets (see `CLAUDE.md` for details, `data/scripts/prepare_data.py`)
- [ ] M2: Baseline Testing
- [ ] M3: LoRA Fine-tuning
- [ ] M4: Fine-tuned Evaluation

---

## Blog Structure

1. **Part 1**: Why Code Generation? Why Llama 2-7B? Design decisions breakdown
2. **Part 2**: MVP Results — Fine-tuning effect size and metrics
3. **Part 3**: Production Optimization — Quantization and inference speedup
4. **Part 4**: Cost Analysis — Fine-tune vs API calls trade-off and recommendations

---

## References

- [HumanEval](https://github.com/openai/human-eval)
- [Llama 2 Paper](https://arxiv.org/abs/2307.09288)
- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)
- [vLLM: Easy, Fast, and Cheap LLM Serving with PagedAttention](https://arxiv.org/abs/2309.06180)
