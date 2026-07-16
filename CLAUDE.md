# CLAUDE.md

Working notes for continuing this project across machines with Claude Code. Design rationale (why HumanEval, why Llama 2-7B, why QLoRA, etc.) lives in `README.md` — read that first for context, this file is just "where things stand" and "how to run things".

## Environment

- Python runs **only** inside the `myenv` env. Invocation depends on the machine:
  - **Windows PC** (branch `dev/t`): `conda run -n myenv <cmd>`
  - **Mac** (branch `dev/mbp`): `micromamba run -n myenv <cmd>`
- Deps are tracked in `requirements.txt` (install with the machine's respective `<manager> run -n myenv pip install -r requirements.txt`).
- Neither machine has CUDA — only CPU-bound work (data prep, small scripts, JSON/notebook validation, unit tests) runs locally. GPU work (baseline inference, LoRA/QLoRA training, eval generation) needs Colab, per README's VRAM-budget decisions.

## Status

- **M1 (Data Preparation): done.** `data/scripts/prepare_data.py` pulls the 164-problem `openai/openai_humaneval` dataset, shuffles with a fixed seed (42), and splits into two **disjoint** 50-problem sets:
  - `data/processed/train_50.jsonl` — completion-style `{task_id, prompt, completion}`, for fine-tuning.
  - `data/processed/eval_50.jsonl` — `{task_id, prompt, canonical_solution, test, entry_point}`, for M2/M4 pass@1 scoring (kept disjoint from train so fine-tuned improvement reflects generalization, not memorization — see GitHub issue for the reasoning).
- **M2 (Baseline Testing): partially done.**
  - `eval/scoring.py` — pass@1 scoring harness (executes generated code against HumanEval tests, subprocess + timeout sandboxed). Reused unchanged by M4. Unit-tested locally (`eval/tests/test_scoring.py`, `pytest eval/tests/`), no GPU needed.
  - `eval/generate_baseline.py` — generates completions from the un-fine-tuned, **4-bit-quantized** Llama-2-7B (same quantization M3 will use, so M2→M4 improvement isn't conflated with a quantization change — see GitHub issue). **Not yet run** — needs a CUDA GPU (bitsandbytes doesn't work on Mac/MPS) and a Hugging Face token with the Llama 2 license accepted. Run on Colab: `pip install -r requirements.txt -r requirements-colab.txt`, then `huggingface-cli login`, then the script.
  - Still to do: actually run generation on Colab, then score with `eval/scoring.py` to get the baseline pass@1 number.
- **M3 (LoRA Fine-tuning): not started.**
- **M4 (Fine-tuned Evaluation): not started.**

## Working conventions

- Interesting problems, ambiguities, or design forks hit during implementation get filed as GitHub issues on this repo (not a formal spec process — just a running log of "things worth remembering"). Issues are written in Chinese.
- Git commit messages are written in Chinese, and do not include a `Co-Authored-By` trailer.
- No formal spec/ticket pipeline in use for this project — it's a learn-by-building project, developed interactively and incrementally.
