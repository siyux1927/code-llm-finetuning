# CLAUDE.md

Working notes for continuing this project across machines with Claude Code. Design rationale (why HumanEval, why Llama 2-7B, why QLoRA, etc.) lives in `README.md` — read that first for context, this file is just "where things stand" and "how to run things".

## Environment

- Python runs **only** inside the `myenv` micromamba env: `micromamba run -n myenv <cmd>`.
- Deps are tracked in `requirements.txt` (`micromamba run -n myenv pip install -r requirements.txt`).
- Local dev machine is a Mac (Apple Silicon, no CUDA) — only CPU-bound work (data prep, small scripts) runs locally. GPU work (baseline inference, LoRA/QLoRA training, eval generation) needs Colab, per README's VRAM-budget decisions.

## Status

- **M1 (Data Preparation): done.** `data/scripts/prepare_data.py` pulls the 164-problem `openai/openai_humaneval` dataset, shuffles with a fixed seed (42), and splits into two **disjoint** 50-problem sets:
  - `data/processed/train_50.jsonl` — completion-style `{task_id, prompt, completion}`, for fine-tuning.
  - `data/processed/eval_50.jsonl` — `{task_id, prompt, canonical_solution, test, entry_point}`, for M2/M4 pass@1 scoring (kept disjoint from train so fine-tuned improvement reflects generalization, not memorization — see GitHub issue for the reasoning).
- **M2 (Baseline Testing): not started.**
- **M3 (LoRA Fine-tuning): not started.**
- **M4 (Fine-tuned Evaluation): not started.**

## Working conventions

- Interesting problems, ambiguities, or design forks hit during implementation get filed as GitHub issues on this repo (not a formal spec process — just a running log of "things worth remembering"). Issues are written in Chinese.
- No formal spec/ticket pipeline in use for this project — it's a learn-by-building project, developed interactively and incrementally.
