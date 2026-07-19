# TODO

Deferred action items that came out of grilling / planning sessions. Items here should get done before the phase they block.

---

（M1 evaluation-set 扩容 + M2 max_new_tokens 提升已直接在 dev/t 上完成，见 commit 历史。）

- **`notebooks/run_train_lora_colab.ipynb` 和 `run_finetuned_colab.ipynb` 疑似已废弃**：M2/M3/M4 实际是合并跑在 `run_baseline_colab.ipynb`（含执行输出）里的，这两个分开的 launcher notebook 体积很小、看着从没跑出过 output——需要确认是否该删掉，还是保留作为"分模块跑"的备选路径。
