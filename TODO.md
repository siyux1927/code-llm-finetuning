# TODO

Deferred action items that came out of grilling / planning sessions. Items here should get done before the phase they block.

---

（M1 evaluation-set 扩容 + M2 max_new_tokens 提升已直接在 dev/t 上完成，见 commit 历史。）

- **`notebooks/run_train_lora_colab.ipynb` 和 `run_finetuned_colab.ipynb` 疑似已废弃**：M2/M3/M4 实际是合并跑在 `run_baseline_colab.ipynb`（含执行输出）里的，这两个分开的 launcher notebook 体积很小、看着从没跑出过 output——需要确认是否该删掉，还是保留作为"分模块跑"的备选路径。

- **`train/quantize_gptq.py` 需要一个拆成 cell 的 dev notebook**：M5 实跑时连续踩了好几个环境坑（peft/gptqmodel AWQ 改名、merge 存盘 OOM、量化阶段 disk offload、pack_model 阶段显存耗尽），每次都得重跑整个 `!python train/quantize_gptq.py`（哪怕加了 `--skip-merge`，quantize() 内部失败了也要整个函数重来）——merge/量化都是分钟级的操作，一个环节失败就要重新等好几分钟，debug 效率很低。想法：把脚本内容拆到 notebook 的多个 cell 里（load base → merge → save → load calib → quantize → save 各自一个 cell），失败只需重跑那一个 cell，中间状态（已经加载/merge 好的模型对象）留在 kernel 里不丢。不需要开单独 issue，等 M5 真正跑顺之后再排这个。
