# llm-from-scratch-stack

A compact, reproducible, offline-first LLM training stack covering **pretraining, evaluation, SFT, and DPO**.

## What it is / isn't
- ✅ Educational, production-quality small-scale stack with reproducibility and CI.
- ❌ Not a frontier-scale training system.

## Install
```bash
pip install -e .
pip install -e ".[datasets,wandb]"  # optional extras
```

## Quickstart (offline)
```bash
python scripts/train_tokenizer.py data.train_path=data/toy/train.jsonl out_dir=artifacts/tokenizer_toy
python scripts/pretrain.py config-name=pretrain model=gpt_small data=toy train.max_steps=200
python scripts/evaluate.py eval.checkpoint_path=<path_to_last.ckpt> eval.max_batches=50
python scripts/sft.py sft.data_path=data/toy/sft.jsonl sft.base_checkpoint=<path_to_last.ckpt>
python scripts/dpo.py dpo.data_path=data/toy/prefs.jsonl dpo.base_checkpoint=<path_to_last.ckpt>
```

Hydra overrides are supported, e.g.
```bash
python scripts/pretrain.py train.max_steps=1000 model.n_layers=6
```

## Reproducibility
Each run stores `config.yaml`, `manifest.json` (timestamp, host, git sha, pip freeze, command), JSONL logs, and TensorBoard logs.
Checkpoints save model/optimizer/scheduler/scaler/step/RNG state.

## Logging and metrics
- `metrics.jsonl`: scalar logs (loss, lr, tokens/sec, grad_norm, val_loss).
- TensorBoard under `tb/`.
- Validation perplexity uses distributed-safe token-weighted reduction path in evaluation utilities.

## Safety and compliance
This repository includes only synthetic toy data. Do not train on sensitive/proprietary data without consent, policy review, and legal checks.

## Roadmap
- richer packed-sequence boundary masking
- stronger probe tasks
- broader FSDP checkpoint modes
- benchmark scripts for multi-node
