# llm-from-scratch-stack

A **small but real** end-to-end LLM training stack for education/research. It covers tokenizer training, GPT-style pretraining, evaluation (perplexity + toy probes), SFT, and DPO.

## What this is / isn't
- ✅ A practical baseline with clean boundaries and reproducibility features.
- ❌ Not a frontier-scale training system.

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -e .
```

Train tokenizer:
```bash
python scripts/train_tokenizer.py data.train_path=data/toy/train.jsonl out_dir=artifacts/tokenizer_toy
```

Pretrain small:
```bash
python scripts/pretrain.py config-name=pretrain model=gpt_small data=toy train.max_steps=200
```

Evaluate:
```bash
python scripts/evaluate.py eval.checkpoint_path=... eval.max_batches=50
```

SFT:
```bash
python scripts/sft.py posttrain.sft.data_path=data/toy/sft.jsonl posttrain.sft.base_checkpoint=...
```

DPO:
```bash
python scripts/dpo.py posttrain.dpo.data_path=data/toy/prefs.jsonl posttrain.dpo.base_checkpoint=...
```

## Configs
- Core config: `configs/pretrain.yaml`
- Model sizes: `configs/model/gpt_small.yaml`, `configs/model/gpt_medium.yaml`
- Data config: `configs/data/toy.yaml`

CLI overrides are supported via Hydra, e.g.:
```bash
python scripts/pretrain.py train.max_steps=1000 model.n_layers=6
```

## Reproducibility
- Explicit seeding (`llmstack.utils.seed`)
- Manifest includes timestamp, hostname, git SHA, command, and config snapshot
- Configs saved per run in run directory

## Logs and metrics
- JSONL: `metrics.jsonl`
- TensorBoard: `tb/`
- Eval report: JSON from `scripts/evaluate.py` including perplexity + probe metrics

## Safety note
This repository is for research/education. Users are responsible for legal, policy, and compliance obligations for datasets and deployments.

## Roadmap
- FlashAttention and better kernel dispatch
- Richer offline capability probes
- Dataset mixtures + weighted sampling
- Stronger FSDP checkpoint sharding support
