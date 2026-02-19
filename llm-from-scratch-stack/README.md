# llm-from-scratch-stack

A **small but real** end-to-end LLM training stack designed for researchers who want to understand — not just use — modern language model pipelines.

This repository implements a clean, reproducible path from:

* Tokenizer training
* GPT-style autoregressive pretraining
* Evaluation (perplexity + targeted capability probes)
* Supervised fine-tuning (SFT)
* Direct Preference Optimization (DPO)

The goal is not scale. The goal is clarity, rigor, and research leverage.

---

## Why This Exists

Most open-source LLM repos are either:

* Highly abstracted training frameworks where critical details are hidden, or
* Production-heavy distributed systems that obscure learning dynamics.

This stack sits in the middle:

* Small enough to reason about completely
* Structured enough to reflect real frontier training pipelines
* Clean enough to run controlled experiments

It is intended as a research scaffold, not a toy notebook.

---

## Design Principles

* **Transparent architecture** — minimal magic, explicit modules
* **Hydra-configured experiments** — reproducible overrides
* **Experiment manifests** — git SHA, hostname, config snapshot
* **Metrics as first-class artifacts** — JSONL + TensorBoard + structured eval reports
* **Modular boundaries** — tokenizer, model, trainer, eval, post-training separated

---

## What This Is / Isn’t

### This is

* A compact but faithful implementation of a modern LLM lifecycle
* Suitable for scaling-law studies at small scale
* A base for experimentation with optimization, data, and alignment tradeoffs

### This is not

* A frontier-scale distributed training system
* A benchmark-chasing implementation
* A drop-in production inference stack

---

## Repository Structure

```
llm_from_scratch_stack/
├── llmstack/
│   ├── data/
│   ├── model/
│   ├── train/
│   ├── eval/
│   ├── posttrain/
│   └── utils/
├── configs/
├── scripts/
├── artifacts/
└── tests/
```

Each stage of the lifecycle is intentionally isolated to make ablations and substitutions easy.

---

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -e .
```

### Train tokenizer

```bash
python scripts/train_tokenizer.py \
  data.train_path=data/toy/train.jsonl \
  out_dir=artifacts/tokenizer_toy
```

### Pretrain (small GPT)

```bash
python scripts/pretrain.py \
  config-name=pretrain \
  model=gpt_small \
  data=toy \
  train.max_steps=200
```

### Evaluate

```bash
python scripts/evaluate.py \
  eval.checkpoint_path=... \
  eval.max_batches=50
```

### Supervised Fine-Tuning (SFT)

```bash
python scripts/sft.py \
  posttrain.sft.data_path=data/toy/sft.jsonl \
  posttrain.sft.base_checkpoint=...
```

### Direct Preference Optimization (DPO)

```bash
python scripts/dpo.py \
  posttrain.dpo.data_path=data/toy/prefs.jsonl \
  posttrain.dpo.base_checkpoint=...
```

---

## Configuration System

All experiments are Hydra-driven.

Core configuration:

* `configs/pretrain.yaml`
* `configs/model/gpt_small.yaml`
* `configs/model/gpt_medium.yaml`
* `configs/data/toy.yaml`

Example override:

```bash
python scripts/pretrain.py train.max_steps=1000 model.n_layers=6
```

Every run stores the full resolved config inside its run directory.

---

## Reproducibility Guarantees

Each run logs:

* Random seed
* Git SHA
* Timestamp
* Hostname
* Full config snapshot
* CLI command

Metrics:

* `metrics.jsonl`
* TensorBoard logs in `tb/`
* Structured evaluation reports from `scripts/evaluate.py`

The intent is to support small-scale scaling-law or ablation studies with minimal friction.

---

## Evaluation Philosophy

Evaluation is not limited to perplexity.

The eval module supports:

* Validation perplexity
* Toy reasoning probes
* Prompt-format sensitivity checks
* Basic instruction-following assessments

This encourages capability measurement beyond loss curves.

---

## Research Use Cases

This stack is suitable for:

* Scaling experiments across small model sizes
* Optimizer ablations
* Context-length sensitivity studies
* Data filtering experiments
* Post-training tradeoff analysis (SFT vs DPO)

It is intentionally simple enough that the entire lifecycle can be reasoned about without black boxes.

---

## Roadmap

Planned improvements include:

* FlashAttention integration
* Expanded offline capability probes
* Dataset mixture weighting + curriculum support
* Stronger FSDP checkpoint sharding
* Optional LoRA adapters for post-training

---

## Safety and Compliance

This repository is provided for research and educational purposes only. Users are responsible for dataset licensing, policy compliance, and responsible deployment.

---

## Contribution

Pull requests are welcome for:

* Additional eval probes
* Training stability improvements
* New post-training objectives
* Systems-level performance improvements

The long-term goal is a compact but serious research baseline for understanding modern LLM training dynamics.
