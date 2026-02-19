"""Hydra config schemas."""

from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    vocab_size: int = 50257
    n_layers: int = 4
    n_heads: int = 4
    d_model: int = 256
    d_ff: int = 1024
    max_seq_len: int = 256
    dropout: float = 0.1
    norm_type: str = "layernorm"
    rope_enabled: bool = False
    tie_embeddings: bool = True


@dataclass
class DataConfig:
    train_path: str = "data/toy/train.jsonl"
    val_path: str = "data/toy/val.jsonl"
    format: str = "jsonl"
    text_field: str = "text"
    seq_len: int = 128
    pack_sequences: bool = True
    num_workers: int = 0
    shuffle: bool = True
    streaming: bool = False


@dataclass
class TrainConfig:
    seed: int = 42
    device: str = "cuda"
    precision: str = "fp32"
    compile: bool = False
    grad_clip: float = 1.0
    grad_accum_steps: int = 1
    global_batch_tokens: int | None = None
    micro_batch_size: int = 8
    max_steps: int = 200
    eval_interval: int = 50
    log_interval: int = 10
    save_interval: int = 50
    out_dir: str = "runs"
    resume_path: str | None = None
    max_eval_batches: int = 50


@dataclass
class OptimConfig:
    name: str = "adamw"
    lr: float = 3e-4
    betas: tuple[float, float] = (0.9, 0.95)
    weight_decay: float = 0.1
    eps: float = 1e-8


@dataclass
class SchedConfig:
    name: str = "warmup_cosine"
    warmup_steps: int = 100
    min_lr: float = 3e-5


@dataclass
class DistConfig:
    mode: str = "single"
    backend: str = "nccl"
    find_unused_params: bool = False
    fsdp_shard_strategy: str = "FULL_SHARD"
    fsdp_cpu_offload: bool = False
    fsdp_mixed_precision: bool = True


@dataclass
class EvalConfig:
    checkpoint_path: str = ""
    max_batches: int = 50
    report_name: str = "eval_report.json"


@dataclass
class SFTConfig:
    data_path: str = "data/toy/sft.jsonl"
    base_checkpoint: str = ""
    separator: str = "\n### Response:\n"


@dataclass
class DPOConfig:
    data_path: str = "data/toy/prefs.jsonl"
    base_checkpoint: str = ""
    beta: float = 0.1


@dataclass
class PostTrainConfig:
    sft: SFTConfig = field(default_factory=SFTConfig)
    dpo: DPOConfig = field(default_factory=DPOConfig)


@dataclass
class RootConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    sched: SchedConfig = field(default_factory=SchedConfig)
    dist: DistConfig = field(default_factory=DistConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    posttrain: PostTrainConfig = field(default_factory=PostTrainConfig)
    tokenizer_dir: str = "artifacts/tokenizer_toy"
    use_wandb: bool = False
    project_name: str = "llm-from-scratch-stack"
