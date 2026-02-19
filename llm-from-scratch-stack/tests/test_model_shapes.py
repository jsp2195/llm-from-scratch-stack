import torch
from omegaconf import OmegaConf

from llmstack.model.gpt import GPTModel


def test_model_forward_shape():
    cfg = OmegaConf.create(
        {
            "model": {
                "vocab_size": 32,
                "n_layers": 2,
                "n_heads": 2,
                "d_model": 16,
                "d_ff": 32,
                "max_seq_len": 8,
                "dropout": 0.0,
                "norm_type": "layernorm",
                "tie_embeddings": True,
            }
        }
    )
    model = GPTModel(cfg)
    x = torch.randint(0, 32, (2, 8))
    out = model(x, x)
    assert out["logits"].shape == (2, 8, 32)
    assert torch.isfinite(out["loss"])
