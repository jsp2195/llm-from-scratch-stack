import torch
from types import SimpleNamespace

from llmstack.model.gpt import GPTModel


def test_model_shapes():
    cfg = SimpleNamespace(vocab_size=128, n_layers=2, n_heads=2, d_model=32, d_ff=64, max_seq_len=16, dropout=0.0, norm_type='layernorm', rope_enabled=False, tie_embeddings=True, gradient_checkpointing=False)
    m = GPTModel(cfg)
    x = torch.randint(0, 128, (2, 16))
    logits, loss = m(x, labels=x)
    assert logits.shape == (2, 16, 128)
    assert torch.isfinite(loss)
    assert sum(p.numel() for p in m.parameters()) > 0
