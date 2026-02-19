from types import SimpleNamespace

import torch

from llmstack.model.gpt import GPTModel
from llmstack.optim.adamw import build_adamw
from llmstack.optim.schedulers import build_scheduler
from llmstack.train.checkpointing import save_checkpoint


def test_checkpoint_roundtrip(tmp_path):
    cfg = SimpleNamespace(vocab_size=64, n_layers=1, n_heads=2, d_model=16, d_ff=32, max_seq_len=8, dropout=0.0, norm_type='layernorm', rope_enabled=False, tie_embeddings=False, gradient_checkpointing=False)
    m = GPTModel(cfg)
    ocfg = SimpleNamespace(weight_decay=0.0, lr=1e-3, betas=(0.9,0.95), eps=1e-8)
    opt = build_adamw(m, ocfg)
    scfg = SimpleNamespace(name='linear', warmup_steps=0, min_lr=0.1)
    sch = build_scheduler(opt, scfg, 10)
    x = torch.randint(0,64,(2,8))
    _, loss = m(x, labels=x)
    loss.backward(); opt.step(); sch.step()
    p = tmp_path / 'c.ckpt'
    save_checkpoint(str(p), {'model': m.state_dict(), 'optimizer': opt.state_dict(), 'scheduler': sch.state_dict(), 'step': 1})
    m2 = GPTModel(cfg); o2 = build_adamw(m2, ocfg); s2 = build_scheduler(o2, scfg, 10)
    ck = torch.load(p, map_location='cpu')
    m2.load_state_dict(ck['model']); o2.load_state_dict(ck['optimizer']); s2.load_state_dict(ck['scheduler'])
    for a,b in zip(m.parameters(), m2.parameters()):
        assert torch.allclose(a,b)
