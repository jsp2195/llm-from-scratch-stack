import torch

from llmstack.train.checkpointing import load_checkpoint, save_checkpoint


def test_checkpoint_roundtrip(tmp_path):
    p = tmp_path / "ckpt.pt"
    state = {"a": torch.tensor([1, 2, 3])}
    save_checkpoint(str(p), state)
    loaded = load_checkpoint(str(p))
    assert torch.equal(loaded["a"], state["a"])
