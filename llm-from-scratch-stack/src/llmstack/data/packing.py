"""Sequence packing utilities."""

from collections.abc import Iterable, Iterator

import torch


def pack_tokens_to_blocks(
    tokens_iter: Iterable[int], block_size: int, drop_remainder: bool = True
) -> Iterator[torch.Tensor]:
    """Pack a token stream into fixed-size blocks."""
    buf: list[int] = []
    for token in tokens_iter:
        buf.append(int(token))
        while len(buf) >= block_size:
            chunk = buf[:block_size]
            buf = buf[block_size:]
            yield torch.tensor(chunk, dtype=torch.long)
    if not drop_remainder and buf:
        padded = buf + [0] * (block_size - len(buf))
        yield torch.tensor(padded, dtype=torch.long)


def build_loss_mask(input_ids: torch.Tensor, pad_token_id: int = 0) -> torch.Tensor:
    return (input_ids != pad_token_id).float()
