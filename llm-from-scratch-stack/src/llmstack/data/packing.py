from __future__ import annotations

from collections.abc import Iterable, Iterator


def pack_tokens_to_blocks(tokens_iter: Iterable[list[int]], block_size: int, drop_remainder: bool = True) -> Iterator[list[int]]:
    buf: list[int] = []
    for tokens in tokens_iter:
        buf.extend(tokens)
        while len(buf) >= block_size:
            yield buf[:block_size]
            buf = buf[block_size:]
    if buf and not drop_remainder:
        yield buf


def make_loss_mask(length: int, target_len: int) -> list[int]:
    return [1] * length + [0] * (target_len - length)
