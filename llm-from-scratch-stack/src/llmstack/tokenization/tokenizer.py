from __future__ import annotations

from pathlib import Path

from tokenizers import Tokenizer as HFTokenizer


class Tokenizer:
    def __init__(self, artifact_dir: str | Path):
        base = Path(artifact_dir)
        tok_path = base / "tokenizer.json"
        if not tok_path.exists():
            raise FileNotFoundError(f"tokenizer.json not found in {base}")
        self._tok = HFTokenizer.from_file(str(tok_path))
        self._special = {}
        for t in ["<pad>", "<bos>", "<eos>", "<unk>"]:
            tid = self._tok.token_to_id(t)
            if tid is None:
                raise ValueError(f"missing special token {t}")
            self._special[t] = tid

    @property
    def vocab_size(self) -> int:
        return self._tok.get_vocab_size()

    @property
    def pad_id(self) -> int:
        return self._special["<pad>"]

    @property
    def bos_id(self) -> int:
        return self._special["<bos>"]

    @property
    def eos_id(self) -> int:
        return self._special["<eos>"]

    @property
    def unk_id(self) -> int:
        return self._special["<unk>"]

    def encode(self, text: str) -> list[int]:
        return self._tok.encode(text).ids

    def decode(self, ids: list[int]) -> str:
        return self._tok.decode(ids)
