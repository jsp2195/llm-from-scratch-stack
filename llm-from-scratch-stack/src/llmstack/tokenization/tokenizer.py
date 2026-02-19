"""Tokenizer wrapper built on HuggingFace tokenizers."""

from pathlib import Path

from tokenizers import ByteLevelBPETokenizer, Tokenizer

SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]


class TokenizerWrapper:
    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer
        self.pad_id = tokenizer.token_to_id("<pad>") or 0
        self.bos_id = tokenizer.token_to_id("<bos>") or 1
        self.eos_id = tokenizer.token_to_id("<eos>") or 2
        self.unk_id = tokenizer.token_to_id("<unk>") or 3

    @classmethod
    def from_dir(cls, path: str) -> "TokenizerWrapper":
        p = Path(path)
        tok_json = p / "tokenizer.json"
        if tok_json.exists():
            tok = Tokenizer.from_file(str(tok_json))
            return cls(tok)
        vocab = p / "vocab.json"
        merges = p / "merges.txt"
        if vocab.exists() and merges.exists():
            bpe = ByteLevelBPETokenizer(str(vocab), str(merges))
            bpe.add_special_tokens(SPECIAL_TOKENS)
            return cls(bpe._tokenizer)
        raise FileNotFoundError(f"No tokenizer files found under {path}")

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        ids = self.tokenizer.encode(text).ids
        if add_special_tokens:
            return [self.bos_id, *ids, self.eos_id]
        return ids

    def decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size()
