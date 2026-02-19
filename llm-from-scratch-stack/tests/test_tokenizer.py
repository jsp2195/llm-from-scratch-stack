from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from llmstack.tokenization.tokenizer import TokenizerWrapper


def test_tokenizer_roundtrip(tmp_path):
    vocab = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3, "hello": 4, "world": 5}
    tok = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
    tok.pre_tokenizer = Whitespace()
    path = tmp_path / "tokenizer.json"
    tok.save(str(path))
    w = TokenizerWrapper.from_dir(str(tmp_path))
    ids = w.encode("hello world")
    assert ids[0] == w.bos_id
    assert ids[-1] == w.eos_id
    assert "hello" in w.decode(ids)
