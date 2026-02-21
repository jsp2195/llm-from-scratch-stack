from tokenizers import Tokenizer as HFTokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.normalizers import NFKC

from llmstack.tokenization.tokenizer import Tokenizer


def test_tokenizer_roundtrip(tmp_path):
    data = tmp_path / "d.txt"
    data.write_text("hello world\nsmall corpus\n")

    tok = HFTokenizer(BPE(unk_token="<unk>"))
    tok.normalizer = NFKC()
    tok.pre_tokenizer = ByteLevel(add_prefix_space=True)
    tok.decoder = ByteLevelDecoder()

    tok.train(
        [str(data)],
        trainer=BpeTrainer(
            vocab_size=100,
            special_tokens=["<pad>", "<bos>", "<eos>", "<unk>"],
        ),
    )

    out = tmp_path / "tok"
    out.mkdir()
    tok.save(str(out / "tokenizer.json"))

    t = Tokenizer(out)

    text = "hello world"
    ids = t.encode(text)
    decoded = t.decode(ids)

    assert len(ids) > 0
    assert decoded.strip() == text
    assert t.pad_id >= 0
    assert t.bos_id >= 0
    assert t.eos_id >= 0
    assert t.unk_id >= 0
