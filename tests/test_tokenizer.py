from tokenizers import Tokenizer as HFTokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

from llmstack.tokenization.tokenizer import Tokenizer


def test_tokenizer_roundtrip(tmp_path):
    data = tmp_path / 'd.txt'
    data.write_text('hello world\nsmall corpus\n')
    tok = HFTokenizer(BPE(unk_token='<unk>'))
    tok.pre_tokenizer = ByteLevel()
    tok.train([str(data)], trainer=BpeTrainer(vocab_size=100, special_tokens=['<pad>', '<bos>', '<eos>', '<unk>']))
    out = tmp_path / 'tok'; out.mkdir()
    tok.save(str(out / 'tokenizer.json'))
    t = Tokenizer(out)
    ids = t.encode('hello world')
    assert len(ids) > 0
    assert t.pad_id >= 0 and t.bos_id >= 0 and t.eos_id >= 0 and t.unk_id >= 0
