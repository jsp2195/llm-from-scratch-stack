from llmstack.data.packing import pack_tokens_to_blocks


def test_pack_tokens_to_blocks():
    blocks = list(pack_tokens_to_blocks(range(10), block_size=4, drop_remainder=False))
    assert len(blocks) == 3
    assert blocks[0].tolist() == [0, 1, 2, 3]
