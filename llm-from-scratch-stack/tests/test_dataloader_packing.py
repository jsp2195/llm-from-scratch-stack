from llmstack.data.packing import make_loss_mask, pack_tokens_to_blocks


def test_packing():
    blocks = list(pack_tokens_to_blocks([[1,2,3],[4,5],[6,7,8]], 4, drop_remainder=False))
    assert blocks[0] == [1,2,3,4]
    assert blocks[1] == [5,6,7,8]
    mask = make_loss_mask(3, 5)
    assert mask == [1,1,1,0,0]
