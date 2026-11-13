import pytest

from oglodon import OglodonConfig


def test_defaults_and_derived_values():
    c = OglodonConfig(vocab_size=100, hidden_size=64, num_heads=8, num_kv_heads=2)
    assert c.head_dim == 8 and c.attention_type == "GQA"
    assert c.intermediate_size == 176  # ceil(8*64/3 / 8) * 8
    assert (
        OglodonConfig(vocab_size=10, hidden_size=64, num_heads=8, use_swiglu=False).intermediate_size == 256
    )


def test_attention_type_names():
    assert OglodonConfig(vocab_size=9, hidden_size=16, num_heads=4).attention_type == "MHA"
    assert OglodonConfig(vocab_size=9, hidden_size=16, num_heads=4, num_kv_heads=1).attention_type == "MQA"


@pytest.mark.parametrize(
    "kwargs, message",
    [
        (dict(vocab_size=0), "vocab_size"),
        (dict(vocab_size=10, num_layers=0), "num_layers"),
        (dict(vocab_size=10, max_seq_len=-1), "max_seq_len"),
        (dict(vocab_size=10, hidden_size=30, num_heads=4), "divisible by num_heads"),
        (dict(vocab_size=10, hidden_size=32, num_heads=4, num_kv_heads=3), "divisible by num_kv_heads"),
        (dict(vocab_size=10, hidden_size=12, num_heads=4), "even head dimension"),  # head_dim 3 with RoPE
        (dict(vocab_size=10, dropout=1.0), "dropout"),
        (dict(vocab_size=10, eos_token_id=10), "eos_token_id"),
        (dict(vocab_size=True), "vocab_size"),
    ],
)
def test_invalid_configs_raise_clear_errors(kwargs, message):
    with pytest.raises(ValueError, match=message):
        OglodonConfig(**kwargs)


def test_odd_head_dim_ok_without_rope():
    assert OglodonConfig(vocab_size=10, hidden_size=12, num_heads=4, use_rope=False).head_dim == 3


def test_roundtrip_dict_and_file(tmp_path):
    c = OglodonConfig(vocab_size=50, hidden_size=32, num_heads=4, num_kv_heads=2, eos_token_id=3)
    assert OglodonConfig.from_dict(c.to_dict()) == c
    c.save(tmp_path / "c.json")
    assert OglodonConfig.load(tmp_path / "c.json") == c


def test_unknown_key_rejected():
    with pytest.raises(ValueError, match="Unknown config keys"):
        OglodonConfig.from_dict({"vocab_size": 10, "hidden": 4})
