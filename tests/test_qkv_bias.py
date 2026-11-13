"""qkv_bias: new, optional, and invisible when off (backward compatibility)."""

import json

import torch

from oglodon import Oglodon, OglodonConfig
from oglodon.validation import validate_model
from oglodon.verify import verify_causal_lm


def cfg(**kw):
    return OglodonConfig(
        vocab_size=37, hidden_size=32, num_layers=2, num_heads=4, num_kv_heads=2, max_seq_len=48, **kw
    )


def test_default_is_unchanged_no_biases_same_parameter_names(tmp_path):
    m = Oglodon(cfg())
    assert not any(n.endswith("bias") for n, _ in m.named_parameters())
    old = {
        k: v for k, v in cfg().to_dict().items() if k != "qkv_bias"
    }  # a config saved before the option existed
    (tmp_path / "c.json").write_text(json.dumps(old))
    assert OglodonConfig.load(tmp_path / "c.json").qkv_bias is False


def test_bias_model_has_exactly_qkv_biases_zero_initialised():
    m = Oglodon(cfg(qkv_bias=True))
    names = [n for n, _ in m.named_parameters() if n.endswith("bias")]
    assert len(names) == 2 * 3 and all("q_proj" in n or "k_proj" in n or "v_proj" in n for n in names)
    assert all(float(p.detach().abs().max()) == 0 for n, p in m.named_parameters() if n.endswith("bias"))


def test_bias_model_checkpoint_roundtrip_and_full_validation(tmp_path):
    torch.manual_seed(0)
    m = Oglodon(cfg(qkv_bias=True))
    with torch.no_grad():
        for n, p in m.named_parameters():
            if n.endswith("bias"):
                p.normal_(0, 0.3)
    m.save_pretrained(tmp_path / "ck")
    back = Oglodon.from_pretrained(tmp_path / "ck").eval()
    x = torch.randint(0, 37, (2, 9))
    assert torch.equal(m.eval()(x), back(x))
    r = validate_model(m)  # reference implementation, cache, training, resume... with biases
    assert r.passed and not r.failures, str(r)
    v = verify_causal_lm(m)
    assert v.passed and not v.failures, str(v)


def test_invalid_value_rejected():
    import pytest

    with pytest.raises(ValueError, match="qkv_bias"):
        cfg(qkv_bias="yes")
