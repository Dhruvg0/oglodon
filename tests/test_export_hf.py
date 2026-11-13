"""Oglodon -> Hugging Face export: weights must reproduce the same function, also after a save/load cycle."""

import pytest
import torch

from oglodon import Oglodon, OglodonConfig
from oglodon.convert import export_hf, from_hf_llama, oglodon_config_to_hf, to_hf_llama
from oglodon.validation import Status
from oglodon.verify import verify_conversion

tr = pytest.importorskip("transformers")

BASE = dict(vocab_size=64, hidden_size=32, num_layers=2, num_heads=4, num_kv_heads=2, max_seq_len=64)
CASES = {
    "llama": {},
    "llama_untied": {"tie_embeddings": False},
    "mha": {"num_kv_heads": 4},
    "mqa": {"num_kv_heads": 1},
    "qwen2_bias": {"qkv_bias": True, "tie_embeddings": True},
    "mistral_window": {"sliding_window": 5},
    "rope_base": {"rope_base": 500000.0, "norm_eps": 1e-5},
}


def stressed(**kw):
    torch.manual_seed(0)
    m = Oglodon(OglodonConfig(**{**BASE, **kw})).eval()
    with torch.no_grad():  # large random weights (also biases) so that every path matters
        for p in m.parameters():
            p.add_(torch.randn_like(p) * 0.15)
    return m


@pytest.mark.parametrize("name", CASES)
def test_exported_model_matches_logits_loss_and_generation(name):
    m = stressed(**CASES[name])
    hf = to_hf_llama(m)
    x = torch.randint(0, 64, (3, 14))
    with torch.no_grad():
        assert (m(x) - hf(x).logits).abs().max() < 5e-6
        assert torch.allclose(m(x, labels=x).loss, hf(x, labels=x).loss, atol=1e-6)
        mask = torch.ones(3, 14, dtype=torch.long)
        mask[0, :4] = 0
        pos = (mask.cumsum(-1) - 1).clamp(min=0)
        a = m(x, attention_mask=mask, position_ids=pos)
        b = hf(x, attention_mask=mask, position_ids=pos).logits
        assert (a - b).abs()[mask.bool()].max() < 5e-6  # compared on real-token rows
    ours = m.generate(x[:, :6], max_new_tokens=8, eos_token_id=None)
    # explicit all-ones mask: with pad_token_id set, HF would otherwise treat a prompt token equal to it as padding
    theirs = hf.generate(
        x[:, :6],
        attention_mask=torch.ones(3, 6, dtype=torch.long),
        max_new_tokens=8,
        do_sample=False,
        pad_token_id=0,
    )
    assert torch.equal(ours, theirs)


@pytest.mark.parametrize("name", ["llama", "qwen2_bias", "mistral_window"])
def test_roundtrip_oglodon_to_hf_and_back_is_exact(name):
    m = stressed(**CASES[name])
    back = from_hf_llama(to_hf_llama(m))
    for (n, p), (_, q) in zip(m.state_dict().items(), back.state_dict().items(), strict=True):
        assert torch.equal(p, q), n
    rep = verify_conversion(to_hf_llama(m), from_hf_llama)
    assert not [c.name for c in rep.checks if c.status is Status.FAIL]


def test_export_to_disk_and_reload_with_auto_model(tmp_path):
    m = stressed(**CASES["qwen2_bias"])
    export_hf(m, tmp_path)
    reloaded = tr.AutoModelForCausalLM.from_pretrained(tmp_path).eval()
    x = torch.randint(0, 64, (2, 10))
    with torch.no_grad():
        assert (m(x) - reloaded(x).logits).abs().max() < 5e-6


@pytest.mark.parametrize(
    "kw, reason",
    [
        ({"use_rope": False}, "positions"),
        ({"use_rope": False, "use_alibi": True}, "positions"),
        ({"use_rmsnorm": False}, "LayerNorm"),
        ({"use_swiglu": False}, "feed-forward"),
        ({"ffn_activation": "gelu"}, "feed-forward"),
        ({"qkv_bias": True, "sliding_window": 4}, "sliding window"),
    ],
)
def test_unsupported_architectures_refuse_instead_of_exporting_something_else(kw, reason):
    with pytest.raises(NotImplementedError, match=reason):
        oglodon_config_to_hf(OglodonConfig(**{**BASE, **kw}))
