"""ALiBi positions and feed-forward activation variants: validated against the float64 reference."""

import pytest
import torch

from oglodon import Oglodon, OglodonConfig
from oglodon.validation import validate_model
from oglodon.verify import verify_causal_lm

BASE = dict(vocab_size=48, hidden_size=48, num_layers=2, num_heads=6, num_kv_heads=2, max_seq_len=32,
            pad_token_id=0, eos_token_id=1)  # fmt: skip

VARIANTS = {
    "alibi": dict(use_rope=False, use_alibi=True),
    "alibi_mha_layernorm": dict(use_rope=False, use_alibi=True, num_kv_heads=6, use_rmsnorm=False),
    "alibi_window": dict(use_rope=False, use_alibi=True, sliding_window=5),
    "geglu": dict(ffn_activation="gelu"),
    "relu2_gated": dict(ffn_activation="relu2"),
    "relu2_plain": dict(use_swiglu=False, ffn_activation="relu2"),
    "silu_plain": dict(use_swiglu=False, ffn_activation="silu"),
}


@pytest.mark.parametrize("name", VARIANTS)
@pytest.mark.parametrize("impl", ["eager", "sdpa"])
def test_variants_pass_validation_and_verification(name, impl):
    cfg = OglodonConfig(**{**BASE, **VARIANTS[name], "attn_impl": impl})
    torch.manual_seed(0)
    model = Oglodon(cfg)
    rep = validate_model(model, cfg, device="cpu")
    bad = [c.name for c in rep.checks if c.status.name == "FAIL"]
    assert not bad and rep.passed, bad
    vrep = verify_causal_lm(model, vocab_size=cfg.vocab_size, device="cpu")
    assert not [c.name for c in vrep.checks if c.status.name == "FAIL"]


def test_defaults_unchanged_and_activation_resolution():
    cfg = OglodonConfig(vocab_size=10, hidden_size=16, num_heads=2)
    assert cfg.ffn_activation is None and cfg.ffn_act == "silu" and not cfg.use_alibi
    assert OglodonConfig(vocab_size=10, hidden_size=16, num_heads=2, use_swiglu=False).ffn_act == "gelu"
    # a variant switching use_swiglu must not inherit the base model's resolved activation
    assert cfg.to_dict()["ffn_activation"] is None


def test_alibi_has_no_position_parameters_and_bad_combinations_fail():
    a = Oglodon(OglodonConfig(**{**BASE, **VARIANTS["alibi"]}))
    assert not any("position" in n for n, _ in a.named_parameters())
    learned = Oglodon(OglodonConfig(**{**BASE, "use_rope": False}))
    assert sum(p.numel() for p in learned.parameters()) - sum(p.numel() for p in a.parameters()) == 32 * 48
    with pytest.raises(ValueError, match="use_alibi"):
        OglodonConfig(**{**BASE, "use_alibi": True})  # RoPE still on
    with pytest.raises(ValueError, match="ffn_activation"):
        OglodonConfig(**{**BASE, "ffn_activation": "swish"})


def test_alibi_prefers_nearby_tokens():
    """With identical keys, ALiBi must put more attention on closer positions (and steeper heads decay faster)."""
    cfg = OglodonConfig(**{**BASE, **VARIANTS["alibi"], "num_kv_heads": 6})
    m = Oglodon(cfg).eval()
    ids = torch.zeros(1, 10, dtype=torch.long)  # identical tokens -> identical scores before the bias
    w = m(ids, output_attentions=True).attentions[0][0]  # (H, T, T)
    last = w[:, -1]  # attention of the last query
    assert (last[:, 1:] >= last[:, :-1] - 1e-9).all()  # monotone in recency
    assert (
        last[0, -1] > last[3, -1]
    )  # head 0 has the steepest slope (0.25 vs 0.0039): more weight on the newest token


def test_alibi_roundtrip_and_old_configs_load(tmp_path):
    cfg = OglodonConfig(**{**BASE, **VARIANTS["alibi"], "ffn_activation": "relu2"})
    m = Oglodon(cfg).eval()
    m.save_pretrained(tmp_path)
    m2 = Oglodon.from_pretrained(tmp_path)
    x = torch.randint(0, 48, (2, 9))
    assert torch.equal(m(x), m2(x))
    old = {k: v for k, v in cfg.to_dict().items() if k not in ("use_alibi", "ffn_activation")}
    assert OglodonConfig.from_dict(old).ffn_activation is None  # configs saved before these fields still load
