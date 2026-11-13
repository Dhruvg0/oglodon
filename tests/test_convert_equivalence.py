"""Oglodon vs Hugging Face: two independent implementations of Llama/Mistral with identical weights must agree."""

import pytest
import torch

from oglodon.convert import from_hf_llama, hf_llama_config_to_oglodon
from oglodon.validation import Status
from oglodon.verify import verify_conversion

tr = pytest.importorskip("transformers")


def llama(**kw):
    base = dict(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
    )
    base.update(kw)
    torch.manual_seed(0)
    return tr.LlamaForCausalLM(tr.LlamaConfig(**base)).eval()


def mistral(**kw):
    base = dict(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        sliding_window=5,
    )
    base.update(kw)
    torch.manual_seed(0)
    return tr.MistralForCausalLM(tr.MistralConfig(**base)).eval()


def qwen2(**kw):
    base = dict(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        tie_word_embeddings=True,
    )
    base.update(kw)
    torch.manual_seed(0)
    return tr.Qwen2ForCausalLM(tr.Qwen2Config(**base)).eval()


def stress(m, std=0.2):
    with torch.no_grad():
        for p in m.parameters():
            if p.dim() >= 2:
                p.normal_(0, std)
    return m


@pytest.mark.parametrize(
    "make",
    [
        lambda: llama(),  # GQA
        lambda: llama(num_key_value_heads=4),  # MHA
        lambda: llama(num_key_value_heads=1),  # MQA
        lambda: llama(tie_word_embeddings=True),
        lambda: llama(tie_word_embeddings=False, rope_theta=500.0, rms_norm_eps=1e-5),
        lambda: mistral(),  # sliding window
        lambda: mistral(sliding_window=3),
        lambda: qwen2(),  # q/k/v biases
        lambda: qwen2(tie_word_embeddings=False, num_key_value_heads=1),
    ],
    ids=[
        "gqa",
        "mha",
        "mqa",
        "tied",
        "untied-theta-eps",
        "mistral-window5",
        "mistral-window3",
        "qwen2",
        "qwen2-untied-mqa",
    ],
)
def test_oglodon_logits_equal_hugging_face(make):
    hf = stress(make())
    d = from_hf_llama(hf)
    x = torch.randint(0, 64, (3, 24))
    with torch.no_grad():
        assert (hf(x).logits - d(x)).abs().max() < 1e-4


def test_cross_implementation_report_passes_on_llama_mistral_and_qwen2():
    for m in (llama(), mistral(), qwen2()):
        r = verify_conversion(m, from_hf_llama)
        assert r.passed and r.complete and not r.failures, str(r)
        assert {c.name for c in r.checks} >= {
            "equivalence.logits_stress_weights",
            "equivalence.greedy_generation",
        }


def test_a_wrong_rope_conversion_is_caught():
    """Skip the head re-ordering: the two implementations then rotate different channel pairs."""
    import oglodon.convert.hf as h

    real = h._rope_rows
    h._rope_rows = lambda w, n: w
    try:
        r = verify_conversion(llama(), from_hf_llama)
    finally:
        h._rope_rows = real
    assert r.get("equivalence.logits_stress_weights").status is Status.FAIL and not r.passed


def test_a_wrong_oglodon_scale_is_caught(monkeypatch):
    """A Oglodon attention with the wrong softmax scale must disagree with Hugging Face."""
    import torch.nn.functional as F

    hf = stress(llama())
    m = from_hf_llama(hf)
    m.set_attn_impl("sdpa")
    x = torch.randint(0, 64, (2, 12))
    with torch.no_grad():
        want, ok = hf(x).logits, m(x)  # both computed before the bug is planted
        orig = F.scaled_dot_product_attention
        monkeypatch.setattr(
            F, "scaled_dot_product_attention", lambda *a, **k: orig(*a, **{**k, "scale": 0.5})
        )
        bad = m(x)
    assert (want - ok).abs().max() < 1e-4
    assert (want - bad).abs().max() > 1e-2


def test_unsupported_architectures_are_skipped_not_passed_and_never_silently_converted():
    gpt2 = tr.GPT2LMHeadModel(tr.GPT2Config(vocab_size=50, n_embd=32, n_layer=1, n_head=4, n_positions=32))
    r = verify_conversion(gpt2, from_hf_llama)
    assert not r.passed and all(c.status is Status.SKIPPED for c in r.checks)
    with pytest.raises(NotImplementedError, match="llama, mistral and qwen2"):
        hf_llama_config_to_oglodon(gpt2.config)
    with pytest.raises(NotImplementedError, match="bias"):
        hf_llama_config_to_oglodon(llama(attention_bias=True).config)
    with pytest.raises(NotImplementedError, match="rope"):
        hf_llama_config_to_oglodon(
            llama(rope_parameters={"rope_type": "linear", "factor": 2.0, "rope_theta": 10000.0}).config
        )


def test_cli_cross_check(tmp_path):
    import json

    from oglodon.cli import main

    cfg = tmp_path / "llama.json"
    cfg.write_text(
        json.dumps(
            {
                "model_type": "llama",
                "vocab_size": 64,
                "hidden_size": 32,
                "intermediate_size": 64,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "max_position_embeddings": 64,
            }
        )
    )
    out = tmp_path / "r.json"
    assert main(["verify", "--hf-config", str(cfg), "--cross-check", "--strict", "--out", str(out)]) == 0
    d = json.loads(out.read_text())
    assert d["metadata"]["cross_check"] == "oglodon.convert.from_hf_llama"
    assert any(c["name"] == "equivalence.padded_logits" and c["status"] == "pass" for c in d["checks"])


# ------------------------------------------------------------------------------ trained (not random) weights
def _train(model, steps=120):
    """A real, short training run so the weights have learned structure (sharp attention, non-random scale)."""
    from oglodon.data import make_corpus
    from oglodon.experiments import ExperimentData

    data = ExperimentData.from_text(make_corpus(300), block_size=24)
    assert data.vocab_size <= model.config.vocab_size
    ids = torch.tensor(data.train_ids)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.0)
    g = torch.Generator().manual_seed(0)
    model.train()
    first = last = None
    for _ in range(steps):
        starts = torch.randint(0, len(ids) - 24, (16,), generator=g)
        x = torch.stack([ids[s : s + 24] for s in starts])
        loss = model(input_ids=x, labels=x).loss
        opt.zero_grad()
        loss.backward()
        opt.step()
        first = float(loss.detach()) if first is None else first
        last = float(loss.detach())
    model.eval()
    return first, last


@pytest.mark.parametrize("make", [llama, mistral, qwen2], ids=["llama", "mistral", "qwen2"])
def test_trained_weights_pass_verification_and_match_oglodon(make):
    from oglodon.verify import verify_causal_lm

    m = make(vocab_size=40)
    first, last = _train(m)
    assert last < first - 1.0, f"training did not learn: {first:.2f} -> {last:.2f}"
    rep = verify_causal_lm(m)
    assert rep.passed and not rep.failures, str(rep)
    eq = verify_conversion(m, from_hf_llama)
    assert eq.passed and eq.complete and not eq.failures, str(eq)
    d = from_hf_llama(m)  # the trained weights, converted
    x = torch.randint(0, 40, (4, 24))
    with torch.no_grad():
        assert (m(x).logits - d(x)).abs().max() < 1e-4


def test_a_wrong_bias_permutation_is_caught():
    """Qwen2 biases must be re-ordered like the weights; forgetting that is a real, silent conversion bug."""
    import oglodon.convert.hf as h

    real = h._rope_rows
    h._rope_rows = lambda w, n: real(w, n) if w.dim() == 2 else w
    try:
        r = verify_conversion(qwen2(), from_hf_llama)
    finally:
        h._rope_rows = real
    assert r.get("equivalence.logits_stress_weights").status is Status.FAIL


def test_qwen2_per_layer_sliding_window_is_refused():
    with pytest.raises(NotImplementedError, match="use_sliding_window"):
        hf_llama_config_to_oglodon(
            qwen2(use_sliding_window=True, sliding_window=4, max_window_layers=1).config
        )


def test_low_precision_near_ties_do_not_fail_greedy_equivalence_but_real_disagreement_does():
    """bfloat16 plus large random weights makes free-running greedy decoding diverge at near-ties; that is not a bug."""
    m = llama().to(torch.bfloat16)
    r = verify_conversion(m, from_hf_llama)
    g = r.get("equivalence.greedy_generation")
    assert g.status is Status.PASS, g.detail
    # a genuinely different function must still fail the same check
    import oglodon.convert.hf as h

    real = h._rope_rows
    h._rope_rows = lambda w, n: w
    try:
        bad = verify_conversion(llama(), from_hf_llama)
    finally:
        h._rope_rows = real
    assert bad.get("equivalence.greedy_generation").status is Status.FAIL
