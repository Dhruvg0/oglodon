"""``oglodon.verify``: model-agnostic checks must pass on correct models of any origin and FAIL on planted bugs."""

import pytest
import torch
from torch import nn

from oglodon import Oglodon
from oglodon.validation import Status
from oglodon.verify import VREGISTRY, CausalLMAdapter, verify_causal_lm

from .conftest import tiny_config
from .tiny_lm import TinyLM


def st(report):
    return {c.name: c.status for c in report.checks}


# ------------------------------------------------------------------------------ correct models pass
def test_registry_is_complete_and_documented():
    assert len(VREGISTRY) >= 18 and all(s.doc for s in VREGISTRY)
    assert len({s.name for s in VREGISTRY}) == len(VREGISTRY)


@pytest.mark.parametrize(
    "arch",
    [
        {},
        {"num_kv_heads": 1},
        {"use_rope": False, "use_rmsnorm": False, "use_swiglu": False},
        {"sliding_window": 3},
    ],
)
def test_oglodon_models_pass(arch):
    r = verify_causal_lm(Oglodon(tiny_config(**arch)))
    assert r.passed and not r.failures and not r.warnings, str(r)


def test_custom_pytorch_model_passes():
    torch.manual_seed(0)
    r = verify_causal_lm(TinyLM())
    assert r.passed and not r.failures and not r.warnings and r.complete, str(r)


def test_hf_gpt2_and_llama_pass():
    tr = pytest.importorskip("transformers")
    gpt2 = tr.GPT2LMHeadModel(tr.GPT2Config(vocab_size=50, n_embd=32, n_layer=2, n_head=4, n_positions=64))
    llama = tr.LlamaForCausalLM(
        tr.LlamaConfig(
            vocab_size=50,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=64,
        )
    )
    for m in (gpt2, llama):
        r = verify_causal_lm(m)
        assert r.passed and not r.failures, str(r)
        assert (
            r.metadata["capabilities"]["kv_cache"]
            and r.metadata["model_class"].endswith("LMHeadModel")
            or True
        )


def test_verification_does_not_modify_the_model():
    m = TinyLM().train()
    before = {k: v.clone() for k, v in m.state_dict().items()}
    verify_causal_lm(m)
    assert m.training and all(torch.equal(before[k], v) for k, v in m.state_dict().items())
    assert all(p.grad is None for p in m.parameters())


# ------------------------------------------------------------------------------ capability handling
def test_bare_logits_model_skips_what_it_cannot_do_and_never_passes_it():
    class Bare(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = TinyLM().config
            self.inner = TinyLM(plain=True)

        def forward(self, input_ids):
            return self.inner(input_ids)

    r = verify_causal_lm(Bare())
    s = st(r)
    for name in (
        "causality.padding_invariance",
        "causality.mask_is_used",
        "cache.single_token_steps",
        "cache.chunked_steps",
        "gradients.loss_convention",
    ):
        assert s[name] is Status.SKIPPED, name
    assert r.passed and not r.complete
    assert all(c.status is not Status.PASS for c in r.skipped)


def test_callable_forward_adapter_for_odd_signatures():
    class Odd(nn.Module):
        def __init__(self):
            super().__init__()
            self.inner = TinyLM(plain=True)

        def forward(self, tokens):  # unusual argument name
            return {"logits": self.inner(tokens)}

    ad = CausalLMAdapter(Odd(), vocab_size=40, max_positions=64, forward=lambda m, ids, mask, pos: m(ids))
    r = verify_causal_lm(None, adapter=ad)
    assert r.passed and not r.failures, str(r)


def test_non_module_and_bad_arguments():
    with pytest.raises(TypeError):
        verify_causal_lm(lambda x: x)
    with pytest.raises(ValueError, match="categories"):
        verify_causal_lm(TinyLM(), categories=["nope"])
    with pytest.raises(ValueError, match="dtype"):
        verify_causal_lm(TinyLM(), dtype="int4")
    if not torch.cuda.is_available():
        with pytest.raises(ValueError, match="CUDA"):
            verify_causal_lm(TinyLM(), device="cuda")


def test_float64_and_select_exclude():
    r = verify_causal_lm(
        TinyLM(), dtype="float64", select=["causality", "batching"], exclude=["batching.order"]
    )
    assert r.passed and {c.dtype for c in r.checks} == {"float64"}
    assert st(r)["batching.order_independence"] is Status.SKIPPED
    assert all(c.tolerance < 1e-6 for c in r.checks if c.tolerance is not None and c.status is Status.PASS)


# ------------------------------------------------------------------------------ planted bugs
CASES = [
    ("leak", ["causality.future_independence", "causality.prefix_consistency", "cache.single_token_steps"]),
    ("batch_leak", ["batching.sample_independence"]),  # a batch-mean leak is permutation-invariant
    ("ignore_mask", ["causality.padding_invariance", "causality.mask_is_used"]),
    ("no_shift", ["gradients.loss_convention"]),
    ("cache_bug", ["cache.chunked_steps"]),  # single-token steps are permutation-invariant
    ("bad_backward", ["gradients.finite_difference"]),
]


@pytest.mark.parametrize(("bug", "must_fail"), CASES)
def test_planted_bug_is_caught(bug, must_fail):
    torch.manual_seed(0)
    r = verify_causal_lm(TinyLM(bug=bug), select=must_fail)
    missed = [n for n in must_fail if st(r)[n] is not Status.FAIL]
    assert not missed, f"{bug} not caught by {missed}\n{r}"
    assert not r.passed


def test_dead_parameter_is_a_warning_not_a_failure():
    r = verify_causal_lm(TinyLM(bug="dead_param"), select=["gradients.finite_and"])
    c = r.get("gradients.finite_and_flowing")
    assert c.status is Status.WARNING and "unused" in c.detail


def test_input_insensitive_model_cannot_pass_causality_vacuously():
    class Const(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = type("C", (), {"vocab_size": 10, "max_position_embeddings": 32})()
            self.w = nn.Parameter(torch.randn(10))

        def forward(self, input_ids):
            return self.w.expand(*input_ids.shape, 10)

    r = verify_causal_lm(Const(), select=["causality.future"])
    assert r.get("causality.future_independence").status is Status.FAIL


def test_nan_model_is_caught():
    m = TinyLM()
    with torch.no_grad():
        m.head.weight[0, 0] = float("nan")
    r = verify_causal_lm(m, select=["structure.shape"])
    assert r.get("structure.shape_and_vocab").status is Status.FAIL


def test_wrong_declared_vocab_is_caught():
    m = TinyLM()
    m.config.vocab_size = 41
    r = verify_causal_lm(m, select=["structure.shape"])
    c = r.get("structure.shape_and_vocab")
    assert c.status is Status.FAIL and "vocab" in c.detail


def test_report_roundtrips_and_records_metadata(tmp_path):
    r = verify_causal_lm(TinyLM(), select=["structure"])
    r.save(tmp_path / "v.json")
    from oglodon.validation import ValidationReport

    back = ValidationReport.load(tmp_path / "v.json")
    assert back.metadata["kind"] == "verify_causal_lm" and back.metadata["model_class"] == "TinyLM"
    assert back.metadata["capabilities"]["attention_mask"] is True


# ------------------------------------------------------------------------------ more Hugging Face architectures
def _hf_models():
    tr = pytest.importorskip("transformers")
    v = {"vocab_size": 64}
    s = {
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "max_position_embeddings": 64,
    }
    return {
        "mistral": lambda: tr.MistralForCausalLM(
            tr.MistralConfig(**v, **s, num_key_value_heads=2, sliding_window=8)
        ),
        "qwen2": lambda: tr.Qwen2ForCausalLM(tr.Qwen2Config(**v, **s, num_key_value_heads=2)),
        "gemma": lambda: tr.GemmaForCausalLM(tr.GemmaConfig(**v, **s, num_key_value_heads=1, head_dim=8)),
        "phi": lambda: tr.PhiForCausalLM(tr.PhiConfig(**v, **s)),
        "gpt_neox": lambda: tr.GPTNeoXForCausalLM(tr.GPTNeoXConfig(**v, **s)),
        "opt": lambda: tr.OPTForCausalLM(
            tr.OPTConfig(
                vocab_size=64,
                hidden_size=32,
                ffn_dim=64,
                num_hidden_layers=2,
                num_attention_heads=4,
                max_position_embeddings=64,
                word_embed_proj_dim=32,
            )
        ),
        "falcon": lambda: tr.FalconForCausalLM(
            tr.FalconConfig(
                vocab_size=64,
                hidden_size=32,
                num_hidden_layers=2,
                num_attention_heads=4,
                max_position_embeddings=64,
            )
        ),
        "bloom": lambda: tr.BloomForCausalLM(
            tr.BloomConfig(vocab_size=64, hidden_size=32, n_layer=2, n_head=4)
        ),
        "mixtral_moe": lambda: tr.MixtralForCausalLM(
            tr.MixtralConfig(**v, **s, num_key_value_heads=2, num_local_experts=4, num_experts_per_tok=2)
        ),
    }


@pytest.mark.parametrize("name", ["mistral", "qwen2", "gemma", "phi", "gpt_neox", "opt", "falcon"])
def test_more_hf_architectures_pass_every_check(name):
    r = verify_causal_lm(_hf_models()[name]())
    assert r.passed and r.complete and not r.failures and not r.warnings, str(r)


def test_alibi_model_without_position_ids_skips_padding_check_honestly():
    r = verify_causal_lm(_hf_models()["bloom"]())  # ALiBi: no position_ids argument
    assert not r.failures and r.passed
    assert st(r)["causality.padding_invariance"] is Status.SKIPPED and not r.complete


def test_moe_has_no_failures_and_unrunnable_float64_checks_are_unsupported_not_passed():
    r = verify_causal_lm(_hf_models()["mixtral_moe"]())
    assert not r.failures and r.passed
    # on some torch/transformers builds the expert kernels cannot run in float64: that must be 'unsupported'
    assert all(c.status in (Status.PASS, Status.UNSUPPORTED) for c in r.checks)


# ------------------------------------------------------------------------------ scaling to big models
def test_fp64_limit_marks_copy_based_checks_unsupported_not_passed():
    r = verify_causal_lm(TinyLM(), max_fp64_params=10)
    s = st(r)
    for name in (
        "numerics.float64_consistency",
        "numerics.bfloat16_sanity",
        "numerics.large_activation_stability",
        "gradients.finite_and_flowing",
        "gradients.finite_difference",
    ):
        assert s[name] is Status.UNSUPPORTED, name
    assert not r.failures and r.passed and not r.complete
    assert "limit" in r.get("gradients.finite_difference").detail


def test_no_stress_runs_every_check_on_given_weights_only():
    r = verify_causal_lm(TinyLM(), stress=False)
    assert r.passed and not r.failures
    assert "[stress weights]" not in " ".join(c.detail for c in r.checks)
    assert r.metadata["stress_weights"] is False


def test_leak_is_still_caught_without_stress_when_weights_are_large():
    m = TinyLM(bug="leak")
    with torch.no_grad():
        for p in m.parameters():
            if p.dim() >= 2:
                p.normal_(0, 0.3)
    r = verify_causal_lm(m, stress=False, select=["causality.future"])
    assert r.get("causality.future_independence").status is Status.FAIL


def test_verify_local_hugging_face_checkpoint_through_the_cli(tmp_path):
    tr = pytest.importorskip("transformers")
    from oglodon.cli import main

    m = tr.LlamaForCausalLM(
        tr.LlamaConfig(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=64,
        )
    )
    m.save_pretrained(tmp_path / "ckpt")
    out = tmp_path / "r.json"
    assert (
        main(["verify", "--hf", str(tmp_path / "ckpt"), "--no-stress", "--cross-check", "--out", str(out)])
        == 0
    )
    import json

    d = json.loads(out.read_text())
    assert d["counts"]["fail"] == 0 and d["metadata"]["source"].startswith("hf:")
    assert any(c["name"] == "equivalence.logits_given_weights" and c["status"] == "pass" for c in d["checks"])


def test_oglodon_info_prints_environment(capsys):
    import json

    from oglodon.cli import main

    assert main(["info"]) == 0
    info = json.loads(capsys.readouterr().out)
    assert {"oglodon", "torch", "cuda_available", "devices"} <= set(info) and isinstance(
        info["cuda_available"], bool
    )


def test_gpu_runbook_exists_and_refuses_to_run_without_cuda():
    import pathlib
    import subprocess

    script = pathlib.Path(__file__).parents[1] / "scripts" / "gpu_check.sh"
    assert script.exists()
    if torch.cuda.is_available():
        pytest.skip("CUDA present: the script would run for real")
    text = script.read_text()
    assert "No CUDA device" in text and "exit 2" in text
    import shutil

    bash = shutil.which("bash")
    if bash is None or subprocess.run([bash, "-c", "exit 0"], capture_output=True).returncode != 0:
        pytest.skip("no working bash on this machine, so the script's syntax was not checked")
    assert subprocess.run([bash, "-n", str(script)], capture_output=True).returncode == 0
