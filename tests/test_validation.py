"""The validator must (a) pass on correct models and (b) FAIL when a bug is planted.

Part (b) is the important one: each test below injects one realistic bug and asserts that the
named checks notice it. A validator that has never been seen to fail proves nothing.
"""

import json

import pytest
import torch
import torch.nn.functional as F

import oglodon.layers.attention as attn_mod
from oglodon import Oglodon
from oglodon.layers import KVCache
from oglodon.tokenization import CharTokenizer
from oglodon.training import Trainer
from oglodon.validation import REGISTRY, Status, ValidationReport, validate_model
from oglodon.validation import reference as ref

from .conftest import tiny_config

ARCH = [
    dict(),
    dict(num_kv_heads=4),
    dict(num_kv_heads=1),
    dict(use_rope=False, use_rmsnorm=False, use_swiglu=False, tie_embeddings=False),
    dict(sliding_window=3),
    dict(attn_impl="sdpa", dropout=0.1),
]


def run(model=None, config=None, **kw):
    return validate_model(model or Oglodon(tiny_config()), config, **kw)


def statuses(report):
    return {c.name: c.status for c in report.checks}


# ----------------------------------------------------------------------- correct models pass
@pytest.mark.parametrize("arch", ARCH, ids=lambda a: ",".join(f"{k}={v}" for k, v in a.items()) or "default")
def test_correct_models_have_no_failures_or_warnings(arch):
    torch.manual_seed(0)
    report = validate_model(Oglodon(tiny_config(**arch)))
    assert report.passed, str(report)
    assert not report.failures and not report.warnings, str(report)
    assert report.counts()["pass"] >= 19


def test_every_registered_check_is_reported():
    report = run()
    assert [c.name for c in report.checks] == [s.name for s in REGISTRY]
    assert {s.category for s in REGISTRY} == {"attention", "kv_cache", "model", "training"}


def test_validation_does_not_modify_the_model_or_its_mode():
    m = Oglodon(tiny_config()).train()
    before = {k: v.clone() for k, v in m.state_dict().items()}
    validate_model(m)
    assert m.training and all(p.grad is None for p in m.parameters())
    assert all(torch.equal(before[k], v) for k, v in m.state_dict().items())


# ----------------------------------------------------------------------- statuses are honest
def test_skipped_unsupported_and_excluded_are_never_counted_as_passed():
    r = run(exclude=("attention.eager",))
    s = statuses(r)
    assert (
        s["attention.eager_vs_sdpa"] is Status.SKIPPED
        and "excluded" in r.get("attention.eager_vs_sdpa").detail
    )
    assert s["model.vocab_and_embeddings"] is Status.SKIPPED  # no tokenizer given
    assert s["training.mixed_precision_fp16"] is Status.UNSUPPORTED  # no CUDA in the test environment
    assert not r.complete and r.passed
    n = r.counts()
    assert n["pass"] + n["fail"] + n["warning"] + n["skipped"] + n["unsupported"] == len(r.checks)
    assert all(c.status is not Status.PASS for c in r.skipped + r.unsupported)


@pytest.mark.skipif(torch.cuda.is_available(), reason="needs a CPU-only environment")
def test_fp16_training_is_reported_unsupported_not_passed_without_cuda():
    c = run(categories=["training"]).get("training.mixed_precision_fp16")
    assert c.status is Status.UNSUPPORTED and "NOT executed" in c.detail


def test_nothing_run_is_not_a_pass():
    r = run(select=["no.such.check"])
    assert r.checks == [] and not r.passed
    assert not run(categories=["model"], exclude=("model.",)).passed


def test_requesting_unavailable_cuda_raises_instead_of_silently_using_cpu():
    if torch.cuda.is_available():
        pytest.skip("CUDA present")
    with pytest.raises(ValueError, match="CUDA"):
        validate_model(Oglodon(tiny_config()), device="cuda")
    with pytest.raises(ValueError, match="categories"):
        validate_model(Oglodon(tiny_config()), categories=["nope"])
    with pytest.raises(ValueError, match="dtype"):
        validate_model(Oglodon(tiny_config()), dtype="int8")


def test_tokenizer_compatibility_pass_warning_and_fail():
    cfg = tiny_config(vocab_size=20, eos_token_id=1, pad_token_id=0)
    m = Oglodon(cfg)
    ok = CharTokenizer("abcdefghijklmnopqr")  # 2 specials + 18 = 20
    assert (
        run(m, tokenizer=ok, select=["model.vocab"]).get("model.vocab_and_embeddings").status is Status.PASS
    )
    small = CharTokenizer("abc")
    assert (
        run(m, tokenizer=small, select=["model.vocab"]).get("model.vocab_and_embeddings").status
        is Status.WARNING
    )
    big = CharTokenizer("abcdefghijklmnopqrstuvwxyz")
    c = run(m, tokenizer=big, select=["model.vocab"]).get("model.vocab_and_embeddings")
    assert c.status is Status.FAIL and "only 20 embeddings" in c.detail


def test_config_mismatch_is_a_failure():
    m = Oglodon(tiny_config())
    other = tiny_config(num_kv_heads=4)
    c = run(m, other, select=["model.config"]).get("model.config_matches")
    assert c.status is Status.FAIL and "num_kv_heads" in c.detail
    assert run(m, m.config, select=["model.config"]).get("model.config_matches").passed


def test_crashing_check_is_a_failure_not_a_pass(monkeypatch):
    monkeypatch.setattr(Oglodon, "forward", lambda self, *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    r = run(select=["model.shape"])
    assert (
        r.get("model.shape_invariants").status is Status.FAIL
        and "boom" in r.get("model.shape_invariants").detail
    )


# ----------------------------------------------------------------------- report contents
def test_report_records_diffs_tolerances_device_dtype_and_metadata(tmp_path):
    r = run(select=["attention.matches", "kv_cache.logits"])
    for c in r.checks:
        assert c.max_abs_diff is not None and c.tolerance is not None and 0 <= c.max_abs_diff <= c.tolerance
        assert c.device == "cpu" and c.dtype == "float32" and c.duration_s >= 0
    m = r.metadata
    for key in (
        "oglodon_version",
        "torch_version",
        "python_version",
        "platform",
        "device",
        "dtype",
        "seed",
        "timestamp_utc",
        "config",
        "config_sha256",
        "torch_num_threads",
        "tolerances",
        "categories",
    ):
        assert key in m, key
    assert m["config"]["num_kv_heads"] == 2
    r.save(tmp_path / "r.json")
    back = ValidationReport.load(tmp_path / "r.json")
    assert [c.to_dict() for c in back.checks] == [c.to_dict() for c in r.checks] and back.passed == r.passed
    data = json.loads((tmp_path / "r.json").read_text())
    assert data["counts"]["pass"] == 2 and data["complete"] is True


def test_report_json_is_strict_json_even_with_nan(tmp_path):
    from oglodon.validation.report import CheckResult

    rep = ValidationReport([CheckResult("x", "model", Status.FAIL, max_abs_diff=float("nan"))])
    rep.save(tmp_path / "n.json")
    json.loads((tmp_path / "n.json").read_text(), parse_constant=lambda c: pytest.fail(f"non-JSON {c}"))
    assert ValidationReport.load(tmp_path / "n.json").checks[0].max_abs_diff != 0


def test_validation_is_reproducible():
    a = run(select=["attention.matches", "kv_cache.logits"], seed=3)
    b = run(select=["attention.matches", "kv_cache.logits"], seed=3)
    assert [c.max_abs_diff for c in a.checks] == [c.max_abs_diff for c in b.checks]
    assert a.metadata["config_sha256"] == b.metadata["config_sha256"]


def test_float64_validation_uses_tighter_tolerances_and_still_passes():
    r = run(select=["attention.matches", "model.reference", "kv_cache.logits"], dtype="float64")
    assert r.passed and all(c.dtype == "float64" and c.tolerance < 1e-6 for c in r.checks), str(r)


def test_bfloat16_validation_runs_and_is_labelled():
    r = run(select=["model.reference", "kv_cache.logits"], dtype="bfloat16")
    assert all(c.dtype == "bfloat16" for c in r.checks)
    assert r.passed, str(r)


# ----------------------------------------------------------------------- the reference itself
def test_reference_attention_mask_spec():
    m = ref.ref_mask(2, 4, torch.tensor([[1, 1, 0, 1]]), window=None)[0]
    assert m.tolist() == [[True, True, False, False], [True, True, False, True]]
    assert ref.ref_mask(1, 5, None, window=2)[0, 0].tolist() == [False, False, False, True, True]


def test_reference_forward_matches_model_on_all_architectures():
    for arch in ARCH:
        m = Oglodon(tiny_config(init_std=0.3, **arch)).eval()
        x = torch.randint(0, 37, (2, 9))
        got = m(x).double()
        want = ref.reference_forward(ref.reference_model(m), x)
        assert (got - want).abs().max() < 1e-4, arch


# ----------------------------------------------------------------------- planted bugs are caught
def planted(monkeypatch, name):
    """Inject one bug and return the checks (names) that must notice it."""
    orig_bias = attn_mod.build_attention_bias
    if name == "causal_off_by_one":

        def bad(q_len, kv_len, device, dtype, key_padding_mask=None, window=None):
            allowed = torch.ones(q_len, kv_len, dtype=torch.bool, device=device).tril(kv_len - q_len + 1)
            return torch.zeros(1, 1, q_len, kv_len, dtype=dtype, device=device).masked_fill(
                ~allowed, torch.finfo(dtype).min
            )

        monkeypatch.setattr(attn_mod, "build_attention_bias", bad)
        return [
            "attention.causal_mask",
            "attention.no_future_gradient",
            "attention.matches_reference",
            "model.reference_forward_backward",
        ]
    if name == "padding_ignored":
        monkeypatch.setattr(
            attn_mod,
            "build_attention_bias",
            lambda q, k, d, t, pad=None, window=None: orig_bias(q, k, d, t, None, window),
        )
        return ["attention.padding_masks", "attention.matches_reference"]
    if name == "window_ignored":
        monkeypatch.setattr(
            attn_mod,
            "build_attention_bias",
            lambda q, k, d, t, pad=None, window=None: orig_bias(q, k, d, t, pad, None),
        )
        return ["attention.local_mask", "attention.matches_reference"]
    if name == "gqa_wrong_head_mapping":
        orig = torch.Tensor.repeat_interleave

        def bad_ri(self, repeats, dim=None):
            if dim == 1 and isinstance(repeats, int):
                return self.repeat(*[repeats if i == 1 else 1 for i in range(self.dim())])
            return orig(self, repeats, dim=dim)

        monkeypatch.setattr(torch.Tensor, "repeat_interleave", bad_ri)
        return [
            "attention.matches_reference",
            "attention.layouts_vs_reference",
            "model.reference_forward_backward",
        ]
    if name == "rope_sign":
        real = attn_mod.apply_rope
        monkeypatch.setattr(attn_mod, "apply_rope", lambda x, cos, sin: real(x, cos, -sin))
        return [
            "attention.matches_reference",
            "model.reference_forward_backward",
            "kv_cache.logits_vs_reference",
        ]
    if name == "sdpa_wrong_scale":
        real = F.scaled_dot_product_attention
        monkeypatch.setattr(F, "scaled_dot_product_attention", lambda *a, **k: real(*a, scale=0.7, **k))
        return ["attention.eager_vs_sdpa"]
    if name == "cache_position_lost":
        monkeypatch.setattr(KVCache, "seq_len", property(lambda self: 0))
        return ["kv_cache.logits_vs_reference", "kv_cache.multi_step_and_chunking"]
    if name == "cache_reset_noop":
        monkeypatch.setattr(KVCache, "reset", lambda self: None)
        return ["kv_cache.reset_and_batch"]
    if name == "cache_wrong_order":

        def bad_update(self, i, k, v):
            if self.keys[i] is None:
                self.keys[i], self.values[i] = k, v
            else:
                self.keys[i], self.values[i] = (
                    torch.cat([k, self.keys[i]], 2),
                    torch.cat([v, self.values[i]], 2),
                )
            return self.keys[i], self.values[i]

        monkeypatch.setattr(KVCache, "update", bad_update)
        return ["kv_cache.multi_step_and_chunking"]  # single-token steps are permutation-invariant
    if name == "accumulation_unweighted":

        def bad(self, micro):
            self.optimizer.zero_grad(set_to_none=True)
            for b in micro:
                (self.model(**b).loss / len(micro)).backward()
            return 0.0

        monkeypatch.setattr(Trainer, "compute_gradients", bad)
        return ["training.gradient_accumulation"]
    if name == "eval_unweighted":

        def bad(self):
            losses = [
                self.model(**{k: v.to(self.device) for k, v in b.items()}).loss.item()
                for b in self.eval_dataloader
            ]
            return sum(losses) / len(losses)

        monkeypatch.setattr(Trainer, "evaluate", bad)
        return ["training.eval_loss_weighting"]
    if name == "checkpoint_corrupts":
        import oglodon.training.checkpoint as ck

        real = ck.load_file
        monkeypatch.setattr(
            ck, "load_file", lambda p, device="cpu": {k: v + 1e-3 for k, v in real(p, device=device).items()}
        )
        return ["training.checkpoint_roundtrip"]
    if name == "resume_forgets_data_position":
        real = Trainer.load_checkpoint

        def bad(self, path):
            real(self, path)
            self._skip_batches = 0

        monkeypatch.setattr(Trainer, "load_checkpoint", bad)
        return ["training.resume_equivalence"]
    raise KeyError(name)


BUGS = [
    "causal_off_by_one",
    "padding_ignored",
    "window_ignored",
    "gqa_wrong_head_mapping",
    "rope_sign",
    "sdpa_wrong_scale",
    "cache_position_lost",
    "cache_reset_noop",
    "cache_wrong_order",
    "accumulation_unweighted",
    "eval_unweighted",
    "checkpoint_corrupts",
    "resume_forgets_data_position",
]


@pytest.mark.parametrize("bug", BUGS)
def test_validator_catches_planted_bug(monkeypatch, bug):
    must_fail = planted(monkeypatch, bug)
    model = Oglodon(
        tiny_config(init_std=0.02)
    )  # default tiny init: the validator must not depend on a lucky model
    report = validate_model(model, select=must_fail)
    status = statuses(report)
    missed = [n for n in must_fail if status[n] is not Status.FAIL]
    assert not missed, f"{bug}: not caught by {missed}\n{report}"
    assert not report.passed


def test_weight_tying_bug_is_caught_by_parameter_consistency():
    m = Oglodon(tiny_config())
    m.lm_head.weight = torch.nn.Parameter(m.embed_tokens.weight.detach().clone())  # silently untied
    c = validate_model(m, select=["model.parameters"]).get("model.parameters_match_config")
    assert c.status is Status.FAIL and "tie" in c.detail


def test_nan_weights_are_caught():
    m = Oglodon(tiny_config())
    with torch.no_grad():
        m.blocks[0].ffn.w1.weight[0, 0] = float("nan")
    r = validate_model(m, select=["model.finite"])
    assert r.get("model.finite_outputs_and_gradients").status is Status.FAIL


def test_dead_parameters_in_the_given_weights_are_a_warning():
    m = Oglodon(tiny_config())
    with torch.no_grad():
        m.blocks[0].ffn.w2.weight.zero_()
    c = validate_model(m, select=["model.finite"]).get("model.finite_outputs_and_gradients")
    assert c.status is Status.WARNING and "exactly zero" in c.detail


def test_unsupported_when_sdpa_missing(monkeypatch):
    monkeypatch.delattr(F, "scaled_dot_product_attention")
    assert (
        validate_model(Oglodon(tiny_config()), select=["attention.eager"])
        .get("attention.eager_vs_sdpa")
        .status
        is Status.UNSUPPORTED
    )
