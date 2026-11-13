import json

import pytest
import torch
import torch.nn.functional as F

from oglodon import Oglodon
from oglodon.cli import main
from oglodon.quantization import (
    Int8Linear,
    is_quantized,
    model_nbytes,
    quantization_report,
    quantize_int8,
)
from oglodon.tokenization import CharTokenizer
from oglodon.validation import validate_model
from oglodon.validation.report import Status
from oglodon.verify import verify_causal_lm

from .conftest import tiny_config


def _linear(out=8, inp=64, bias=True, seed=0):
    torch.manual_seed(seed)
    return torch.nn.Linear(inp, out, bias=bias)


def test_per_channel_roundtrip_error_is_half_a_step():
    lin = _linear()
    with torch.no_grad():
        lin.weight[0, 3] = 40.0  # outlier row
        lin.weight[1].zero_()  # all-zero row
    q = Int8Linear.from_linear(lin)
    err = (lin.weight - q.dequantize_weight()).abs()
    assert (err <= q.scale / 2 + 1e-7).all()
    assert torch.equal(q.dequantize_weight()[1], torch.zeros(64))
    assert q.qweight.dtype == torch.int8 and int(q.qweight.min()) >= -127


def test_per_group_is_at_least_as_accurate_with_outliers():
    lin = _linear(out=16, inp=128)
    with torch.no_grad():
        lin.weight[:, 0] = 30.0
    e_ch = (lin.weight - Int8Linear.from_linear(lin).dequantize_weight()).norm()
    e_gr = (lin.weight - Int8Linear.from_linear(lin, group_size=16).dequantize_weight()).norm()
    assert e_gr < e_ch


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("bias", [True, False])
def test_forward_equals_linear_with_dequantized_weight(dtype, bias):
    lin = _linear(bias=bias)
    q = Int8Linear.from_linear(lin, group_size=32).to(dtype)
    x = torch.randn(3, 5, 64, dtype=dtype)
    ref = F.linear(x, q.dequantize_weight(dtype), None if q.bias is None else q.bias.to(dtype))
    assert torch.equal(q(x), ref)
    assert q(x).dtype == dtype


def test_group_size_must_divide():
    with pytest.raises(ValueError, match="does not divide"):
        Int8Linear(64, 8, False, group_size=48)
    model = Oglodon(tiny_config())
    with pytest.raises(ValueError, match="must divide"):
        quantize_int8(model, group_size=24)
    with pytest.raises(ValueError, match="must divide"):
        quantize_int8(model, group_size=0)


def test_embeddings_and_tied_head_stay_float(model):
    q = quantize_int8(model)
    assert is_quantized(q) and not is_quantized(model)  # original untouched
    assert q.lm_head.weight is q.embed.weight if hasattr(q, "embed") else True
    assert all(n.startswith("blocks.") for n in q.quantization_info["modules"])
    assert isinstance(q.lm_head, torch.nn.Linear)


def test_include_lm_head_rules():
    with pytest.raises(ValueError, match="tie_embeddings"):
        quantize_int8(Oglodon(tiny_config()), include_lm_head=True)
    untied = Oglodon(tiny_config(tie_embeddings=False)).eval()
    q = quantize_int8(untied, include_lm_head=True)
    assert isinstance(q.lm_head, Int8Linear)
    ids = torch.randint(2, 37, (2, 10))
    assert (q(ids) - untied(ids)).abs().max() < 0.1


def test_double_quantize_and_nothing_to_quantize(model):
    q = quantize_int8(model)
    with pytest.raises(ValueError, match="already quantized"):
        quantize_int8(q)
    with pytest.raises(ValueError, match="no linear"):
        quantize_int8(
            model,
            skip=[
                n
                for n, m in model.named_modules()
                if n.startswith("blocks.") and isinstance(m, torch.nn.Linear)
            ],
        )


def test_output_close_to_float_and_inplace(model):
    ids = torch.randint(2, 37, (2, 12))
    ref = model(ids)
    q = quantize_int8(model, group_size=8)
    assert (q(ids) - ref).abs().max() < 0.05
    same = quantize_int8(model, inplace=True)
    assert same is model and is_quantized(model)


def test_save_load_roundtrip_exact(model, tmp_path):
    q = quantize_int8(model, group_size=8)
    q.save_pretrained(tmp_path)
    assert json.loads((tmp_path / "quantization.json").read_text())["group_size"] == 8
    back = Oglodon.from_pretrained(tmp_path)
    assert is_quantized(back)
    ids = torch.randint(2, 37, (2, 12))
    assert torch.equal(back(ids), q(ids))
    model.save_pretrained(tmp_path / "float")
    assert (tmp_path / "model.safetensors").stat().st_size < (
        tmp_path / "float" / "model.safetensors"
    ).stat().st_size
    assert not (tmp_path / "float" / "quantization.json").exists()
    assert model_nbytes(q) < model_nbytes(model)


def test_generate_cached_equals_uncached(model):
    q = quantize_int8(model)
    ids = torch.randint(2, 37, (1, 5))
    a = q.generate(ids, max_new_tokens=10, use_cache=True, eos_token_id=None)
    b = q.generate(ids, max_new_tokens=10, use_cache=False, eos_token_id=None)
    assert torch.equal(a, b)


def test_verify_has_no_failures_on_quantized_model(model):
    rep = verify_causal_lm(quantize_int8(model))
    failed = [c.name for c in rep.checks if c.status == Status.FAIL]
    assert failed == []


def test_validate_model_refuses_int8(model):
    with pytest.raises(ValueError, match="int8"):
        validate_model(quantize_int8(model))


def test_report_detects_corruption(model):
    q = quantize_int8(model)
    tokens = torch.randint(2, 37, (200,))
    good = quantization_report(model, q, tokens, block_size=20)
    assert good["quantized_layers"] == len(q.quantization_info["modules"])
    assert good["top1_agreement"] > 0.9 and good["size_ratio"] < 1.0
    with torch.no_grad():
        next(m for m in q.modules() if isinstance(m, Int8Linear)).scale.mul_(3.0)
    bad = quantization_report(model, q, tokens, block_size=20)
    assert bad["weight_relative_error_max"] > good["weight_relative_error_max"]
    with pytest.raises(ValueError, match="full window"):
        quantization_report(model, q, tokens[:1], block_size=20)


def test_cli_quantize_then_use(tmp_path, capsys):
    text = "the cat sat on the mat. " * 60
    tok = CharTokenizer.from_text(text)
    torch.manual_seed(0)
    m = Oglodon(tiny_config(vocab_size=tok.vocab_size, max_seq_len=64))
    src, out = tmp_path / "src", tmp_path / "int8"
    m.save_pretrained(src, tokenizer=tok)
    data = tmp_path / "d.txt"
    data.write_text(text)
    assert main(["quantize", str(src), "--out", str(out), "--text", str(data), "--block-size", "32"]) == 0
    rep = json.loads((out / "quantization_report.json").read_text())
    assert rep["file_bytes_quantized"] < rep["file_bytes_reference"] and "perplexity_quantized" in rep
    assert "perplexity" in capsys.readouterr().out
    assert main(["generate", str(out), "--prompt", "the", "--max-new-tokens", "5"]) == 0
    assert main(["eval", str(out), "--text", str(data), "--block-size", "32"]) == 0
    assert main(["quantize", str(out), "--out", str(tmp_path / "again")]) == 2  # already quantized
    assert main(["quantize", str(src), "--out", str(tmp_path / "g"), "--group-size", "7"]) == 2
    assert main(["quantize", str(src), "--out", str(tmp_path / "n")]) == 0
    assert "NOT measured" in capsys.readouterr().out


def test_trained_model_drift_is_small():
    from torch.utils.data import DataLoader

    from oglodon import Trainer
    from oglodon.data import TokenBlockDataset, make_corpus

    text = make_corpus(800)
    tok = CharTokenizer.from_text(text)
    ids = torch.tensor(tok.encode(text))
    torch.manual_seed(0)
    m = Oglodon(tiny_config(vocab_size=tok.vocab_size, hidden_size=64, max_seq_len=64))
    ds = TokenBlockDataset(ids[: int(len(ids) * 0.9)], 64, stride=32)
    Trainer(m, DataLoader(ds, batch_size=16, shuffle=True), max_steps=120, learning_rate=3e-3,
            warmup_steps=10, device="cpu", logger=None).train()  # fmt: skip
    r = quantization_report(m, quantize_int8(m, group_size=16), ids[int(len(ids) * 0.9) :], block_size=64)
    assert abs(r["perplexity_change_percent"]) < 2.0 and r["top1_agreement"] > 0.97
