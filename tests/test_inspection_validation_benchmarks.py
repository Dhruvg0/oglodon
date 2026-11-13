import matplotlib
import pytest
import torch

matplotlib.use("Agg")

from oglodon.benchmarks import compare_architectures, estimate_kv_cache_bytes, format_results
from oglodon.inspection import AttentionInspector

from .conftest import tiny_config


def test_inspector_shapes_selection_and_causality(model):
    x = torch.randint(2, 37, (2, 9))
    r = AttentionInspector(model).inspect(x, layer=1)
    assert r.weights.shape == (2, 4, 9, 9) and r.heads == [0, 1, 2, 3]
    assert (
        torch.allclose(r.weights.sum(-1), torch.ones(2, 4, 9), atol=1e-5) and (r.weights.triu(1) == 0).all()
    )
    r2 = AttentionInspector(model).inspect(x[0], layer=-1, head=[1, 3])
    assert r2.weights.shape == (1, 2, 9, 9) and r2.layer == 1 and r2.heads == [1, 3]
    with pytest.raises(ValueError, match="layer"):
        AttentionInspector(model).inspect(x, layer=5)
    with pytest.raises(ValueError, match="head"):
        AttentionInspector(model).inspect(x, head=9)


def test_inspector_matches_model_weights_and_restores_mode(model):
    model.train()
    x = torch.randint(2, 37, (1, 6))
    r = AttentionInspector(model).inspect(x, layer=0)
    assert model.training
    model.eval()
    assert torch.equal(r.weights, model(x, output_attentions=True).attentions[0])


def test_normal_inference_does_not_keep_attention_tensors(model):
    out = model(torch.randint(2, 37, (1, 6)), use_cache=True)
    assert out.attentions is None and not hasattr(model.blocks[0].attn, "last_attention")


def test_plot_returns_figure_with_token_labels(model):
    x = torch.randint(2, 37, (1, 5))
    insp = AttentionInspector(model)
    fig = insp.plot(insp.inspect(x, layer=0), tokens=list("abcde"))
    assert len(fig.axes) >= 4
    with pytest.raises(ValueError, match="token labels"):
        insp.plot(insp.inspect(x), tokens=["a"])


def test_compare_architectures_measures_and_estimates_separately():
    cfgs = [tiny_config(num_kv_heads=k, max_seq_len=64) for k in (4, 2, 1)]
    res = compare_architectures(
        cfgs, batch_size=2, sequence_length=16, new_tokens=4, names=["mha", "gqa", "mqa"], repeats=2
    )
    assert [r.attention for r in res] == ["MHA", "GQA", "MQA"]
    assert res[0].num_params > res[1].num_params > res[2].num_params
    assert [r.measured_kv_cache_bytes for r in res] == [r.estimated_kv_cache_bytes for r in res]
    assert res[0].measured_kv_cache_bytes == 4 * res[2].measured_kv_cache_bytes
    assert all(r.forward_ms > 0 and r.generate_ms_per_token > 0 for r in res)
    assert all(r.peak_memory_mb is None for r in res) or torch.cuda.is_available()
    assert "kv (est)" in format_results(res)
    assert estimate_kv_cache_bytes(cfgs[0], 2, 16) == 2 * 2 * 2 * 4 * 16 * 8 * 4


def test_compare_architectures_validates_inputs():
    with pytest.raises(ValueError, match="max_seq_len"):
        compare_architectures([tiny_config(max_seq_len=20)], sequence_length=16, new_tokens=8)
    with pytest.raises(ValueError, match="names"):
        compare_architectures([tiny_config()], names=["a", "b"])
