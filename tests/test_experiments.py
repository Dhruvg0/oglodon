import json

import pytest

from oglodon import OglodonConfig
from oglodon.data import make_corpus
from oglodon.experiments import ExperimentData, run_comparison
from oglodon.experiments.compare import match_intermediate_size
from oglodon.models.gpt import Oglodon

from .conftest import tiny_config


@pytest.fixture(scope="module")
def data():
    return ExperimentData.from_text(make_corpus(200), block_size=16)


def base_cfg(data, **kw):
    return OglodonConfig(
        vocab_size=data.vocab_size, hidden_size=32, num_layers=1, num_heads=4, max_seq_len=32, **kw
    )


def n_params(cfg):
    return sum(p.numel() for p in Oglodon(cfg).parameters())


def test_match_intermediate_size_gets_close_and_keeps_everything_else():
    target = n_params(tiny_config(num_kv_heads=4))
    gqa = tiny_config(num_kv_heads=1)
    matched = match_intermediate_size(gqa, target)
    assert abs(n_params(matched) - target) < abs(n_params(gqa) - target)
    assert matched.intermediate_size > gqa.intermediate_size  # GQA freed parameters, FFN absorbs them
    assert {**matched.to_dict(), "intermediate_size": 0} == {**gqa.to_dict(), "intermediate_size": 0}


def test_data_split_is_disjoint_and_stable(data):
    assert len(data.train_ids) > len(data.val_ids) > 0
    assert data.fingerprint() == ExperimentData.from_text(make_corpus(200), block_size=16).fingerprint()
    with pytest.raises(ValueError):
        ExperimentData.from_text("abc", block_size=16)


@pytest.fixture(scope="module")
def report(data):
    return run_comparison(
        base_cfg(data),
        {"mha": {"num_kv_heads": 4}, "gqa": {"num_kv_heads": 1}},
        data,
        steps=6,
        batch_size=4,
        seeds=(0, 1),
        latency_repeats=1,
        generate_tokens=2,
        validation_categories=("attention",),
        device="cpu",
        logger=None,
    )


def test_report_has_all_requested_metrics(report):
    assert len(report.runs) == 4 and {r.variant for r in report.runs} == {"mha", "gqa"}
    for s in report.summary.values():
        for k in (
            "params",
            "val_loss_mean",
            "val_ppl_mean",
            "tokens_per_second_mean",
            "forward_ms_mean",
            "generate_ms_per_token_mean",
            "kv_cache_bytes",
            "peak_memory_mb",
        ):
            assert k in s
    assert report.summary["gqa"]["kv_cache_bytes"] * 4 == report.summary["mha"]["kv_cache_bytes"]
    assert report.summary["mha"]["peak_memory_mb"] is None  # CPU: not measured, not invented
    assert all(v["counts"]["fail"] == 0 for v in report.validation.values())


def test_same_data_tokens_and_param_matching(report):
    assert len({r.train_tokens for r in report.runs}) == 1
    p = {n: s["params"] for n, s in report.summary.items()}
    assert abs(p["gqa"] - p["mha"]) / p["mha"] < 0.05
    assert report.variants["gqa"]["intermediate_size"] > report.variants["mha"]["intermediate_size"]


def test_paired_stats_and_fairness_notes(report):
    pr = report.paired["gqa"]
    assert pr["n"] == 2 and "mean_diff" in pr and "verdict" in pr
    text = " ".join(report.fairness_notes).lower()
    assert "cpu" in text and "seed" in text


def test_json_and_markdown_outputs(report, tmp_path):
    report.save(tmp_path / "c.json")
    d = json.loads((tmp_path / "c.json").read_text())
    assert {"metadata", "summary", "paired_vs_baseline", "fairness_notes", "runs"} <= set(d)
    md = report.to_markdown()
    assert "Fairness notes" in md and "gqa" in md


def test_bad_baseline_is_rejected(data):
    with pytest.raises(ValueError):
        run_comparison(base_cfg(data), {"a": {}}, data, baseline="zzz", steps=1, seeds=(0,), logger=None)
