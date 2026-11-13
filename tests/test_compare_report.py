"""Comparison report: curves, config diff, reproducibility metadata, multi-file CLI."""

import json

import pytest

from oglodon import OglodonConfig
from oglodon.cli import main
from oglodon.data import make_corpus
from oglodon.experiments import ExperimentData, config_diff, run_comparison


@pytest.fixture(scope="module")
def report():
    data = ExperimentData.from_text(make_corpus(200), 32)
    base = OglodonConfig(vocab_size=1, hidden_size=32, num_layers=1, num_heads=4, max_seq_len=64)
    return run_comparison(
        base,
        {"mha": {"num_kv_heads": 4}, "gqa": {"num_kv_heads": 2, "use_rmsnorm": False}},
        data,
        steps=20,
        seeds=(0, 1),
        batch_size=8,
        validate=False,
        logger=None,
    )


def test_curves_have_validation_points_and_end_at_final_loss(report):
    for name, curve in report.curves.items():
        vals = [p for p in curve if "val_loss" in p]
        assert len(vals) == 10 and vals[-1]["step"] == 20
        final = sum(r.val_loss for r in report.runs if r.variant == name) / 2
        assert vals[-1]["val_loss"] == pytest.approx(final, rel=1e-9)
        assert vals[0]["val_loss"] > vals[-1]["val_loss"]  # it learns


def test_config_diff_lists_only_changed_fields(report):
    d = report.config_diff["gqa"]
    assert d["num_kv_heads"] == [4, 2] and d["use_rmsnorm"] == [True, False]
    assert "hidden_size" not in d and "mha" not in report.config_diff
    assert config_diff({"a": {"x": 1}, "b": {"x": 1}}, "a") == {"b": {}}


def test_reproducibility_metadata(report):
    m = report.metadata
    assert m["seeds"] == [0, 1] and m["torch_version"] and m["numpy_version"]
    assert set(m["variant_config_sha256"]) == {"mha", "gqa"}
    assert m["variant_config_sha256"]["mha"] != m["variant_config_sha256"]["gqa"]
    assert m["train_tokens_in_corpus"] > 0 and m["tokenizer"] == "CharTokenizer"
    text = report.to_markdown()
    for heading in ("Configuration differences", "Validation loss during training", "Reproducibility"):
        assert heading in text


def test_throughput_excludes_evaluation(report):
    for r in report.runs:
        assert r.eval_seconds > 0 and r.tokens_per_second == pytest.approx(r.train_tokens / r.train_seconds)


def test_json_roundtrip_and_plot(report, tmp_path):
    p = tmp_path / "r.json"
    report.save(p)
    data = json.loads(p.read_text())
    assert data["config_diff"]["gqa"]["num_kv_heads"] == [4, 2] and "curves" in data
    pytest.importorskip("matplotlib")
    assert report.plot(tmp_path / "r.png").stat().st_size > 1000


def _write(path, kv_heads, steps=12):
    path.write_text(
        json.dumps(
            {
                "model": {"vocab_size": 1, "hidden_size": 32, "num_layers": 1, "num_heads": 4,
                          "num_kv_heads": kv_heads, "max_seq_len": 64},
                "data": {"block_size": 16},
                "train": {"steps": steps, "batch_size": 8, "seeds": [0], "validate": False},
            }
        )
    )  # fmt: skip


def test_cli_compare_several_config_files(tmp_path):
    a, b = tmp_path / "mha.json", tmp_path / "gqa.json"
    _write(a, 4)
    _write(b, 1)
    out = tmp_path / "out.json"
    assert main(["compare", str(a), str(b), "--out", str(out), "--plot"]) == 0
    data = json.loads(out.read_text())
    assert data["config"]["baseline"] == "mha"
    assert data["config_diff"]["gqa"]["num_kv_heads"] == [4, 1]
    assert data["metadata"]["source_config_sha256"] and data["metadata"]["source_configs"]
    assert out.with_suffix(".png").exists()


def test_cli_compare_rejects_mismatched_training_settings(tmp_path, capsys):
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    _write(a, 4)
    _write(b, 2, steps=99)
    assert main(["compare", str(a), str(b)]) == 2
    assert "different 'train:'" in capsys.readouterr().err


def test_cli_plot_requires_out(tmp_path, capsys):
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    _write(a, 4)
    _write(b, 2)
    assert main(["compare", str(a), str(b), "--plot"]) == 2
