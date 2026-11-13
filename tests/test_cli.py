import json

import pytest
import torch

from oglodon.cli import load_config_file, main

CFG = {
    "model": {
        "vocab_size": 37,
        "hidden_size": 32,
        "num_layers": 2,
        "num_heads": 4,
        "num_kv_heads": 2,
        "max_seq_len": 48,
        "eos_token_id": 1,
        "pad_token_id": 0,
    },
    "validation": {"categories": ["attention", "model"]},
}


@pytest.fixture
def cfg_json(tmp_path):
    p = tmp_path / "m.json"
    p.write_text(json.dumps(CFG))
    return p


def test_validate_json_config_writes_report(cfg_json, tmp_path, capsys):
    out = tmp_path / "r.json"
    assert main(["validate", str(cfg_json), "--out", str(out)]) == 0
    d = json.loads(out.read_text())
    assert d["passed"] and d["counts"]["fail"] == 0 and d["metadata"]["device"] == "cpu"
    assert "summary" in capsys.readouterr().out


def test_validate_yaml_config(tmp_path):
    pytest.importorskip("yaml")
    p = tmp_path / "m.yaml"
    p.write_text(
        "model:\n  vocab_size: 37\n  hidden_size: 32\n  num_layers: 1\n  num_heads: 4\n  max_seq_len: 32\n"
    )
    assert main(["validate", str(p), "--categories", "model"]) == 0
    assert load_config_file(p)["model"]["vocab_size"] == 37


def test_shipped_configs_load():
    pytest.importorskip("yaml")
    from pathlib import Path

    for f in Path(__file__).parents[1].glob("configs/*.yaml"):
        assert "model" in load_config_file(f)


def test_planted_bug_gives_exit_code_1(cfg_json, monkeypatch):
    import oglodon.layers.attention as a

    orig = a.build_attention_bias
    monkeypatch.setattr(
        a, "build_attention_bias", lambda q, k, d, t, pad=None, window=None: orig(q, k, d, t, None, window)
    )
    assert main(["validate", str(cfg_json), "--categories", "attention"]) == 1


def test_strict_fails_on_incomplete_report(cfg_json):
    # the default run has a skipped tokenizer check and unsupported fp16 only with 'training'/'model' present
    assert main(["validate", str(cfg_json)]) == 0
    assert main(["validate", str(cfg_json), "--strict"]) == 1


def test_usage_errors_exit_2(cfg_json, tmp_path, capsys):
    if not torch.cuda.is_available():
        assert main(["validate", str(cfg_json), "--device", "cuda"]) == 2
        assert "CUDA" in capsys.readouterr().err
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"model": {"vocab_size": 10, "hidden_size": 30, "num_heads": 4}}))
    assert main(["validate", str(bad)]) == 2
    assert "invalid model config" in capsys.readouterr().err


def test_missing_file_and_no_command(capsys):
    assert main(["validate", "/no/such/file.json"]) == 2
    assert "cannot read config" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        main([])


def test_compare_command(tmp_path):
    cfg = {
        "model": {"vocab_size": 1, "hidden_size": 32, "num_layers": 1, "num_heads": 4, "max_seq_len": 32},
        "variants": {"mha": {"num_kv_heads": 4}, "gqa": {"num_kv_heads": 1}},
        "baseline": "mha",
        "data": {"block_size": 16, "sentences": 150},
        "train": {"steps": 4, "batch_size": 4, "seeds": [0, 1], "validate": False},
    }
    p = tmp_path / "c.json"
    p.write_text(json.dumps(cfg))
    out = tmp_path / "out.json"
    assert main(["compare", str(p), "--device", "cpu", "--out", str(out)]) == 0
    assert json.loads(out.read_text())["paired_vs_baseline"]["gqa"]["n"] == 2
    assert out.with_suffix(".md").exists()
    cfg.pop("variants")
    p.write_text(json.dumps(cfg))
    assert main(["compare", str(p)]) == 2


# ------------------------------------------------------------------------------------ oglodon verify
def test_verify_oglodon_config_and_report(cfg_json, tmp_path, capsys):
    out = tmp_path / "v.json"
    assert main(["verify", "--config", str(cfg_json), "--out", str(out)]) in (
        0,
    )  # config has a 'model:' section
    d = json.loads(out.read_text())
    assert d["metadata"]["kind"] == "verify_causal_lm" and d["counts"]["fail"] == 0
    assert "source: config:" in capsys.readouterr().out


def test_verify_factory_hf_config_and_usage_errors(tmp_path, capsys, monkeypatch):
    factory = tmp_path / "my_lm_factory.py"
    factory.write_text(
        "from tests.tiny_lm import TinyLM\n\ndef build():\n    return TinyLM()\n\ndef build_bad():\n    return TinyLM(bug='leak')\n"
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.syspath_prepend(str(__import__("pathlib").Path(__file__).parents[1]))
    assert main(["verify", "--factory", "my_lm_factory:build", "--categories", "causality"]) == 0
    assert main(["verify", "--factory", "my_lm_factory:build_bad", "--categories", "causality"]) == 1
    capsys.readouterr()
    assert main(["verify", "--factory", "no_such_module:f"]) == 2
    assert main(["verify", "--factory", "nocolon"]) == 2
    assert main(["verify"]) == 2  # a source is required
    assert main(["verify", "--hf", "x", "--config", "y"]) == 2  # exactly one source
    pytest.importorskip("transformers")
    hf = tmp_path / "hf.json"
    hf.write_text(
        json.dumps(
            {
                "model_type": "gpt2",
                "vocab_size": 50,
                "n_embd": 32,
                "n_layer": 1,
                "n_head": 4,
                "n_positions": 32,
            }
        )
    )
    assert main(["verify", "--hf-config", str(hf), "--categories", "structure", "causality"]) == 0


def test_verify_strict(cfg_json):
    # a Oglodon model with the default categories is complete (nothing skipped), so --strict passes
    assert main(["verify", "--config", str(cfg_json), "--strict"]) == 0


def test_compare_with_bpe_on_files(tmp_path):
    corpus = tmp_path / "corp"
    corpus.mkdir()
    (corpus / "a.txt").write_text("the cat sat on the mat. the dog ate the bone. " * 120)
    cfg = {
        "model": {"vocab_size": 1, "hidden_size": 32, "num_layers": 1, "num_heads": 4, "max_seq_len": 32},
        "variants": {"mha": {"num_kv_heads": 4}, "gqa": {"num_kv_heads": 1}},
        "data": {"path": str(corpus), "tokenizer": "bpe", "vocab_size": 300, "block_size": 16},
        "train": {"steps": 3, "batch_size": 4, "seeds": [0, 1], "validate": False},
    }
    p = tmp_path / "c.json"
    p.write_text(json.dumps(cfg))
    out = tmp_path / "o.json"
    assert main(["compare", str(p), "--device", "cpu", "--out", str(out)]) == 0
    assert json.loads(out.read_text())["summary"]["mha"]["params"] > 0
    cfg["data"]["path"] = str(tmp_path / "missing")
    p.write_text(json.dumps(cfg))
    assert main(["compare", str(p)]) == 2
    cfg["data"] = {"tokenizer": "wordpiece"}
    p.write_text(json.dumps(cfg))
    assert main(["compare", str(p)]) == 2


def test_eval_command_end_to_end(tmp_path, capsys):
    import json as _json

    import torch

    from oglodon import Oglodon, OglodonConfig
    from oglodon.tokenization import CharTokenizer

    text = "the cat sat on the mat. " * 40
    tok = CharTokenizer.from_text(text)
    torch.manual_seed(0)
    model = Oglodon(
        OglodonConfig(vocab_size=tok.vocab_size, hidden_size=32, num_layers=1, num_heads=4, max_seq_len=64)
    )
    ckpt = tmp_path / "ckpt"
    model.save_pretrained(ckpt, tokenizer=tok)
    data = tmp_path / "data.txt"
    data.write_text(text)
    mc = tmp_path / "mc.jsonl"
    mc.write_text(_json.dumps({"context": "the cat", "choices": [" sat", " mat"], "label": 0}) + "\n")
    out = tmp_path / "r.json"
    code = main(["eval", str(ckpt), "--text", str(data), "--block-size", "32", "--stride", "16", "--choices", str(mc),
                 "--prompt", "the", "a cat", "--max-new-tokens", "8", "--out", str(out)])  # fmt: skip
    assert code == 0
    rep = _json.loads(out.read_text())
    assert rep["perplexity"]["tokens_scored"] == len(tok.encode(text)) - 1
    assert rep["perplexity"]["perplexity"] == pytest.approx(
        model(torch.tensor([tok.encode(text)[:32]]), labels=torch.tensor([tok.encode(text)[:32]]))
        .loss.exp()
        .item(),
        rel=0.5,
    )  # same order as a plain forward on one window (an untrained model is near uniform: ~vocab size)
    assert rep["multiple_choice"]["n"] == 1 and rep["generation"]["greedy_deterministic"] is True
    assert "perplexity" in capsys.readouterr().out


def test_eval_command_usage_errors(tmp_path, capsys):
    assert main(["eval", str(tmp_path)]) == 2  # nothing to evaluate
    assert main(["eval", str(tmp_path), "--text", str(tmp_path / "x.txt")]) == 2  # not a checkpoint
    assert "error" in capsys.readouterr().err
