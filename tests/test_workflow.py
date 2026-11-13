"""`oglodon train` / `oglodon generate` and the library functions behind them."""

import json
import os
import shutil
import subprocess
import sys

import pytest
import torch

from oglodon import Oglodon
from oglodon.cli import main
from oglodon.data import make_corpus
from oglodon.tokenization import load_tokenizer
from oglodon.workflow import TrainSettings, _split_for_tokenizer, complete, train_from_files

SMALL = ["--steps", "24", "--hidden-size", "32", "--layers", "1", "--heads", "4", "--block-size", "32",
         "--vocab-size", "290", "--batch-size", "8", "--device", "cpu"]  # fmt: skip


@pytest.fixture()
def corpus(tmp_path):
    d = tmp_path / "data"
    d.mkdir()
    (d / "a.txt").write_text(make_corpus(300))
    (d / "b.md").write_text(make_corpus(200))
    return d


def test_train_then_generate_and_eval_end_to_end(corpus, tmp_path, capsys):
    out = tmp_path / "run"
    assert main(["train", str(corpus), "--out", str(out), *SMALL]) == 0
    for name in ("model.safetensors", "config.json", "tokenizer.json", "trainer_state.pt", "train_settings.json",
                 "training_report.json", "data/corpus.bin", "data/corpus.json"):  # fmt: skip
        assert (out / name).exists(), name
    rep = json.loads((out / "training_report.json").read_text())
    assert rep["result"]["steps"] == 24 and rep["parameters"] == Oglodon.from_pretrained(out).num_parameters()
    assert (
        rep["result"]["val_perplexity"] < rep["result"]["uniform_baseline_perplexity"]
    )  # it learned something
    assert rep["data"]["files"] and rep["data"]["fingerprint"] and rep["torch_version"]
    assert rep["config"]["num_kv_heads"] == 1  # default heads // 4
    capsys.readouterr()

    args = ["generate", str(out), "--prompt", "the ", "--max-new-tokens", "20", "--seed", "3"]
    assert main(args) == 0
    first = capsys.readouterr().out
    assert main(args) == 0 and capsys.readouterr().out == first  # seeded -> reproducible
    assert first.startswith("the ") and len(first) > 6
    assert main([*args, "--only-new", "--no-stream"]) == 0
    assert capsys.readouterr().out.strip() == first[len("the ") :].strip()
    assert (
        main(["generate", str(out), "--prompt", "the ", "--temperature", "0", "--max-new-tokens", "10"]) == 0
    )

    text = corpus / "a.txt"
    assert (
        main(["eval", str(out), "--text", str(text), "--block-size", "32"]) == 0
    )  # checkpoint is eval-ready


def test_train_refuses_to_clobber_and_overwrite_allows_it(corpus, tmp_path, capsys):
    out = tmp_path / "run"
    assert main(["train", str(corpus), "--out", str(out), *SMALL]) == 0
    before = (out / "model.safetensors").read_bytes()
    assert main(["train", str(corpus), "--out", str(out), *SMALL, "--seed", "5"]) == 2
    assert "already contains a model" in capsys.readouterr().err
    assert (out / "model.safetensors").read_bytes() == before  # untouched
    assert main(["train", str(corpus), "--out", str(out), *SMALL, "--seed", "5", "--overwrite"]) == 0
    assert (out / "model.safetensors").read_bytes() != before


class _Crash(Exception):
    pass


def test_resume_after_a_crash_reproduces_the_uninterrupted_run_exactly(corpus, tmp_path):
    s = TrainSettings(vocab_size=290, hidden_size=32, num_layers=1, num_heads=4, block_size=32, steps=20,
                      batch_size=4, save_interval=10)  # fmt: skip
    ref = tmp_path / "ref"
    train_from_files(corpus, ref, s, device="cpu", logger=None)

    def crash_at_12(msg):
        if msg.startswith("step") and msg.split()[1] == "12":
            raise _Crash

    broken = tmp_path / "broken"
    with pytest.raises(_Crash):
        train_from_files(corpus, broken, s, device="cpu", logger=crash_at_12)
    assert (broken / "trainer_state.pt").exists() and not (broken / "training_report.json").exists()
    res = train_from_files(corpus, broken, device="cpu", resume=True, logger=None)
    assert res.resumed_from_step == 10 and res.steps == 20
    a, b = Oglodon.from_pretrained(ref), Oglodon.from_pretrained(broken)
    for (n, p), (_, q) in zip(a.state_dict().items(), b.state_dict().items(), strict=True):
        assert torch.allclose(p, q, atol=1e-6, rtol=1e-5), n


def test_resume_without_anything_to_resume_is_an_error(corpus, tmp_path, capsys):
    assert main(["train", str(corpus), "--out", str(tmp_path / "nope"), "--resume", *SMALL]) == 2
    assert "nothing to resume" in capsys.readouterr().err


def test_errors_are_clean_messages_not_tracebacks(corpus, tmp_path, capsys):
    assert main(["train", str(tmp_path / "missing"), "--out", str(tmp_path / "o1"), *SMALL]) == 2
    assert "no text files" in capsys.readouterr().err
    tiny = tmp_path / "tiny.txt"
    tiny.write_text("hello world " * 5)
    assert (
        main(
            [
                "train",
                str(tiny),
                "--out",
                str(tmp_path / "o2"),
                "--tokenizer",
                "char",
                *SMALL[:-2],
                "--device",
                "cpu",
            ]
        )
        == 2
    )
    assert "too small" in capsys.readouterr().err
    assert (
        main(["train", str(corpus), "--out", str(tmp_path / "o3"), *SMALL, "--heads", "5"]) == 2
    )  # 32 % 5 != 0
    assert "error" in capsys.readouterr().err
    cfg = tmp_path / "bad.json"
    cfg.write_text(json.dumps({"train": {"stepz": 3}}))
    assert main(["train", str(corpus), "--out", str(tmp_path / "o4"), "--config", str(cfg)]) == 2
    assert "unknown training setting" in capsys.readouterr().err
    assert main(["generate", str(tmp_path / "nothing")]) == 2


def test_config_file_is_overridden_by_flags(corpus, tmp_path):
    cfg = tmp_path / "c.json"
    cfg.write_text(json.dumps({"model": {"hidden_size": 32, "num_layers": 1, "num_heads": 4, "block_size": 32},
                               "data": {"vocab_size": 290}, "train": {"steps": 8, "batch_size": 4}}))  # fmt: skip
    out = tmp_path / "run"
    assert (
        main(
            ["train", str(corpus), "--out", str(out), "--config", str(cfg), "--steps", "6", "--device", "cpu"]
        )
        == 0
    )
    saved = json.loads((out / "train_settings.json").read_text())
    assert (
        saved["steps"] == 6 and saved["batch_size"] == 4 and saved["hidden_size"] == 32
    )  # flag wins, file fills in


def test_char_tokenizer_and_extra_model_options(corpus, tmp_path):
    cfg = tmp_path / "c.json"
    cfg.write_text(
        json.dumps({"train": {"model": {"use_rope": False, "use_alibi": True, "ffn_activation": "relu2"}}})
    )
    out = tmp_path / "run"
    assert (
        main(["train", str(corpus), "--out", str(out), "--config", str(cfg), "--tokenizer", "char", *SMALL])
        == 0
    )
    m = Oglodon.from_pretrained(out)
    assert (
        m.config.use_alibi
        and m.config.ffn_activation == "relu2"
        and type(load_tokenizer(out)).__name__ == "CharTokenizer"
    )


def test_tokenizer_never_sees_the_validation_text():
    texts = ["a" * 100, "b" * 100, "c" * 100]
    seen = "".join(_split_for_tokenizer(texts, 0.1, 10**9))
    assert len(seen) == 270 and "c" * 71 not in seen and seen.startswith("a" * 100)
    assert len("".join(_split_for_tokenizer(texts, 0.1, 50))) == 50  # the sample cap applies


def test_complete_stops_on_text_and_validates(corpus, tmp_path):
    out = tmp_path / "run"
    assert main(["train", str(corpus), "--out", str(out), *SMALL]) == 0
    m, tok = Oglodon.from_pretrained(out), load_tokenizer(out)
    full = "".join(complete(m, tok, "the ", max_new_tokens=25, temperature=0, top_k=None))
    stop = full[8:11]
    cut = "".join(complete(m, tok, "the ", max_new_tokens=25, temperature=0, top_k=None, stop=[stop]))
    assert full.startswith(cut) and len(cut) <= len(full)
    assert (
        "".join(complete(m, tok, "", max_new_tokens=5, temperature=0, top_k=None)) is not None
    )  # empty prompt
    long_prompt = "the " * 200  # longer than the context: the tail is kept instead of failing
    assert list(complete(m, tok, long_prompt, max_new_tokens=5, temperature=0, top_k=None)) is not None
    with pytest.raises(ValueError, match="max_new_tokens"):
        list(complete(m, tok, "x", max_new_tokens=10_000))


@pytest.mark.slow
@pytest.mark.skipif(shutil.which("torchrun") is None, reason="torchrun not on PATH")
def test_train_under_torchrun_with_two_processes(corpus, tmp_path):
    out = tmp_path / "ddp"
    env = {**os.environ, "OMP_NUM_THREADS": "1", "CUDA_VISIBLE_DEVICES": ""}
    cmd = [sys.executable, "-m", "torch.distributed.run", "--nproc_per_node=2", "-m", "oglodon", "train", str(corpus),
           "--out", str(out), *SMALL]  # fmt: skip
    r = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=300)
    assert r.returncode == 0, r.stderr[-2000:]
    rep = json.loads((out / "training_report.json").read_text())
    assert (
        rep["processes"] == 2
        and rep["result"]["val_perplexity"] < rep["result"]["uniform_baseline_perplexity"]
    )
    assert "2 processes" in r.stdout and r.stdout.count("checkpoint written") == 1  # only rank 0 talks


def test_training_releases_the_memory_mapped_corpus(corpus, tmp_path):
    """On Windows a still-mapped corpus.bin cannot be overwritten, so no mapping may outlive the call."""
    import gc

    import numpy as np

    out = tmp_path / "run"
    train_from_files(str(corpus), out, TrainSettings(steps=4, hidden_size=32, num_layers=1, num_heads=4,
                     block_size=32, vocab_size=290, batch_size=8), device="cpu", logger=None)  # fmt: skip
    gc.disable()  # a collection must not be what releases it
    try:
        alive = [o for o in gc.get_objects() if type(o) is np.memmap and "corpus.bin" in str(o.filename)]
    finally:
        gc.enable()
    assert alive == []
