import json

import pytest
import torch
from safetensors.torch import load_file

from oglodon import Oglodon
from oglodon.tokenization import CharTokenizer, load_tokenizer

from .conftest import tiny_config


@pytest.mark.parametrize("tie", [True, False])
def test_save_load_gives_identical_outputs(tmp_path, tie):
    torch.manual_seed(0)
    m = Oglodon(tiny_config(tie_embeddings=tie)).eval()
    x = torch.randint(2, 37, (2, 10))
    m.save_pretrained(tmp_path / "ckpt")
    m2 = Oglodon.from_pretrained(tmp_path / "ckpt")
    assert m2.config == m.config and not m2.training
    assert torch.equal(m(x), m2(x))
    if tie:
        assert m2.lm_head.weight is m2.embed_tokens.weight
    assert torch.equal(m.generate(x[:, :4], max_new_tokens=6), m2.generate(x[:, :4], max_new_tokens=6))


def test_files_are_safe_formats_and_rope_tables_not_stored(tmp_path):
    m = Oglodon(tiny_config())
    m.save_pretrained(tmp_path)
    assert {p.name for p in tmp_path.iterdir()} == {"config.json", "model.safetensors"}
    assert json.loads((tmp_path / "config.json").read_text())["vocab_size"] == 37
    keys = load_file(str(tmp_path / "model.safetensors")).keys()
    assert not any("cos" in k or "sin" in k for k in keys) and "lm_head.weight" not in keys


def test_tokenizer_is_saved_and_restored(tmp_path):
    tok = CharTokenizer("abc")
    Oglodon(tiny_config()).save_pretrained(tmp_path, tokenizer=tok)
    loaded = load_tokenizer(tmp_path)
    assert loaded.encode("cab") == tok.encode("cab")
    assert load_tokenizer(tmp_path / "missing") is None


def test_tokenizer_without_save_method_is_rejected(tmp_path):
    with pytest.raises(TypeError, match="save"):
        Oglodon(tiny_config()).save_pretrained(tmp_path, tokenizer=object())


def test_corrupt_or_mismatched_checkpoint_is_detected(tmp_path):
    Oglodon(tiny_config()).save_pretrained(tmp_path)
    cfg = json.loads((tmp_path / "config.json").read_text())
    cfg["num_layers"] = 3  # weights for 2 layers only
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    with pytest.raises(RuntimeError, match="mismatch"):
        Oglodon.from_pretrained(tmp_path)


def test_load_to_device(tmp_path, device):
    Oglodon(tiny_config()).save_pretrained(tmp_path)
    assert Oglodon.from_pretrained(tmp_path, device=device).device.type == device
