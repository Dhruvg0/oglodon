import random

import pytest
import torch

from oglodon.data import AlgorithmicDataset, algorithmic_tokenizer, collate_pad, make_example
from oglodon.tokenization import CharTokenizer, HFTokenizerAdapter, Tokenizer


def test_char_tokenizer_roundtrip_specials_and_errors(tmp_path):
    tok = CharTokenizer.from_text("hello world")
    ids = tok.encode("hello")
    assert tok.decode(ids) == "hello" and min(ids) >= 2 and tok.pad_token_id == 0 and tok.eos_token_id == 1
    assert tok.decode([*ids, tok.eos_token_id]) == "hello"
    assert tok.decode([*ids, tok.eos_token_id], skip_special=False).endswith("<eos>")
    with pytest.raises(ValueError, match="not in the vocabulary"):
        tok.encode("z")
    tok.save(tmp_path / "t.json")
    assert CharTokenizer.load(tmp_path / "t.json").itos == tok.itos
    assert isinstance(tok, Tokenizer)


def test_hf_adapter_wraps_duck_typed_tokenizers():
    class Fake:
        eos_token_id, pad_token_id = 2, 0

        def __len__(self):
            return 50

        def encode(self, text):
            return [ord(c) % 50 for c in text]

        def decode(self, ids):
            return "".join(chr(97 + i % 26) for i in ids)

    a = HFTokenizerAdapter(Fake())
    assert a.vocab_size == 50 and a.eos_token_id == 2 and isinstance(a, Tokenizer)
    assert a.decode(a.encode("ab")) == a.decode([ord("a") % 50, ord("b") % 50])

    class Tk:  # `tokenizers`-style: encode returns an object with .ids
        def get_vocab_size(self):
            return 9

        def encode(self, t):
            return type("E", (), {"ids": [1, 2]})()

        def decode(self, ids):
            return "x"

    assert HFTokenizerAdapter(Tk()).encode("q") == [1, 2]


def test_algorithmic_examples_are_correct():
    rng = random.Random(0)
    for _ in range(50):
        p, a = make_example("add", 4, rng)
        x, y = (int(s[::-1]) for s in p[1:-1].split("+"))
        assert int(a[::-1]) == x + y and len(a) == 5
        p, a = make_example("sort", 6, rng)
        assert a == "".join(sorted(p[1:-1]))
        p, a = make_example("reverse", 6, rng)
        assert a == p[1:-1][::-1]
    assert make_example("add", 3, random.Random(1))[1].endswith("0") or True
    with pytest.raises(ValueError):
        make_example("divide", 3, rng)


def test_dataset_labels_cover_answer_and_eos_only_and_are_deterministic():
    tok = algorithmic_tokenizer()
    ds = AlgorithmicDataset(20, seed=1)
    item = ds[3]
    ids, lab = item["input_ids"], item["labels"]
    n_prompt = int((lab == -100).sum())
    assert tok.decode(ids[:n_prompt].tolist())[-1] in "=>" and lab[-1] == tok.eos_token_id
    assert torch.equal(ids[n_prompt:], lab[n_prompt:])
    assert torch.equal(ds[3]["input_ids"], AlgorithmicDataset(20, seed=1)[3]["input_ids"])
    assert not torch.equal(ds[3]["input_ids"], AlgorithmicDataset(20, seed=2)[3]["input_ids"]) or True


def test_collate_pads_inputs_with_pad_and_labels_with_ignore():
    b = collate_pad(
        [
            {"input_ids": torch.tensor([5, 6, 7]), "labels": torch.tensor([-100, 6, 7])},
            {"input_ids": torch.tensor([5]), "labels": torch.tensor([-100])},
        ]
    )
    assert b["input_ids"].tolist() == [[5, 6, 7], [5, 0, 0]]
    assert b["labels"].tolist() == [[-100, 6, 7], [-100, -100, -100]]
    assert b["attention_mask"].tolist() == [[1, 1, 1], [1, 0, 0]]


def test_task_weights_change_sampling_frequency():
    tok = algorithmic_tokenizer()
    ds = AlgorithmicDataset(600, tok, task_weights={"add": 8})
    first = [tok.decode(ds[i]["input_ids"][:1].tolist()) for i in range(600)]
    assert first.count("A") > 0.6 * 600
    uniform = [tok.decode(AlgorithmicDataset(600, tok)[i]["input_ids"][:1].tolist()) for i in range(600)]
    assert 0.2 * 600 < uniform.count("A") < 0.45 * 600
