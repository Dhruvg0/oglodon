"""Evaluation suite: every number is compared with an independent, deliberately naive computation."""

import json
import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch import nn

from oglodon import Oglodon, OglodonConfig
from oglodon.evaluation import (
    check_generation,
    continuation_logprob,
    evaluate_multiple_choice,
    load_jsonl,
    perplexity,
)
from oglodon.tokenization import CharTokenizer

V = 30


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    cfg = OglodonConfig(vocab_size=V, hidden_size=32, num_layers=2, num_heads=4, max_seq_len=32)
    return Oglodon(cfg).eval()


def naive_nll(model, ids, lo, hi):
    """-log P(ids[t] | ids[lo:t]) summed for t in [lo+1, hi), one prefix at a time (no windows, no batching)."""
    total = 0.0
    for t in range(lo + 1, hi):
        with torch.no_grad():
            lg = model(torch.tensor([ids[lo:t]]))[0, -1]
        total -= F.log_softmax(lg.double(), -1)[ids[t]].item()
    return total


def test_short_sequence_equals_a_single_forward_loss(model):
    ids = torch.randint(0, V, (20,))
    r = perplexity(model, ids)
    want = model(ids[None], labels=ids[None]).loss.item()
    assert r.windows == 1 and r.tokens_scored == 19 and r.loss == pytest.approx(want, rel=1e-5)
    assert r.perplexity == pytest.approx(math.exp(want), rel=1e-5)


def test_non_overlapping_windows_match_naive_prefix_scoring(model):
    ids = torch.randint(0, V, (70,)).tolist()
    r = perplexity(model, ids, block_size=16, stride=16, batch_size=3)
    # windows [0,16) [16,32) [32,48) [48,64) [64,70); the first token of each later window has no context
    want = sum(naive_nll(model, ids, s, min(s + 16, 70)) for s in range(0, 70, 16))
    assert r.tokens_scored == 70 - 5 and r.windows == 5
    assert r.loss * r.tokens_scored == pytest.approx(want, rel=1e-4)


def test_sliding_window_scores_every_token_once_with_full_window_context(model):
    ids = torch.randint(0, V, (50,)).tolist()
    block, stride = 16, 4
    r = perplexity(model, ids, block_size=block, stride=stride, batch_size=4)
    assert r.tokens_scored == 49
    # naive: token t is scored by the first window containing it with >=1 context token
    want, covered = 0.0, 0
    start = 0
    while True:
        end = min(start + block, 50)
        for t in range(max(start + 1, covered + 1), end):
            want -= F.log_softmax(model(torch.tensor([ids[start:t]]))[0, -1].double(), -1)[ids[t]].item()
        covered = end - 1
        if end == 50:
            break
        start += stride
    assert r.loss * 49 == pytest.approx(want, rel=1e-4)
    # more context per token => lower perplexity than the non-overlapping estimate (for any model that uses context)
    r2 = perplexity(model, ids, block_size=block, stride=block)
    assert r.tokens_scored > r2.tokens_scored


def test_perplexity_is_independent_of_batch_size_and_leaves_training_mode(model):
    ids = np.random.default_rng(0).integers(0, V, 90)
    a = perplexity(model, ids, block_size=16, stride=8, batch_size=1)
    b = perplexity(model, ids, block_size=16, stride=8, batch_size=16)
    assert a.loss == pytest.approx(b.loss, rel=1e-5)
    model.train()
    perplexity(model, ids, block_size=16)
    assert model.training
    model.eval()


def test_perplexity_validation_errors(model):
    with pytest.raises(ValueError, match="at least 2"):
        perplexity(model, [1])
    with pytest.raises(ValueError, match="stride"):
        perplexity(model, list(range(20)), block_size=8, stride=9)


def test_uniform_model_has_perplexity_equal_to_vocab_size():
    class Uniform(nn.Module):
        config = type("C", (), {"vocab_size": V, "max_position_embeddings": 64})()

        def __init__(self):
            super().__init__()
            self.p = nn.Parameter(torch.zeros(1))

        def forward(self, input_ids):
            return torch.zeros(*input_ids.shape, V) + self.p

    r = perplexity(Uniform(), list(range(V)) * 2, block_size=20)
    assert r.perplexity == pytest.approx(V, rel=1e-6) and r.bits_per_token == pytest.approx(math.log2(V))


def test_continuation_logprob_matches_manual_chain_rule(model):
    ctx, cont = [3, 5, 7], [11, 2, 9]
    want = 0.0
    seq = ctx + cont
    for t in range(3, 6):
        want += F.log_softmax(model(torch.tensor([seq[:t]]))[0, -1].double(), -1)[seq[t]].item()
    assert continuation_logprob(model, ctx, cont) == pytest.approx(want, rel=1e-5)


class Planted(nn.Module):
    """A fake LM that always puts its mass on (previous token + 1) % V: its preferences are known exactly."""

    config = type("C", (), {"vocab_size": V, "max_position_embeddings": 64})()

    def __init__(self):
        super().__init__()
        self.p = nn.Parameter(torch.zeros(1))

    def forward(self, input_ids):
        logits = torch.full((*input_ids.shape, V), -10.0) + self.p
        nxt = (input_ids + 1) % V
        return logits.scatter(-1, nxt[..., None], 10.0)


class Letters:
    def encode(self, s):
        return [ord(c) - 97 for c in s]  # 'a' -> 0


def test_multiple_choice_with_a_planted_model_has_known_accuracy_and_stderr():
    ex = [
        {"context": "ab", "choices": ["cd", "xx", "zz"], "label": 0},  # b(1)->c(2)->d(3): the chain continues
        {"context": "ef", "choices": ["gh", "aa"], "label": 0},
        {"context": "ab", "choices": ["cd", "ee"], "label": 1},  # wrong label on purpose
        {"context": "mn", "choices": ["zz", "op"], "label": 1},
    ]
    r = evaluate_multiple_choice(Planted(), ex, Letters())
    assert r.n == 4 and r.acc == pytest.approx(0.75) and r.acc_norm == pytest.approx(0.75)
    assert r.stderr == pytest.approx(math.sqrt(0.75 * 0.25 / 4))
    assert [p["pred"] for p in r.predictions] == [0, 0, 0, 1]


def test_multiple_choice_errors_and_jsonl(tmp_path):
    with pytest.raises(ValueError, match="label"):
        evaluate_multiple_choice(Planted(), [{"context": "a", "choices": ["b", "c"], "label": 5}], Letters())
    with pytest.raises(ValueError, match="zero tokens"):
        evaluate_multiple_choice(Planted(), [{"context": "a", "choices": ["", "c"], "label": 1}], Letters())
    f = tmp_path / "d.jsonl"
    f.write_text(json.dumps({"context": "a", "choices": ["b", "c"], "label": 0}) + "\n\n")
    assert len(load_jsonl(f)) == 1
    f.write_text("{oops\n")
    with pytest.raises(ValueError, match=":1:"):
        load_jsonl(f)


def test_multiple_choice_batching_of_choices_is_exact(model):
    tok = CharTokenizer.from_text("abcdefghijklmnopqrstuvwxyz ")
    cfg = OglodonConfig(vocab_size=tok.vocab_size, hidden_size=32, num_layers=1, num_heads=4, max_seq_len=64)
    torch.manual_seed(1)
    m = Oglodon(cfg).eval()
    ex = {"context": "hello wor", "choices": ["ld", "ld is long", "k"], "label": 0}
    r = evaluate_multiple_choice(m, [ex], tok)
    for k, c in enumerate(ex["choices"]):  # one-at-a-time scoring (no padding) must give the same numbers
        solo = continuation_logprob(m, tok.encode(ex["context"]), tok.encode(c))
        assert r.predictions[0]["logprobs"][k] == pytest.approx(solo, rel=1e-5)


def test_generation_report_on_oglodon_model(model):
    rep = check_generation(model, [[2, 3, 4], [5, 6], [7]], max_new_tokens=12)
    assert rep.greedy_deterministic and rep.cache_matches_full is True
    assert rep.prompts == 3 and rep.new_tokens_per_prompt == 12
    assert 0 < rep.distinct_1 <= 1 and 0 < rep.distinct_2 <= 1 and 0 <= rep.repeated_3gram_fraction <= 1


def test_generation_report_flags_a_loop():
    class Loop(Oglodon):
        def forward(self, input_ids, **kw):
            out = super().forward(input_ids, **kw)
            fake = torch.full((*input_ids.shape, self.config.vocab_size), -5.0)
            fake = fake.scatter(-1, (input_ids * 0 + 4)[..., None], 5.0)  # always token 4
            if isinstance(out, torch.Tensor):
                return fake
            out.logits = fake
            return out

    cfg = OglodonConfig(vocab_size=V, hidden_size=16, num_layers=1, num_heads=2, max_seq_len=64)
    rep = check_generation(Loop(cfg), [[1, 2]], max_new_tokens=10)
    assert rep.distinct_1 == pytest.approx(0.1) and rep.repeated_3gram_fraction > 0.8
