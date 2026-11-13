import random

import numpy as np
import pytest
import torch

from oglodon import Oglodon
from oglodon.data import TokenFile, collect_text_files, read_texts, tokenize_to_bin
from oglodon.experiments import ExperimentData
from oglodon.tokenization import BPETokenizer, load_tokenizer

TEXT = (
    "the quick brown fox jumps over the lazy dog. " * 40 + "snake_case_name = 42 # tab\there\n\n  indented\n"
) * 5


@pytest.fixture(scope="module")
def tok():
    return BPETokenizer.train(TEXT, 300)


def test_training_learns_merges_and_is_deterministic(tok):
    assert 256 + 2 < tok.vocab_size <= 300
    assert BPETokenizer.train(TEXT, 300).merges == tok.merges
    assert len(tok.encode(TEXT)) < len(TEXT.encode()) / 2  # real compression
    assert len(tok.encode(" quick")) <= 2  # a frequent word is (nearly) a single token


def test_lossless_on_training_text_underscores_whitespace_and_unicode(tok):
    for s in [TEXT, "snake_case__x", "  \n\t \n", "héllo wörld 你好 🙂", "", "<<>>", "a" * 500]:
        assert tok.decode(tok.encode(s), skip_special=False) == s


def test_lossless_fuzz(tok):
    rng = random.Random(1)
    pools = [(0, 0x2FF), (0x400, 0x4FF), (0x4E00, 0x4E80), (0x1F600, 0x1F640), (32, 32), (10, 10), (95, 95)]
    for _ in range(300):
        s = "".join(chr(rng.randint(*rng.choice(pools))) for _ in range(rng.randint(0, 50)))
        assert tok.decode(tok.encode(s), skip_special=False) == s


def test_special_tokens(tok):
    ids = tok.encode("hi <eos> there")
    assert tok.eos_token_id in ids and tok.pad_token_id == 0
    assert tok.decode(ids) == "hi  there"  # specials skipped by default
    assert tok.decode(ids, skip_special=False) == "hi <eos> there"
    assert tok.encode("x", add_eos=True)[-1] == tok.eos_token_id
    with pytest.raises(ValueError):
        tok.decode([tok.vocab_size])


def test_unseen_text_never_fails_and_vocab_bounds(tok):
    ids = tok.encode("completely different words 12345 ñ")
    assert max(ids) < tok.vocab_size and min(ids) >= 0


def test_save_load_and_checkpoint_roundtrip(tok, tmp_path):
    tok.save(tmp_path / "t.json")
    assert BPETokenizer.load(tmp_path / "t.json").encode(TEXT) == tok.encode(TEXT)
    from oglodon import OglodonConfig

    model = Oglodon(
        OglodonConfig(vocab_size=tok.vocab_size, hidden_size=16, num_layers=1, num_heads=2, max_seq_len=16)
    )
    model.save_pretrained(tmp_path / "ck", tok)
    back = load_tokenizer(tmp_path / "ck")
    assert isinstance(back, BPETokenizer) and back.encode("the fox") == tok.encode("the fox")


def test_bad_arguments():
    with pytest.raises(ValueError):
        BPETokenizer.train(TEXT, 100)
    with pytest.raises(ValueError):
        BPETokenizer(special_tokens=["a", "a"])


def test_min_frequency_stops_early():
    t = BPETokenizer.train("abcdefg", 1000, min_frequency=2)
    assert t.vocab_size == 2 + 256  # no pair repeats, so nothing is merged


# ------------------------------------------------------------------------------------ data pipeline
@pytest.fixture
def corpus(tmp_path):
    d = tmp_path / "corp"
    (d / "sub").mkdir(parents=True)
    (d / "a.txt").write_text("alpha beta gamma. " * 80)
    (d / "sub" / "b.md").write_text("delta epsilon zeta. " * 80)
    (d / "ignore.bin").write_bytes(b"\x00\x01")
    return d


def test_collect_text_files_dir_glob_file_and_missing(corpus):
    files = collect_text_files(corpus)
    assert [f.name for f in files] == ["a.txt", "b.md"]
    assert [f.name for f in collect_text_files(str(corpus / "*.txt"))] == ["a.txt"]
    assert collect_text_files(corpus / "a.txt") == [corpus / "a.txt"]
    with pytest.raises(FileNotFoundError):
        collect_text_files(corpus / "nope")


def test_tokenize_to_bin_roundtrip_and_metadata(corpus, tmp_path):
    files = collect_text_files(corpus)
    tok = BPETokenizer.train(read_texts(files), 300)
    tf = tokenize_to_bin(files, tok, tmp_path / "c.bin")
    arr = np.fromfile(tmp_path / "c.bin", dtype=np.uint16)
    assert len(arr) == len(tf) == tf.meta["n_tokens"] == sum(f["tokens"] for f in tf.meta["files"])
    text = tok.decode(arr.tolist(), skip_special=True)
    assert "alpha beta gamma." in text and "delta epsilon zeta." in text
    assert (arr == tok.eos_token_id).sum() == 1  # one separator between two files
    again = TokenFile.open(tmp_path / "c.bin")
    assert again.vocab_size == tok.vocab_size and again.fingerprint() == tf.fingerprint()
    assert all(len(f["sha256"]) == 64 for f in tf.meta["files"])


def test_train_val_windows_never_overlap_and_memmap_is_lazy(corpus, tmp_path):
    files = collect_text_files(corpus)
    tok = BPETokenizer.train(read_texts(files), 300)
    tf = tokenize_to_bin(files, tok, tmp_path / "c.bin")
    tr, va = tf.dataset(16, "train"), tf.dataset(16, "val")
    cut = tf.split_point(0.1)
    assert tr.hi == cut == va.lo and len(tr) * 16 <= cut
    item = tr[0]
    assert item["input_ids"].dtype == torch.long and torch.equal(item["input_ids"], item["labels"])
    assert isinstance(tr.tokens, np.memmap)
    with pytest.raises(IndexError):
        tr[len(tr)]
    with pytest.raises(ValueError):
        tf.dataset(10_000, "val")
    last_train = tr[len(tr) - 1]["input_ids"]
    assert torch.equal(
        last_train,
        torch.from_numpy(
            np.fromfile(tmp_path / "c.bin", dtype=np.uint16)[(len(tr) - 1) * 16 : len(tr) * 16].astype(
                np.int64
            )
        ),
    )


def test_large_vocab_uses_uint32(tmp_path):
    class Fake:
        vocab_size = 70000
        eos_token_id = None

        def encode(self, t):
            return [69999, 1, 2]

    f = tmp_path / "x.txt"
    f.write_text("hi")
    tf = tokenize_to_bin([f], Fake(), tmp_path / "x.bin")
    assert tf.meta["dtype"] == "uint32" and int(tf.array()[0]) == 69999


def test_out_of_range_ids_are_rejected(tmp_path):
    class Bad:
        vocab_size = 10
        eos_token_id = None

        def encode(self, t):
            return [3, 11]

    f = tmp_path / "x.txt"
    f.write_text("hi")
    with pytest.raises(ValueError, match="outside"):
        tokenize_to_bin([f], Bad(), tmp_path / "x.bin")


def test_experiment_data_bpe_trains_tokenizer_on_train_split_only():
    train_part = "alpha beta gamma delta. " * 200
    val_part = "zzyzx quux zzyzx quux. " * 40
    d = ExperimentData.from_text(
        train_part + val_part,
        block_size=16,
        val_fraction=len(val_part) / (len(train_part) + len(val_part)),
        bpe_vocab_size=400,
    )
    assert d.vocab_size == d.tokenizer.vocab_size
    # 'zzyzx' never appeared in the training text, so it must not have been merged into a single token
    assert len(d.tokenizer.encode(" zzyzx")) > 2
    with pytest.raises(ValueError):
        ExperimentData.from_text(TEXT, tokenizer=object(), bpe_vocab_size=300)


def test_parallel_tokenisation_is_byte_identical(corpus, tmp_path):
    files = collect_text_files(corpus)
    tok = BPETokenizer.train(read_texts(files), 300)
    a = tokenize_to_bin(files, tok, tmp_path / "a.bin", workers=1)
    b = tokenize_to_bin(files, tok, tmp_path / "b.bin", workers=2)
    assert (tmp_path / "a.bin").read_bytes() == (tmp_path / "b.bin").read_bytes()
    assert a.meta == {**b.meta, "files": a.meta["files"]} and a.fingerprint() == b.fingerprint()
    with pytest.raises(ValueError):
        tokenize_to_bin(files, tok, tmp_path / "c.bin", workers=0)


def test_heap_training_matches_reference_slow_implementation():
    """A straightforward O(P)-per-merge implementation must pick exactly the same merges."""
    from collections import Counter

    from oglodon.tokenization.bpe import _PRETOKEN

    text = ("low lower lowest newer wider widest. " * 30) + ("snake_case_name = 42\n" * 20)
    base = 2
    words = Counter(tuple(base + b for b in p.encode()) for p in _PRETOKEN.findall(text))
    seqs = [list(w) for w in words]
    freq = list(words.values())
    slow: list[tuple[int, int]] = []
    while len(slow) < 60:
        counts: Counter = Counter()
        for w, f in zip(seqs, freq, strict=True):
            for pr in zip(w, w[1:], strict=False):
                counts[pr] += f
        if not counts:
            break
        best, n = max(counts.items(), key=lambda kv: (kv[1], -kv[0][0], -kv[0][1]))
        if n < 2:
            break
        slow.append(best)
        new = base + 256 + len(slow) - 1
        for idx, w in enumerate(seqs):
            out, i = [], 0
            while i < len(w):
                if i < len(w) - 1 and (w[i], w[i + 1]) == best:
                    out.append(new)
                    i += 2
                else:
                    out.append(w[i])
                    i += 1
            seqs[idx] = out
    fast = BPETokenizer.train(text, base + 256 + 60).merges
    assert fast == slow
