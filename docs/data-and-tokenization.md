# Data and tokenization

## Tokenizers

| tokenizer | use | notes |
|---|---|---|
| `BPETokenizer` | default for real text | byte-level BPE you train on your own text; lossless for any string (no unknown token); deterministic; saved as JSON in checkpoints |
| `CharTokenizer` | tiny experiments | one id per character; ids `<pad>=0`, `<eos>=1`, then the characters |
| `HFTokenizerAdapter` | existing Hugging Face tokenizers | wraps `tokenizers`/`transformers`; has no `.save`, so it cannot be stored inside a checkpoint |

Any object with `encode`, `decode` and `vocab_size` works with the model, which only sees integer ids.

```python
from oglodon.tokenization import BPETokenizer

tok = BPETokenizer.train(texts, vocab_size=2048)  # texts: a string or an iterable of strings
ids = tok.encode("hello")
text = tok.decode(ids)  # special ids: tok.pad_token_id, tok.eos_token_id
tok.save("tokenizer.json")
tok = BPETokenizer.load("tokenizer.json")
```

Speed (measured on a 2-core CPU): a 16,000-token vocabulary trains on 4.6 MB in about 5 s; encoding runs at roughly
4-5 MB/s per core. The tokenizer is plain Python, fine for hundreds of megabytes; for much larger corpora use a Rust
tokenizer through `HFTokenizerAdapter`. `tokenize_to_bin(..., workers=N)` encodes files in parallel with byte-identical
output (spawned processes: call it under `if __name__ == "__main__":` in scripts).

The BPE is learned from the **training part of the text only** (`oglodon train` and
`ExperimentData.from_text`), so validation text never influences it.

## Token files

```python
from oglodon.data import collect_text_files, tokenize_to_bin, TokenFile

files = collect_text_files("corpus/")  # file, folder, glob or a list of those
tf = tokenize_to_bin(files, tok, "corpus.bin")  # streams file by file; writes corpus.bin + corpus.json
train, val = tf.dataset(128, "train", 0.05), tf.dataset(128, "val", 0.05)
```

`corpus.bin` is a flat `uint16`/`uint32` array read lazily through `numpy.memmap`, so corpora larger than RAM work. The
`.json` sidecar records the vocabulary size, token count, dtype and each source file with its SHA-256, which is what
makes a run reproducible. An end-of-text token is inserted between files. Train and validation are one cut point in the
stream, so they never overlap.

## Checkpoints store the tokenizer

`save_pretrained(path, tokenizer=tok)` writes `tokenizer.json` for `BPETokenizer` and `CharTokenizer`
(anything with `.save(path)`); `load_tokenizer(path)` restores those two. For a Hugging Face tokenizer, save it yourself
next to the checkpoint.
