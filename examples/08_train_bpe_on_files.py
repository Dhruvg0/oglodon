"""Train a BPE tokenizer and a small Oglodon model on your own text files, end to end.

    python examples/08_train_bpe_on_files.py [folder-or-glob]      (default: this repository's docs and source)

Steps: collect files -> train BPE on the TRAIN part only -> tokenise to a memory-mapped .bin ->
train with the Trainer -> report validation loss/perplexity -> sample text.
The corpus here is tiny, so expect a model that has learned the vocabulary and local structure of the
text, not a language model. Point it at a bigger folder for better results.
"""

import sys
import tempfile
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from oglodon import Oglodon, OglodonConfig, Trainer
from oglodon.data import collect_text_files, read_texts, tokenize_to_bin
from oglodon.generation import generate
from oglodon.tokenization import BPETokenizer

root = Path(__file__).resolve().parents[1]
source = sys.argv[1] if len(sys.argv) > 1 else [root / "docs", root / "README.md", root / "src"]
files = collect_text_files(source, suffixes=(".md", ".py", ".txt"))
print(f"{len(files)} files")

# Fit the tokenizer on the training part of the files only (the validation text must not influence it).
texts = list(read_texts(files))
split_file = max(1, int(len(texts) * 0.9))
tok = BPETokenizer.train(texts[:split_file], vocab_size=768)
print(f"BPE vocabulary: {tok.vocab_size} tokens")

work = Path(tempfile.mkdtemp())
tf = tokenize_to_bin(files, tok, work / "corpus.bin")
block = 64
train, val = tf.dataset(block, "train"), tf.dataset(block, "val")
print(f"{len(tf):,} tokens -> {len(train)} train / {len(val)} validation windows")

cfg = OglodonConfig(
    vocab_size=tok.vocab_size,
    hidden_size=128,
    num_layers=4,
    num_heads=8,
    num_kv_heads=2,
    max_seq_len=block,
    eos_token_id=tok.eos_token_id,
    pad_token_id=tok.pad_token_id,
)
model = Oglodon(cfg)
trainer = Trainer(
    model,
    DataLoader(train, batch_size=16, shuffle=True, generator=torch.Generator().manual_seed(0)),
    eval_dataloader=DataLoader(val, batch_size=16),
    max_steps=300,
    learning_rate=3e-3,
    warmup_steps=20,
    eval_interval=100,
    log_interval=50,
    device="cpu",
)
history = trainer.train()
final = [h for h in history if "val_loss" in h][-1]
print(
    f"validation loss {final['val_loss']:.3f}  perplexity {final['val_ppl']:.1f}  (uniform guess would be {tok.vocab_size})"
)

prompt = torch.tensor([tok.encode("The validator")])
out = generate(
    model, prompt, max_new_tokens=30, temperature=0.8, top_k=20, generator=torch.Generator().manual_seed(0)
)
print(repr(tok.decode(out[0].tolist())))
