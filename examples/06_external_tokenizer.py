"""Plug an external tokenizer into Oglodon.

Oglodon only needs ``encode`` / ``decode`` / a vocab size. With the optional extra
(``pip install "oglodon-transformer[hf]"``) you can use a real Hugging Face tokenizer:

    python examples/06_external_tokenizer.py --hf gpt2      # needs network access to download it

Without ``--hf`` the example uses a small word-level tokenizer that follows the same protocol.
"""

import argparse

import torch
from torch.utils.data import DataLoader

from oglodon import Oglodon, OglodonConfig, Trainer
from oglodon.data import TokenBlockDataset
from oglodon.tokenization import HFTokenizerAdapter, Tokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--hf", default=None, help="name of a Hugging Face tokenizer, e.g. gpt2")
args = parser.parse_args()

TEXT = "the cat sat on the mat . the dog sat on the log . the cat saw the dog . the dog saw the cat . " * 40


class WordTokenizer:
    """Any object like this works: encode(str) -> list[int], decode(ids) -> str, vocab_size."""

    def __init__(self, text):
        self.words = sorted(set(text.split()))
        self.stoi = {w: i for i, w in enumerate(self.words)}
        self.vocab_size = len(self.words)

    def encode(self, text):
        return [self.stoi[w] for w in text.split()]

    def decode(self, ids):
        return " ".join(self.words[int(i)] for i in ids)


if args.hf:
    tok = HFTokenizerAdapter.from_pretrained(args.hf)
    text = "Transformers are neural networks built around attention. " * 50
else:
    tok, text = WordTokenizer(TEXT), TEXT
assert isinstance(tok, Tokenizer)
print("tokenizer vocab size:", tok.vocab_size)

ids = tok.encode(text)
loader = DataLoader(TokenBlockDataset(ids, 16, stride=2), batch_size=16, shuffle=True)
cfg = OglodonConfig(
    vocab_size=tok.vocab_size, hidden_size=64, num_layers=2, num_heads=4, num_kv_heads=2, max_seq_len=64
)
model = Oglodon(cfg)
Trainer(
    model, loader, learning_rate=3e-3, max_steps=150, warmup_steps=10, log_interval=50, weight_decay=0.0
).train()

prompt = torch.tensor([tok.encode(" ".join(text.split()[:3]) if not args.hf else "Transformers are")])
print(tok.decode(model.generate(prompt, max_new_tokens=12)[0].tolist()))
