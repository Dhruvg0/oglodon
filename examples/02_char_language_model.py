"""Train a character-level language model on custom text.

python examples/02_char_language_model.py                 # built-in corpus
python examples/02_char_language_model.py --file my.txt   # your own text
"""

import argparse
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from oglodon import Oglodon, OglodonConfig, Trainer
from oglodon.data import TokenBlockDataset
from oglodon.tokenization import CharTokenizer
from oglodon.utils import seed_everything

parser = argparse.ArgumentParser()
parser.add_argument("--file", default=None)
parser.add_argument("--steps", type=int, default=600)
parser.add_argument("--out", default="checkpoints/char_lm")
args = parser.parse_args()

seed_everything(0)
if args.file:
    text = Path(args.file).read_text()
else:
    from _common import CORPUS

    text = CORPUS  # synthetic grammar; pass --file to train on your own text
tok = CharTokenizer.from_text(text)
ids = tok.encode(text)
cut = int(0.9 * len(ids))
block = 64
train = DataLoader(TokenBlockDataset(ids[:cut], block, stride=8), batch_size=32, shuffle=True)
val = DataLoader(TokenBlockDataset(ids[cut:], block, stride=16), batch_size=32)
cfg = OglodonConfig(
    vocab_size=tok.vocab_size,
    hidden_size=96,
    num_layers=3,
    num_heads=4,
    num_kv_heads=2,
    max_seq_len=256,
    pad_token_id=tok.pad_token_id,
    eos_token_id=tok.eos_token_id,
)
model = Oglodon(cfg)
trainer = Trainer(
    model,
    train,
    val,
    learning_rate=3e-3,
    weight_decay=0.01,
    max_steps=args.steps,
    warmup_steps=20,
    eval_interval=args.steps // 4,
    log_interval=args.steps // 8,
)
trainer.train()
val_loss = trainer.evaluate()
print(
    f"final val loss {val_loss:.3f} (perplexity {math.exp(val_loss):.2f}; a uniform guess would be {tok.vocab_size})"
)
model.save_pretrained(args.out, tokenizer=tok)

prompt = torch.tensor([tok.encode(text[:12])])
out = model.generate(
    prompt, max_new_tokens=120, temperature=0.8, top_k=20, generator=torch.Generator().manual_seed(0)
)
print("\nsample:\n" + tok.decode(out[0].tolist()))
