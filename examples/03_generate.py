"""Generate text with a trained checkpoint (trains a tiny one first if none exists).

python examples/03_generate.py --checkpoint checkpoints/char_lm --prompt "the "
"""

import argparse
import time
from pathlib import Path

import torch

from oglodon import Oglodon
from oglodon.tokenization import load_tokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", default="checkpoints/char_lm")
parser.add_argument("--prompt", default="the ")
parser.add_argument("--tokens", type=int, default=100)
args = parser.parse_args()

if not Path(args.checkpoint, "config.json").exists():
    from _common import train_char_lm

    print("no checkpoint found: training a tiny character model first...")
    model, tok = train_char_lm(steps=400, logger=None)
    model.save_pretrained(args.checkpoint, tokenizer=tok)
model = Oglodon.from_pretrained(args.checkpoint)
tok = load_tokenizer(args.checkpoint)
prompt = torch.tensor([tok.encode(args.prompt)])

print("greedy        :", tok.decode(model.generate(prompt, max_new_tokens=args.tokens)[0].tolist()))
g = torch.Generator().manual_seed(42)
print(
    "temp 0.8 top-k:",
    tok.decode(
        model.generate(prompt, max_new_tokens=args.tokens, temperature=0.8, top_k=10, generator=g)[0].tolist()
    ),
)
g = torch.Generator().manual_seed(42)
print(
    "temp 1.0 top-p:",
    tok.decode(
        model.generate(prompt, max_new_tokens=args.tokens, temperature=1.0, top_p=0.9, generator=g)[
            0
        ].tolist()
    ),
)

outs = {}
for use_cache in (True, False):
    t = time.perf_counter()
    outs[use_cache] = model.generate(prompt, max_new_tokens=200, use_cache=use_cache)
    print(f"use_cache={use_cache!s:<5}: {time.perf_counter() - t:.2f}s for 200 tokens")
print("identical output with and without the cache:", torch.equal(outs[True], outs[False]))
