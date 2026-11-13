"""Train a tiny model on addition / sorting / reversal and report exact-match accuracy.

This is the original MiniGPT notebook experiment, rebuilt on the Oglodon API.
    python examples/01_train_algorithmic.py --steps 5000 --out checkpoints/algorithmic
"""

import argparse

import torch
from torch.utils.data import DataLoader

from oglodon import Oglodon, OglodonConfig, Trainer
from oglodon.data import AlgorithmicDataset, algorithmic_tokenizer, collate_pad, exact_match
from oglodon.utils import seed_everything

parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int, default=6000)
parser.add_argument("--out", default="checkpoints/algorithmic")
args = parser.parse_args()

seed_everything(0)
tok = algorithmic_tokenizer()
cfg = OglodonConfig(
    vocab_size=tok.vocab_size,
    hidden_size=128,
    num_layers=4,
    num_heads=4,
    num_kv_heads=2,
    max_seq_len=64,
    pad_token_id=tok.pad_token_id,
    eos_token_id=tok.eos_token_id,
)
model = Oglodon(cfg)
print(f"{model.num_parameters() / 1e6:.2f}M parameters ({cfg.attention_type})")

train = DataLoader(
    AlgorithmicDataset(200_000, tok, seed=0, task_weights={"add": 2}),
    batch_size=64,
    shuffle=True,
    collate_fn=collate_pad,
)
val = DataLoader(AlgorithmicDataset(512, tok, seed=1), batch_size=128, collate_fn=collate_pad)
Trainer(
    model,
    train,
    val,
    learning_rate=2e-3,
    betas=(0.9, 0.98),
    weight_decay=0.01,
    max_steps=args.steps,
    warmup_steps=100,
    eval_interval=max(1, args.steps // 5),
    log_interval=max(1, args.steps // 10),
).train()

print("\nexact match (training lengths: add<=4, sort/reverse<=8; beyond that is out-of-distribution)")
for task, lengths in {"add": range(1, 7), "sort": range(2, 11), "reverse": range(2, 11)}.items():
    row = {n: f"{exact_match(model, task, n, 200):.0%}" for n in lengths}
    print(f"  {task:<8}{row}")

model.save_pretrained(args.out, tokenizer=tok)
print(f"saved to {args.out}")


def solve(prompt: str) -> str:
    out = model.generate(torch.tensor([tok.encode(prompt)]), max_new_tokens=8)
    return tok.decode(out[0, len(prompt) :].tolist())


# 1234 + 8766 = 10000; operands and answer are written least-significant digit first
print("A4321+6678= ->", solve("A4321+6678="), "(expected 00001, i.e. 10000 reversed)")
