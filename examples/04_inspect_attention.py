"""Train a tiny model to reverse digit strings, then look at its attention maps.

python examples/04_inspect_attention.py   # writes attention.png
"""

import matplotlib

matplotlib.use("Agg")
import torch
from torch.utils.data import DataLoader

from oglodon import Oglodon, OglodonConfig, Trainer
from oglodon.data import AlgorithmicDataset, algorithmic_tokenizer, collate_pad
from oglodon.inspection import AttentionInspector
from oglodon.utils import seed_everything

seed_everything(0)
tok = algorithmic_tokenizer()
cfg = OglodonConfig(
    vocab_size=tok.vocab_size, hidden_size=64, num_layers=2, num_heads=4, num_kv_heads=2, max_seq_len=64
)
model = Oglodon(cfg)
loader = DataLoader(
    AlgorithmicDataset(50_000, tok, tasks=("reverse",)), batch_size=64, shuffle=True, collate_fn=collate_pad
)
Trainer(model, loader, learning_rate=3e-3, max_steps=700, warmup_steps=30, log_interval=100).train()

text = "R12345>" + "54321"
ids = torch.tensor([tok.encode(text)])
inspector = AttentionInspector(model)
result = inspector.inspect(ids, layer=1)  # last layer, all heads
print("attention tensor:", tuple(result.weights.shape), "(batch, heads, query, key)")
print("to emit the reversed digits the model should look back at prompt positions 5, 4, 3, 2, 1")
for h in result.heads:
    # queries at positions 6..10 (the '>' and the answer tokens) are the ones predicting answer digits
    print(
        f"head {h}: strongest key position for each of those queries:",
        result.weights[0, h, 6:11].argmax(-1).tolist(),
    )
fig = inspector.plot(result, tokens=list(text))
fig.savefig("attention.png", dpi=120)
print("saved attention.png")
