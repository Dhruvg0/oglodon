"""Validate a model against an independent reference, then run a small controlled MHA-vs-GQA comparison.

    python examples/07_validate_and_compare.py
The comparison is deliberately tiny (seconds on CPU); its numbers are a demonstration, not a result.
"""

from oglodon import Oglodon, OglodonConfig
from oglodon.data import make_corpus
from oglodon.experiments import ExperimentData, run_comparison
from oglodon.validation import validate_model

cfg = OglodonConfig(vocab_size=64, hidden_size=64, num_layers=2, num_heads=8, num_kv_heads=2, max_seq_len=64)
report = validate_model(Oglodon(cfg), cfg)
print(report)
report.save("validation_report.json")

data = ExperimentData.from_text(make_corpus(300), block_size=32)
base = OglodonConfig(vocab_size=data.vocab_size, hidden_size=64, num_layers=2, num_heads=8, max_seq_len=64)
cmp = run_comparison(
    base,
    {"mha": {"num_kv_heads": 8}, "gqa": {"num_kv_heads": 2}},
    data,
    steps=30,
    batch_size=16,
    seeds=(0, 1),
)
print(cmp.to_markdown())
