"""Compare MHA, GQA and MQA: parameters, speed and KV-cache size.

    python examples/05_compare_architectures.py
Timings are measured on YOUR machine; the 'kv (est)' column is an analytic estimate.
"""

from oglodon import OglodonConfig
from oglodon.benchmarks import compare_architectures, format_results

base = dict(vocab_size=1000, hidden_size=256, num_layers=6, num_heads=8, max_seq_len=512)
configs = [OglodonConfig(**base, num_kv_heads=k) for k in (8, 2, 1)]
results = compare_architectures(
    configs, batch_size=4, sequence_length=128, new_tokens=32, names=["mha", "gqa", "mqa"]
)
print(format_results(results))
