"""Measure MHA vs GQA vs MQA on this machine (nothing here is a pre-recorded number).

python benchmarks/run_comparison.py --hidden 256 --layers 6 --heads 8 --seq 128 --batch 4
"""

import argparse

from oglodon import OglodonConfig
from oglodon.benchmarks import compare_architectures, format_results

p = argparse.ArgumentParser()
p.add_argument("--hidden", type=int, default=256)
p.add_argument("--layers", type=int, default=6)
p.add_argument("--heads", type=int, default=8)
p.add_argument("--seq", type=int, default=128)
p.add_argument("--new-tokens", type=int, default=32)
p.add_argument("--batch", type=int, default=4)
p.add_argument("--repeats", type=int, default=5)
a = p.parse_args()

kv_options = {"mha": a.heads, "gqa": max(1, a.heads // 4), "mqa": 1}
configs = [
    OglodonConfig(
        vocab_size=1000,
        hidden_size=a.hidden,
        num_layers=a.layers,
        num_heads=a.heads,
        num_kv_heads=kv,
        max_seq_len=a.seq + a.new_tokens,
    )
    for kv in kv_options.values()
]
results = compare_architectures(configs, a.batch, a.seq, a.new_tokens, list(kv_options), a.repeats)
print(format_results(results))
print("\nforward/generate times and cache sizes are measured; 'kv (est)' is the analytic formula.")
