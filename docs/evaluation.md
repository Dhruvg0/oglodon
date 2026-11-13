# Evaluation

All evaluation works on any causal LM through the same adapter as [`verify`](verify.md) (Oglodon, a Hugging Face model,
your own module): only token ids are needed.

## Perplexity

```bash
oglodon eval run1 --text heldout.txt --block-size 128 --stride 64
```

```python
from oglodon.evaluation import perplexity

r = perplexity(model, token_ids, block_size=128, stride=64)
r.perplexity, r.loss, r.bits_per_token, r.tokens_scored
```

A sliding window scores each token once, in the first window where it has context. With `stride == block_size` (the
default, fastest) the first token of every later window has no context and is skipped, and tokens near a window start
see little history, which inflates perplexity. A smaller stride gives each token more context and a lower, more
faithful number at higher cost. Compare perplexities only for the same tokenizer and text; a uniform model scores
exactly the vocabulary size.

## Multiple choice (log-likelihood)

```bash
oglodon eval run1 --choices task.jsonl
```

One JSON object per line: `{"context": "...", "choices": ["...", "..."], "label": 0}`. Each choice is scored by the
model's log-likelihood of the choice after the context (context and choices are tokenized separately, so include a
leading space in choices when your convention needs one). Reported: `acc` (total log-likelihood), `acc_norm` (per
character, the convention of common harnesses), `acc_token_norm` (per token) and the binomial standard error. Chance
is `1 / len(choices)`; a result near chance means the model has not learned the task. Tiny models trained on small
corpora will sit near chance on real benchmarks, and that is the honest outcome.

To run **standard** benchmarks, export the model ([Interoperability](interoperability.md)) and point
lm-evaluation-harness at the exported folder; that path is documented but was not run here.

## Generation sanity checks

```bash
oglodon eval run1 --prompt "Once upon" "The" --max-new-tokens 32
```

Reports distinct-1/2 (diversity), the share of repeated 3-grams, whether two greedy runs agree, and whether greedy
output with and without the KV cache agrees. These describe behaviour (a looping model scores low on diversity); they
are not a quality score.

## How the numbers were checked

Each evaluation function is tested against a deliberately naive computation (one prefix at a time, no windows or
batching), against models with known preferences, and for batch-size independence.
