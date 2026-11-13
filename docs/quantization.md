# Quantization

`oglodon quantize` makes an **int8 weight-only** copy of a checkpoint, in pure PyTorch (no extra dependency).

```bash
oglodon quantize run1 --out run1-int8 --text my_validation.txt
oglodon generate run1-int8 --prompt "Once upon"
```

## What it does

Every linear layer inside the Transformer blocks stores its weights as int8 with a float32 scale (one per output
channel, or one per `--group-size` input channels), and is dequantized on the fly in each forward pass. Embeddings,
norms and biases stay float. The output matrix stays float unless you pass `--include-lm-head` and the embeddings are
not tied (quantizing a tied matrix would silently untie it).

## What you get, and what you do not

* Smaller block weights on disk and in memory (about 4x for those layers; the whole file shrinks less when the
  vocabulary is large because embeddings stay float).
* **No speed-up**: dequantizing on the fly is not faster than float matmuls on CPU. GPU behaviour has not been run.
* **Inference only**: int8 weights have no gradients; `Trainer` cannot train them.
* **Lossy**. Always pass `--text` with text you care about. It writes `quantization_report.json` with perplexity before and
  after, mean KL divergence, top-1 agreement and the largest logit difference. Without `--text`, drift is not measured.

## Measured example

A 3-layer, 96-wide character model trained for 300 steps on CPU (a toy; do not extrapolate to large models):

| scales | size vs float | perplexity change | top-1 unchanged | mean weight error |
|---|---|---|---|---|
| per channel | 27% | +0.001% | 99.96% | 0.61% |
| group 32 | 30% | +0.001% | 99.98% | 0.52% |
| group 16 | 33% | +0.001% | 100% | 0.46% |

Smaller groups cost a little memory for scales and help most when a weight row has outliers.

## Python

```python
from oglodon import Oglodon
from oglodon.quantization import quantize_int8, quantization_report

model = Oglodon.from_pretrained("run1")
q = quantize_int8(model, group_size=32)
print(quantization_report(model, q, token_ids, block_size=128))
q.save_pretrained("run1-int8")
```

## Compatibility

`oglodon verify` works on int8 models. `validate_model` refuses them (it compares float weights with a float64
reference). Int8 checkpoints cannot be exported to Hugging Face, and an already quantized model cannot be quantized again.
