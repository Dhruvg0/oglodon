# Getting started

Requirements: Python 3.11+, PyTorch. Everything below runs on a CPU.

## Install

```bash
pip install oglodon-transformer            # once published; see publishing.md
# from a source checkout:
pip install -e ".[dev]"
```

Optional extras: `[viz]` (matplotlib, for plots), `[hf]` (transformers, for Hugging Face import/export and
verification), `[yaml]` (PyYAML, for YAML config files; JSON works without it).

Check the installation:

```bash
oglodon --version
oglodon info        # versions, CUDA availability
```

## Train a model on your own text (2 minutes)

Put some `.txt` / `.md` files in a folder (a few hundred kilobytes is enough to see learning) and run:

```bash
oglodon train my_texts/ --out run1 --steps 500
```

This single command trains a byte-level BPE tokenizer, tokenizes the files into a memory-mapped file, trains a small
model (4 layers, hidden size 128, grouped-query attention by default), and writes a checkpoint folder. Progress prints
as it goes, ending with the validation perplexity next to the "uniform guess" baseline (the vocabulary size). A model
that has learned anything scores far below that baseline.

Then:

```bash
oglodon generate run1 --prompt "Once upon a time" --max-new-tokens 80 --seed 1
oglodon eval run1 --text my_texts/some_file.txt --block-size 128
```

`generate` streams text to the terminal. `eval` reports perplexity (use text the model did not train on for a
meaningful number).

## What `run1/` contains

| file | purpose |
|---|---|
| `model.safetensors`, `config.json` | the model (`Oglodon.from_pretrained("run1")`) |
| `tokenizer.json` | the tokenizer (`oglodon.tokenization.load_tokenizer("run1")`) |
| `trainer_state.pt` | optimizer, schedule, RNG and data position, for `--resume` |
| `train_settings.json` | every setting of the run |
| `training_report.json` | losses, perplexity, tokens/s, parameter count, library versions, file hashes |
| `data/corpus.bin`, `data/corpus.json` | tokenized corpus and its sidecar (file list with SHA-256) |

## The same thing from Python

```python
from oglodon.workflow import TrainSettings, train_from_files, complete
from oglodon import Oglodon
from oglodon.tokenization import load_tokenizer

train_from_files("my_texts/", "run1", TrainSettings(steps=500))

model, tok = Oglodon.from_pretrained("run1"), load_tokenizer("run1")
for piece in complete(model, tok, "Once upon a time", max_new_tokens=80, seed=1):
    print(piece, end="", flush=True)
```

## Be realistic about quality

A model trained for minutes on a small corpus learns the vocabulary and local structure of its text; it will not be a
fluent language model. Quality comes from more data, more steps and a larger model; the numbers in the report tell you
honestly where you are.

## Next steps

* More control over training: [Training](training.md).
* Check a model you did not train: [verify](verify.md).
* Compare two architectures fairly: [Experiments](experiments.md).
