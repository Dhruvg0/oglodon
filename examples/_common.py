"""Shared helpers for the examples (a tiny built-in corpus so nothing external is needed)."""

from torch.utils.data import DataLoader

from oglodon import Oglodon, OglodonConfig, Trainer
from oglodon.data import TokenBlockDataset, make_corpus
from oglodon.tokenization import CharTokenizer

CORPUS = make_corpus()


def train_char_lm(steps: int = 400, block_size: int = 64, seed: int = 0, logger=print):
    """Train a ~0.3M-parameter character LM on CORPUS. Returns (model, tokenizer)."""
    import torch

    torch.manual_seed(seed)
    tok = CharTokenizer.from_text(CORPUS)
    ids = tok.encode(CORPUS)
    split = int(len(ids) * 0.9)
    train = TokenBlockDataset(ids[:split], block_size, stride=8)
    val = TokenBlockDataset(ids[split:], block_size, stride=16)
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
        DataLoader(train, batch_size=32, shuffle=True),
        DataLoader(val, batch_size=32),
        learning_rate=3e-3,
        weight_decay=0.01,
        max_steps=steps,
        warmup_steps=20,
        eval_interval=max(1, steps // 4),
        log_interval=max(1, steps // 8),
        logger=logger,
    )
    trainer.train()
    return model, tok
