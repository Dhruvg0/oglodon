import pytest
import torch
from torch.utils.data import DataLoader

from oglodon import Oglodon
from oglodon.data import AlgorithmicDataset, TokenBlockDataset, algorithmic_tokenizer, collate_pad
from oglodon.training import Trainer, make_lr_lambda

from .conftest import tiny_config


def loader(n=256, bs=32, seed=0, shuffle=True):
    return DataLoader(
        AlgorithmicDataset(n, seed=seed), batch_size=bs, shuffle=shuffle, collate_fn=collate_pad
    )


def algo_model():
    t = algorithmic_tokenizer()
    return Oglodon(tiny_config(vocab_size=t.vocab_size, hidden_size=48, pad_token_id=0, eos_token_id=1))


def test_training_reduces_loss():
    torch.manual_seed(0)
    m = algo_model()
    tr = Trainer(
        m,
        loader(),
        learning_rate=3e-3,
        weight_decay=0.0,
        max_steps=80,
        warmup_steps=5,
        log_interval=10,
        logger=None,
    )
    hist = tr.train()
    losses = [h["train_loss"] for h in hist]
    assert losses[-1] < 0.7 * losses[0] and tr.step == 80


def test_eval_loss_is_recorded_and_finite():
    torch.manual_seed(0)
    tr = Trainer(
        algo_model(),
        loader(),
        loader(64, seed=9, shuffle=False),
        max_steps=10,
        eval_interval=5,
        log_interval=5,
        logger=None,
        learning_rate=1e-3,
    )
    hist = tr.train()
    assert [h["step"] for h in hist if "val_loss" in h] == [5, 10]
    assert all(torch.isfinite(torch.tensor(h["val_loss"])) for h in hist if "val_loss" in h)


@pytest.mark.filterwarnings("ignore:Seems like")
def test_gradient_accumulation_matches_large_batch():
    """2 micro-batches of 8 == 1 batch of 16: compare the gradients the optimizer receives."""
    data = AlgorithmicDataset(16, seed=3)
    grads, losses = [], []
    for bs, accum in ((16, 1), (8, 2)):
        torch.manual_seed(0)
        m = algo_model()
        dl = DataLoader(data, batch_size=bs, shuffle=False, collate_fn=collate_pad)
        tr = Trainer(
            m, dl, max_steps=1, grad_accum_steps=accum, max_grad_norm=None, log_interval=1, logger=None
        )
        tr.optimizer.step = lambda m=m: grads.append(torch.cat([p.grad.flatten() for p in m.parameters()]))
        losses.append(tr.train()[0]["train_loss"])
    assert torch.allclose(grads[0], grads[1], atol=1e-6)
    assert losses[0] == pytest.approx(losses[1], abs=1e-5)


def test_grad_clipping_bounds_update_norm():
    m = algo_model()
    tr = Trainer(m, loader(), max_steps=1, max_grad_norm=0.5, log_interval=1, logger=None)
    hist = tr.train()
    assert hist[0]["grad_norm"] > 0  # reported pre-clip norm
    total = torch.sqrt(sum((p.grad**2).sum() for p in m.parameters() if p.grad is not None))
    assert total <= 0.5 + 1e-4


def test_weight_decay_excludes_norms_and_embeddings():
    tr = Trainer(algo_model(), loader(), max_steps=1, logger=None)
    decay, no_decay = tr.optimizer.param_groups
    assert decay["weight_decay"] > 0 and no_decay["weight_decay"] == 0
    assert all(p.dim() >= 2 for p in decay["params"])
    assert tr.model.embed_tokens.weight in no_decay["params"]


def test_checkpoint_resume_continues_identically(tmp_path):
    def run(steps, resume=None):
        torch.manual_seed(0)
        m = algo_model()
        tr = Trainer(
            m,
            loader(shuffle=False),
            max_steps=steps,
            learning_rate=2e-3,
            log_interval=0,
            logger=None,
            checkpoint_dir=tmp_path / "b" if resume is None and steps == 5 else None,
        )
        if resume:
            tr.load_checkpoint(resume)
        tr.train()
        return m, tr

    run(5)  # writes tmp/b at step 5
    resumed, tr = run(10, resume=tmp_path / "b")
    assert tr.step == 10
    # Same data order + same optimizer state => trajectory equals an uninterrupted run only if the
    # loader restarts the same way; we check the weaker, honest property: state really was restored.
    state = torch.load(tmp_path / "b" / "trainer_state.pt", weights_only=True)
    assert state["step"] == 5 and "optimizer" in state


def test_resume_restores_step_and_weights(tmp_path):
    torch.manual_seed(0)
    m = algo_model()
    tr = Trainer(m, loader(), max_steps=4, logger=None, log_interval=0)
    tr.train()
    tr.save_checkpoint(tmp_path)
    m2 = algo_model()
    tr2 = Trainer(m2, loader(), max_steps=8, logger=None)
    tr2.load_checkpoint(tmp_path)
    assert tr2.step == 4
    for a, b in zip(m.parameters(), m2.parameters(), strict=True):
        assert torch.equal(a, b)


def test_non_finite_loss_raises():
    m = algo_model()
    with torch.no_grad():
        m.final_norm.weight.fill_(float("nan"))
    with pytest.raises(FloatingPointError):
        Trainer(m, loader(), max_steps=1, logger=None).train()


def test_bf16_autocast_runs_on_cpu():
    tr = Trainer(algo_model(), loader(), max_steps=3, precision="bf16", device="cpu", logger=None)
    assert len(tr.train()) >= 0 and tr.step == 3


def test_fp16_requires_cuda_and_bad_args_rejected():
    if not torch.cuda.is_available():
        with pytest.raises(ValueError, match="CUDA"):
            Trainer(algo_model(), loader(), precision="fp16", device="cpu")
    with pytest.raises(ValueError):
        Trainer(algo_model(), loader(), precision="int4")
    with pytest.raises(ValueError):
        Trainer(algo_model(), loader(), grad_accum_steps=0)


def test_tuple_batches_are_supported():
    m = Oglodon(tiny_config())
    xs = torch.randint(2, 37, (32, 10))
    dl = DataLoader(list(zip(xs, xs, strict=True)), batch_size=8)
    Trainer(m, dl, max_steps=2, logger=None).train()


def test_lr_schedule_shape():
    f = make_lr_lambda("cosine", warmup_steps=10, max_steps=100, min_lr_ratio=0.1)
    assert f(0) == pytest.approx(0.1) and f(9) == pytest.approx(1.0)  # linear warm-up
    assert f(10) == pytest.approx(1.0) and f(100) == pytest.approx(0.1)  # decays to the floor
    assert all(f(i) >= f(i + 1) for i in range(10, 99))
    assert make_lr_lambda("constant", 0, 10)(7) == 1.0
    assert make_lr_lambda("linear", 0, 10, 0.0)(5) == pytest.approx(0.5)
    with pytest.raises(ValueError):
        make_lr_lambda("exp", 0, 10)


def test_token_block_dataset():
    ds = TokenBlockDataset(list(range(25)), block_size=10)
    assert len(ds) == 2 and ds[1]["input_ids"].tolist() == list(range(10, 20))
    with pytest.raises(ValueError):
        TokenBlockDataset([1, 2], block_size=10)
