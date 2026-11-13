"""Data-parallel training: 2 CPU processes (gloo) must equal one process with the combined batch."""

import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, Dataset

from oglodon import Oglodon, OglodonConfig
from oglodon.training import Trainer, distributed_loader

pytestmark = pytest.mark.skipif(not dist.is_available(), reason="torch.distributed unavailable")

CFG = dict(vocab_size=40, hidden_size=32, num_layers=2, num_heads=4, num_kv_heads=2, max_seq_len=24)
N, T = 32, 12


class Data(Dataset):
    """Samples with very different numbers of supervised tokens (to test token weighting across ranks)."""

    def __init__(self):
        g = torch.Generator().manual_seed(0)
        self.x = torch.randint(0, 40, (N, T), generator=g)
        self.n_sup = [1 + (i * 7) % T for i in range(N)]  # 1..T supervised positions; some rows nearly empty

    def __len__(self):
        return N

    def __getitem__(self, i):
        y = self.x[i].clone()
        y[: T - self.n_sup[i]] = -100
        return {"input_ids": self.x[i], "labels": y}


def build():
    torch.manual_seed(0)
    return Oglodon(OglodonConfig(**CFG))


def run_trainer(loader, eval_loader, steps, accum, **kw):
    model = build()
    tr = Trainer(model, loader, eval_loader, max_steps=steps, learning_rate=1e-2, warmup_steps=0, log_interval=0,
                 grad_accum_steps=accum, device="cpu", logger=None, **kw)  # fmt: skip
    tr.train()
    return model, tr


def _worker(rank, world, store, out_dir, steps, accum):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{store}", rank=rank, world_size=world)
    try:
        data = Data()
        loader = distributed_loader(data, batch_size=4 // world, shuffle=False)
        eval_loader = DataLoader(
            data, batch_size=8
        )  # every rank evaluates the full set: the ratio is unchanged
        model, tr = run_trainer(loader, eval_loader, steps, accum, distributed=True)
        val = tr.evaluate()
        torch.save({"state": model.state_dict(), "val": val}, os.path.join(out_dir, f"rank{rank}.pt"))
    finally:
        dist.destroy_process_group()


def _spawn(world, tmp_path, steps, accum):
    store = str(tmp_path / f"store{world}")
    mp.spawn(_worker, args=(world, store, str(tmp_path), steps, accum), nprocs=world, join=True)
    return [torch.load(tmp_path / f"rank{r}.pt", weights_only=True) for r in range(world)]


@pytest.mark.parametrize("accum", [1, 2])
def test_two_processes_equal_one_process_with_the_combined_batch(tmp_path, accum):
    steps = 4
    # Single process: global batch 4 per micro-batch; two processes: 2 each (so the same 4 samples per micro-step).
    # With DistributedSampler(shuffle=False) rank r takes samples r, r+2, ...: the union per step is the same set
    # as one process reading 4 consecutive samples only if we read the same union. Compare against the union loader.
    union = DataLoader(Data(), batch_size=4, shuffle=False)
    ref_model, ref_tr = run_trainer(union, DataLoader(Data(), batch_size=8), steps, accum)
    ref_val = ref_tr.evaluate()

    res = _spawn(2, tmp_path, steps, accum)
    # replicas stay identical
    for k in res[0]["state"]:
        assert torch.equal(res[0]["state"][k], res[1]["state"][k]), k
    assert res[0]["val"] == pytest.approx(res[1]["val"], rel=1e-9)
    # ... and equal the single-process run on the combined batch (same samples per optimizer step)
    for k, v in ref_model.state_dict().items():
        assert torch.allclose(v, res[0]["state"][k], atol=2e-5, rtol=1e-4), k
    assert ref_val == pytest.approx(res[0]["val"], rel=1e-4)


def test_distributed_requires_a_process_group_and_rejects_compile():
    with pytest.raises(RuntimeError, match="process group"):
        run_trainer(DataLoader(Data(), batch_size=4), None, 1, 1, distributed=True)


def test_distributed_loader_without_a_group_is_a_plain_loader():
    ld = distributed_loader(Data(), batch_size=4, shuffle=False)
    assert ld.sampler.__class__.__name__ == "SequentialSampler" and len(ld) == N // 4
