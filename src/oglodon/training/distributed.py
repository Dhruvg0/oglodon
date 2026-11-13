"""Helpers for multi-process data-parallel training (``torchrun`` + DDP).

Launch with ``torchrun --nproc_per_node=N train.py`` and, inside ``train.py``::

    from oglodon.training import Trainer, setup_distributed, distributed_loader

    ctx = setup_distributed()                      # reads RANK / WORLD_SIZE / LOCAL_RANK set by torchrun
    loader = distributed_loader(dataset, batch_size=8, shuffle=True, seed=0)   # per-process batch size
    Trainer(model, loader, distributed=True, device=ctx.device, ...).train()

Every process holds a full copy of the model (DDP). The effective batch is ``batch_size * world_size``
(times ``grad_accum_steps``). FSDP / tensor / pipeline parallelism are not implemented.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler


@dataclass
class DistContext:
    rank: int
    world_size: int
    local_rank: int
    device: torch.device

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def setup_distributed(backend: str | None = None) -> DistContext:
    """Initialise the default process group from ``torchrun``'s environment variables.

    ``backend`` defaults to ``nccl`` when CUDA is available, else ``gloo`` (CPU). Without the environment
    variables (a plain ``python`` launch) a single-process context is returned and no group is created.
    """
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return DistContext(0, 1, 0, torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    local = int(os.environ.get("LOCAL_RANK", 0))
    backend = backend or ("nccl" if torch.cuda.is_available() else "gloo")
    if backend == "nccl":
        torch.cuda.set_device(local)
        device = torch.device("cuda", local)
    else:
        device = torch.device("cpu")
    if not dist.is_initialized():
        dist.init_process_group(backend=backend)
    return DistContext(rank, world, local, device)


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def distributed_loader(
    dataset: Dataset, batch_size: int, shuffle: bool = True, seed: int = 0, drop_last: bool = True, **kw
) -> DataLoader:
    """A DataLoader that gives every process a different shard of ``dataset`` (``batch_size`` is per process).

    Call ``loader.sampler.set_epoch(epoch)`` between epochs if you iterate epochs yourself (``Trainer``
    cycles the loader without doing so; shuffling then repeats the same order every epoch).
    """
    sampler: DistributedSampler | None = None
    if dist.is_available() and dist.is_initialized():
        sampler = DistributedSampler(dataset, shuffle=shuffle, seed=seed, drop_last=drop_last)
    return DataLoader(
        dataset, batch_size=batch_size, sampler=sampler, shuffle=shuffle and sampler is None,
        drop_last=drop_last, **kw,
    )  # fmt: skip
