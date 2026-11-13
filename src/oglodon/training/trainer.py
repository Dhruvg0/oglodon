"""A small, readable training loop."""

from __future__ import annotations

import contextlib
import math
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from torch import nn
from torch.utils.data import DataLoader

if TYPE_CHECKING:
    from ..models.gpt import Oglodon

from ..utils.reproducibility import get_device
from .checkpoint import load_pretrained, load_training_state, save_training_state
from .scheduler import make_lr_lambda


def _unpack(batch: Any, device: torch.device) -> dict[str, torch.Tensor]:
    """Accept a dict with ``input_ids``/``labels`` (/``attention_mask``) or an ``(x, y)`` pair."""
    if isinstance(batch, dict):
        out = {k: v.to(device) for k, v in batch.items()}
    else:
        x, y = batch
        out = {"input_ids": x.to(device), "labels": y.to(device)}
    out.setdefault("labels", out["input_ids"])
    return out


class Trainer:
    """AdamW + warm-up/decay schedule + grad accumulation/clipping + evaluation.

    Args:
        model: a :class:`oglodon.Oglodon`.
        train_dataloader: yields dicts (``input_ids``, optional ``labels``/``attention_mask``)
            or ``(input_ids, labels)`` tuples. It is cycled until ``max_steps``.
        eval_dataloader: optional; evaluated every ``eval_interval`` steps and at the end.
        max_steps: number of **optimizer** steps (each uses ``grad_accum_steps`` micro-batches).
        distributed: wrap the model in DistributedDataParallel (needs an initialised process group, see
            :mod:`oglodon.training.distributed`). Losses are weighted by the *global* number of supervised tokens, so
            N processes give the same update as one process with the combined batch. Only rank 0 logs and saves.
        gradient_checkpointing: recompute block activations in backward (less memory, ~1/3 more compute).
        compile: run the forward/backward through ``torch.compile`` (first step is slow while it compiles).
        precision: ``"fp32"`` (default), ``"bf16"`` (autocast) or ``"fp16"`` (CUDA only, with grad scaling).
        checkpoint_dir / save_interval: where/how often to write checkpoints (model + optimizer state).
        device: defaults to CUDA if available, else CPU.
    """

    def __init__(
        self,
        model: Oglodon,
        train_dataloader: DataLoader,
        eval_dataloader: DataLoader | None = None,
        learning_rate: float = 3e-4,
        weight_decay: float = 0.1,
        max_steps: int = 1000,
        warmup_steps: int = 0,
        lr_schedule: str = "cosine",
        min_lr_ratio: float = 0.1,
        grad_accum_steps: int = 1,
        max_grad_norm: float | None = 1.0,
        betas: tuple[float, float] = (0.9, 0.95),
        eval_interval: int = 0,
        eval_batches: int | None = None,
        log_interval: int = 50,
        precision: str = "fp32",
        gradient_checkpointing: bool = False,
        compile: bool = False,
        distributed: bool = False,
        device: str | torch.device | None = None,
        checkpoint_dir: str | Path | None = None,
        save_interval: int = 0,
        logger: Callable[[str], None] | None = print,
    ) -> None:
        if grad_accum_steps < 1 or max_steps < 1:
            raise ValueError("grad_accum_steps and max_steps must be >= 1")
        if precision not in ("fp32", "bf16", "fp16"):
            raise ValueError("precision must be 'fp32', 'bf16' or 'fp16'")
        self.device = get_device(device)
        if precision == "fp16" and self.device.type != "cuda":
            raise ValueError("fp16 mixed precision requires CUDA; use 'bf16' on CPU")
        self.model = model.to(self.device)
        if gradient_checkpointing:
            model.gradient_checkpointing_enable()
        # ``model`` stays the object that is saved/loaded; ``_forward`` is what gets called.
        self._forward: Any = torch.compile(self.model) if compile else self.model
        self.distributed = distributed
        self.rank, self.world_size = 0, 1
        if distributed:
            import torch.distributed as dist

            if compile:
                raise ValueError("compile together with distributed is not supported yet")
            if not (dist.is_available() and dist.is_initialized()):
                raise RuntimeError(
                    "distributed=True needs an initialised process group (see setup_distributed())"
                )
            self.rank, self.world_size = dist.get_rank(), dist.get_world_size()
            ids = [self.device.index] if self.device.type == "cuda" else None
            self._forward = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=ids)

        self.train_dataloader, self.eval_dataloader = train_dataloader, eval_dataloader
        self.max_steps, self.grad_accum_steps = max_steps, grad_accum_steps
        self.max_grad_norm, self.precision = max_grad_norm, precision
        self.eval_interval, self.eval_batches = eval_interval, eval_batches
        self.log_interval, self.save_interval = log_interval, save_interval
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.logger = (logger or (lambda _msg: None)) if self.rank == 0 else (lambda _msg: None)

        # No weight decay on 1-D parameters (norm gains) or embeddings.
        decay = [
            p for n, p in model.named_parameters() if p.requires_grad and p.dim() >= 2 and "embed" not in n
        ]
        no_decay = [
            p for n, p in model.named_parameters() if p.requires_grad and (p.dim() < 2 or "embed" in n)
        ]
        self.optimizer = torch.optim.AdamW(
            [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
            lr=learning_rate,
            betas=betas,
        )
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, make_lr_lambda(lr_schedule, warmup_steps, max_steps, min_lr_ratio)
        )
        self.scaler = torch.amp.GradScaler("cuda") if precision == "fp16" else None
        self.step = 0
        self.history: list[dict[str, float]] = []
        self.batches_seen = 0  # micro-batches consumed so far (stored in checkpoints)
        self.eval_seconds = 0.0  # wall-clock spent in periodic evaluation inside train()
        self._batch_iter: Iterator[dict[str, torch.Tensor]] | None = None
        self._skip_batches = 0
        self._pending_rng: dict[str, Any] | None = None

    # ------------------------------------------------------------------ helpers
    def _autocast(self):
        if self.precision == "fp32":
            return contextlib.nullcontext()
        dtype = torch.bfloat16 if self.precision == "bf16" else torch.float16
        return torch.autocast(device_type=self.device.type, dtype=dtype)

    def _batches(self) -> Iterator[dict[str, torch.Tensor]]:
        # On resume, replay (and discard) the micro-batches that were already consumed so that
        # training continues on the data it would have seen. Exact for deterministic loaders.
        # The RNG is restored at the exact point the original run had saved it: right after the
        # last replayed batch (or before the first epoch iterator is created when nothing is replayed).
        # Creating a DataLoader iterator draws from the global RNG, so restoring any later would shift
        # the dropout stream by one draw whenever the checkpoint falls on an epoch boundary.
        skipped = 0
        self._restore_pending_rng_if(self._skip_batches == 0)
        while True:
            for batch in self.train_dataloader:
                if skipped < self._skip_batches:
                    skipped += 1
                    self._restore_pending_rng_if(skipped == self._skip_batches)
                    continue
                yield _unpack(batch, self.device)

    def _restore_pending_rng_if(self, condition: bool) -> None:
        if condition and self._pending_rng is not None:
            self._restore_rng(self._pending_rng)
            self._pending_rng = None

    def _next_batch(self) -> dict[str, torch.Tensor]:
        if self._batch_iter is None:
            self._batch_iter = self._batches()
        self.batches_seen += 1
        return next(self._batch_iter)

    @staticmethod
    def _capture_rng() -> dict[str, Any]:
        state: dict[str, Any] = {"cpu": torch.get_rng_state()}
        if torch.cuda.is_available():
            state["cuda"] = torch.cuda.get_rng_state_all()
        return state

    @staticmethod
    def _restore_rng(state: dict[str, Any]) -> None:
        torch.set_rng_state(state["cpu"].cpu())
        if "cuda" in state and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([t.cpu() for t in state["cuda"]])

    def compute_gradients(self, micro_batches: list[dict[str, torch.Tensor]]) -> float:
        """Zero the gradients, then accumulate the gradient of the token-mean loss over ``micro_batches``.

        Each micro-batch is weighted by its share of supervised tokens, so N accumulated micro-batches
        give exactly the gradient of one batch N times larger. Micro-batches without supervised
        tokens are skipped (their mean loss would be NaN). Returns the combined loss, or ``nan`` if
        no micro-batch had any supervised token (gradients are then left unset).
        """
        self.optimizer.zero_grad(set_to_none=True)
        counts = [int((b["labels"][:, 1:] != -100).sum()) for b in micro_batches]
        total = sum(counts)
        scale = 1
        if self.distributed:
            # Weight by the *global* token count; DDP then averages over processes, so multiply by the
            # world size to get the sum, i.e. the gradient of the token-mean loss over all processes' batches.
            total = int(self._all_reduce_sum(float(total)))
            scale = self.world_size
        if total == 0:
            return float("nan")
        step_loss = 0.0
        last = len(micro_batches) - 1
        for i, (batch, n_tok) in enumerate(zip(micro_batches, counts, strict=True)):
            # DDP synchronises gradients only on the last micro-batch of the step.
            sync = contextlib.nullcontext()
            if self.distributed and i < last:
                sync = self._forward.no_sync()
            with sync:
                if n_tok == 0:
                    if not self.distributed:
                        continue
                    # Every process must take part in the backward all-reduce, so a process with nothing
                    # supervised still runs a zero-weight pass instead of skipping it.
                    with self._autocast():
                        dummy = self._forward(**batch).logits.float().sum() * 0.0
                    (self.scaler.scale(dummy) if self.scaler else dummy).backward()
                    continue
                with self._autocast():
                    loss = self._forward(**batch).loss * (n_tok / total)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite loss at step {self.step}: {loss.item()}")
                scaled = loss * scale
                (self.scaler.scale(scaled) if self.scaler else scaled).backward()
                step_loss += loss.item()
        if self.distributed:
            step_loss = self._all_reduce_sum(step_loss)  # each process contributed its weighted part
        return step_loss

    def _all_reduce_sum(self, value: float) -> float:
        import torch.distributed as dist

        t = torch.tensor(value, dtype=torch.float64, device=self.device)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        return float(t.item())

    # ------------------------------------------------------------------ public API
    @torch.no_grad()
    def evaluate(self) -> float:
        """Mean token-weighted loss over the eval dataloader."""
        if self.eval_dataloader is None:
            raise ValueError("no eval_dataloader was given")
        was_training = self.model.training
        self.model.eval()
        total, count = 0.0, 0
        for i, batch in enumerate(self.eval_dataloader):
            if self.eval_batches is not None and i >= self.eval_batches:
                break
            b = _unpack(batch, self.device)
            if int((b["labels"][:, 1:] != -100).sum()) == 0:
                continue
            with self._autocast():
                out = self._forward(**b)
            n = int((b["labels"][:, 1:] != -100).sum())
            if n == 0:  # nothing supervised: its mean loss is NaN
                continue
            total += out.loss.item() * n
            count += n
        self.model.train(was_training)
        if self.distributed:  # sums over processes; the ratio is also right if every process saw the full set
            total, count = self._all_reduce_sum(total), int(self._all_reduce_sum(float(count)))
        return total / max(count, 1)

    def train(self, until: int | None = None) -> list[dict[str, float]]:
        """Run until ``max_steps`` optimizer steps (or ``until`` if smaller); returns the logged history.

        ``until`` stops early *without* changing the learning-rate schedule (which is defined by
        ``max_steps``), so a later ``train()`` or a resumed run continues the same schedule.
        """
        target = self.max_steps if until is None else min(until, self.max_steps)
        self.model.train()
        t0, running, running_n = time.time(), 0.0, 0
        while self.step < target:
            micro = [self._next_batch() for _ in range(self.grad_accum_steps)]
            step_loss = self.compute_gradients(micro)
            grad_norm = None
            if math.isnan(step_loss):
                self.logger(
                    f"step {self.step + 1}: no supervised tokens in this step's batches, skipping update"
                )
                step_loss = 0.0
            else:
                if self.scaler:
                    self.scaler.unscale_(self.optimizer)
                if self.max_grad_norm is not None:
                    grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                if self.scaler:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
            self.scheduler.step()
            self.step += 1
            running, running_n = running + step_loss, running_n + 1

            if self.log_interval and (self.step % self.log_interval == 0 or self.step == self.max_steps):
                rec: dict[str, float] = {
                    "step": self.step,
                    "train_loss": running / running_n,
                    "lr": float(self.scheduler.get_last_lr()[0]),
                }
                if grad_norm is not None:
                    rec["grad_norm"] = float(grad_norm)
                self.history.append(rec)
                self.logger(
                    f"step {self.step:>6} | loss {rec['train_loss']:.4f} | lr {rec['lr']:.2e}"
                    f" | {time.time() - t0:.0f}s"
                )
                running, running_n = 0.0, 0
            if (
                self.eval_dataloader is not None
                and self.eval_interval
                and (self.step % self.eval_interval == 0 or self.step == self.max_steps)
            ):
                t_eval = time.perf_counter()
                val = self.evaluate()
                self.eval_seconds += time.perf_counter() - t_eval
                self.history.append({"step": self.step, "val_loss": val, "val_ppl": math.exp(min(val, 20))})
                self.logger(f"step {self.step:>6} | val loss {val:.4f}")
            if self.checkpoint_dir and self.save_interval and self.step % self.save_interval == 0:
                self.save_checkpoint()
        if self.checkpoint_dir:
            self.save_checkpoint()
        return self.history

    # ---------------------------------------------------------------- checkpoints
    def save_checkpoint(self, path: str | Path | None = None, tokenizer: Any = None) -> Path:
        """Write the model (safetensors) and optimizer/scheduler/step state."""
        path = Path(path or self.checkpoint_dir or "checkpoint")
        if self.rank != 0:  # data-parallel replicas are identical: only rank 0 writes
            return path
        self.model.save_pretrained(path, tokenizer)
        save_training_state(
            path,
            {
                "step": self.step,
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "history": self.history,
                "batches_seen": self.batches_seen,
                "rng": self._capture_rng(),
                "scaler": self.scaler.state_dict() if self.scaler else None,
            },
        )
        return path

    def load_checkpoint(self, path: str | Path) -> None:
        """Restore weights, optimizer, scheduler, step counter, RNG state and data position.

        Training then continues as if it had never stopped, provided the dataloader is deterministic
        (``shuffle=False``, or seeded identically to the original run).
        """
        loaded = load_pretrained(type(self.model), path, self.device)
        self.model.load_state_dict(loaded.state_dict())
        state = load_training_state(path, self.device)
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        self.step, self.history = int(state["step"]), list(state["history"])
        if self.scaler and state.get("scaler"):
            self.scaler.load_state_dict(state["scaler"])
        # Data replay + RNG restore make resume continue the original run (deterministic loaders).
        self._batch_iter = None
        self.batches_seen = self._skip_batches = int(state.get("batches_seen", 0))
        self._pending_rng = state.get("rng")
