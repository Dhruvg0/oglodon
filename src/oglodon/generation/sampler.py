"""Token sampling and autoregressive generation."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any

import torch

_UNSET: Any = object()  # distinguishes "not given" (use config) from an explicit None (no EOS)


def filter_logits(
    logits: torch.Tensor, top_k: int | None = None, top_p: float | None = None, min_p: float | None = None
) -> torch.Tensor:
    """Mask logits outside the min-p / top-k / nucleus (top-p) set with ``-inf``.

    ``min_p`` keeps tokens whose probability is at least ``min_p`` times that of the most likely token.

    Top-p keeps the smallest set of most likely tokens whose cumulative
    probability reaches ``top_p`` (always at least one token).
    """
    if min_p is not None:
        if not 0.0 <= min_p <= 1.0:
            raise ValueError("min_p must be in [0, 1]")
        probs = logits.softmax(dim=-1)
        logits = logits.masked_fill(probs < min_p * probs.max(dim=-1, keepdim=True).values, float("-inf"))
    if top_k is not None:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        k = min(top_k, logits.shape[-1])
        kth = torch.topk(logits, k, dim=-1).values[..., -1, None]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if top_p is not None:
        if not 0.0 < top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1]")
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        probs = sorted_logits.softmax(dim=-1)
        # Drop a token if the mass *before* it already reaches top_p.
        drop = (probs.cumsum(dim=-1) - probs) >= top_p
        sorted_logits = sorted_logits.masked_fill(drop, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(-1, sorted_idx, sorted_logits)
    return logits


def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 0.0,
    top_k: int | None = None,
    top_p: float | None = None,
    generator: torch.Generator | None = None,
    min_p: float | None = None,
) -> torch.Tensor:
    """Pick the next token for each row of ``logits`` (``(B, V)``) -> ``(B,)``.

    ``temperature == 0`` is greedy argmax; otherwise logits are divided by the
    temperature, filtered by top-k/top-p and sampled.
    """
    if temperature < 0:
        raise ValueError("temperature must be >= 0")
    if temperature == 0:
        return logits.argmax(dim=-1)
    logits = filter_logits(logits.float() / temperature, top_k, top_p, min_p)
    probs = logits.softmax(dim=-1)
    if generator is not None and generator.device != probs.device:
        # a CPU generator (the usual way to seed) with CUDA logits: draw on the generator's device
        return (
            torch.multinomial(probs.to(generator.device), 1, generator=generator).squeeze(-1).to(probs.device)
        )
    return torch.multinomial(probs, 1, generator=generator).squeeze(-1)


def apply_repetition_penalty(
    logits: torch.Tensor, ids: torch.Tensor, mask: torch.Tensor, penalty: float
) -> torch.Tensor:
    """CTRL-style repetition penalty: tokens already present (and not padding) become less likely.

    Positive logits are divided by ``penalty``, negative ones multiplied (``penalty > 1`` discourages repeats).
    """
    if penalty <= 0:
        raise ValueError("repetition_penalty must be > 0")
    seen = torch.zeros(logits.shape, dtype=torch.long, device=logits.device)
    seen.scatter_reduce_(1, ids, mask.long(), reduce="amax", include_self=True)  # padding never counts
    penalised = torch.where(logits > 0, logits / penalty, logits * penalty)
    return torch.where(seen.bool(), penalised, logits)


def generate_stream(
    model,
    input_ids: torch.Tensor,
    max_new_tokens: int = 32,
    temperature: float = 0.0,
    top_k: int | None = None,
    top_p: float | None = None,
    eos_token_id: Any = _UNSET,
    pad_token_id: int | None = None,
    attention_mask: torch.Tensor | None = None,
    use_cache: bool = True,
    generator: torch.Generator | None = None,
    *,
    min_p: float | None = None,
    repetition_penalty: float = 1.0,
    stop_token_ids: Sequence[int] | None = None,
    stop_sequences: Sequence[Sequence[int]] | None = None,
) -> Iterator[torch.Tensor]:
    """Like :func:`generate`, but yields the newly chosen token of every row (``(B,)``) at each step.

    Arguments are validated immediately (not on the first ``next``). Rows that already finished yield the
    padding id. The stream ends when ``max_new_tokens`` is reached or every row is finished.
    ``generate`` is exactly this stream concatenated onto the prompt.

    Stopping: a row finishes on ``eos_token_id``, on any id in ``stop_token_ids``, or when its generated
    tokens end with one of ``stop_sequences`` (token-id sequences; the matched tokens stay in the output).
    ``repetition_penalty > 1`` discourages tokens already in the prompt or output; ``min_p`` is the
    min-p sampling filter. Sampling options apply to sampling only (``temperature > 0``), except
    ``repetition_penalty``, which also changes greedy decoding.
    """
    cfg = model.config
    eos = cfg.eos_token_id if eos_token_id is _UNSET else eos_token_id
    pad = (
        pad_token_id
        if pad_token_id is not None
        else (cfg.pad_token_id if cfg.pad_token_id is not None else eos)
    )
    if input_ids.dim() != 2:
        raise ValueError("input_ids must be (batch, seq)")
    B, T = input_ids.shape
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be >= 0")
    if T + max_new_tokens > cfg.max_seq_len:
        raise ValueError(
            f"prompt ({T}) + max_new_tokens ({max_new_tokens}) exceeds max_seq_len={cfg.max_seq_len}"
        )
    if repetition_penalty <= 0:
        raise ValueError("repetition_penalty must be > 0")
    stops = {int(t) for t in (stop_token_ids or ())}
    if eos is not None:
        stops.add(int(eos))
    for t in stops:
        if not 0 <= t < cfg.vocab_size:
            raise ValueError(f"stop token id {t} is outside the vocabulary")
    seqs = [list(map(int, q)) for q in (stop_sequences or ())]
    if any(len(q) == 0 for q in seqs):
        raise ValueError("stop_sequences must not contain empty sequences")
    if any(not 0 <= t < cfg.vocab_size for q in seqs for t in q):
        raise ValueError("stop_sequences contain ids outside the vocabulary")
    fill = pad if pad is not None else (min(stops) if stops else 0)
    return _stream(
        model, input_ids, max_new_tokens, temperature, top_k, top_p, min_p, repetition_penalty,
        sorted(stops), seqs, fill, attention_mask, use_cache, generator,
    )  # fmt: skip


def _stream(
    model, input_ids, max_new_tokens, temperature, top_k, top_p, min_p, penalty, stops, seqs, fill,
    attention_mask, use_cache, generator,
) -> Iterator[torch.Tensor]:  # fmt: skip
    B, T = input_ids.shape
    was_training = model.training
    model.eval()
    device = input_ids.device
    stop_t = torch.tensor(stops, dtype=torch.long, device=device)
    seq_t = [torch.tensor(q, dtype=torch.long, device=device) for q in seqs]

    if attention_mask is None:
        mask = torch.ones(B, T, dtype=torch.long, device=device)
        pos = torch.arange(T, device=device).expand(B, T)
        next_pos = torch.full((B,), T, device=device)
    else:
        mask = attention_mask.to(device).long()
        pos = (mask.cumsum(-1) - 1).clamp(min=0)
        next_pos = mask.sum(-1)

    ids = input_ids
    finished = torch.zeros(B, dtype=torch.bool, device=device)
    cache = None
    step_ids, step_pos = ids, pos
    try:
        with torch.no_grad():
            for _ in range(max_new_tokens):
                if use_cache:
                    out = model(
                        step_ids,
                        attention_mask=mask,
                        position_ids=step_pos,
                        past_key_values=cache,
                        use_cache=True,
                    )
                    cache, logits = out.past_key_values, out.logits
                else:
                    full_pos = (mask.cumsum(-1) - 1).clamp(min=0)
                    logits = model(ids, attention_mask=mask, position_ids=full_pos)
                last = logits[:, -1]
                if penalty != 1.0:
                    last = apply_repetition_penalty(last, ids, mask, penalty)
                nxt = sample_next_token(last, temperature, top_k, top_p, generator, min_p)
                if stops or seq_t:
                    nxt = torch.where(finished, torch.full_like(nxt, fill), nxt)
                    if stops:
                        finished = finished | torch.isin(nxt, stop_t)
                ids = torch.cat([ids, nxt[:, None]], dim=1)
                mask = torch.cat([mask, torch.ones(B, 1, dtype=mask.dtype, device=device)], dim=1)
                for q in seq_t:
                    n = q.numel()
                    gen = ids[:, T:]
                    if gen.shape[1] >= n:
                        finished = finished | (gen[:, -n:] == q).all(dim=1)
                step_ids, step_pos = nxt[:, None], next_pos[:, None]
                next_pos = next_pos + 1
                yield nxt
                if (stops or seq_t) and bool(finished.all()):
                    break
    finally:
        model.train(was_training)


def generate(
    model,
    input_ids: torch.Tensor,
    max_new_tokens: int = 32,
    temperature: float = 0.0,
    top_k: int | None = None,
    top_p: float | None = None,
    eos_token_id: Any = _UNSET,
    pad_token_id: int | None = None,
    attention_mask: torch.Tensor | None = None,
    use_cache: bool = True,
    generator: torch.Generator | None = None,
    *,
    min_p: float | None = None,
    repetition_penalty: float = 1.0,
    stop_token_ids: Sequence[int] | None = None,
    stop_sequences: Sequence[Sequence[int]] | None = None,
) -> torch.Tensor:
    """Generate ``max_new_tokens`` tokens after ``input_ids`` (``(B, T)``).

    * ``temperature=0`` (default) is greedy; ``> 0`` samples, optionally with
      ``top_k`` / ``top_p`` / ``min_p``. Pass a seeded ``torch.Generator`` for reproducibility.
    * A sequence is *finished* once it emits ``eos_token_id`` (default: the config's; pass ``None`` to
      disable EOS handling entirely), any of ``stop_token_ids``, or ends with one of ``stop_sequences``;
      afterwards it is filled with ``pad_token_id`` (default: the config's pad id, else the eos id).
      Generation stops early when all rows are done.
    * ``repetition_penalty > 1`` makes tokens already seen less likely (greedy decoding included).
    * ``attention_mask`` (``(B, T)``, 1 = real token) enables **left-padded**
      batches; position ids are derived from it so padding never shifts positions.
    * With ``use_cache=True`` only the newest token is fed through the network each
      step; with ``False`` the whole sequence is recomputed (slower, identical result).

    Returns ``(B, T + n_generated)`` including the prompt. See :func:`generate_stream` for token-by-token output.
    """
    stream = generate_stream(
        model, input_ids, max_new_tokens, temperature, top_k, top_p, eos_token_id, pad_token_id,
        attention_mask, use_cache, generator,
        min_p=min_p, repetition_penalty=repetition_penalty, stop_token_ids=stop_token_ids,
        stop_sequences=stop_sequences,
    )  # fmt: skip
    ids = input_ids
    for tok in stream:
        ids = torch.cat([ids, tok[:, None]], dim=1)
    return ids


def iter_text(tokenizer, token_stream: Iterator[torch.Tensor], row: int = 0) -> Iterator[str]:
    """Turn a :func:`generate_stream` into text pieces for one batch row.

    Decodes the growing token list and yields only the new suffix; a trailing incomplete UTF-8 character
    (``\ufffd``, possible with byte-level BPE) is held back until the next token completes it.
    """
    toks: list[int] = []
    emitted = ""
    for step in token_stream:
        toks.append(int(step[row]))
        text = tokenizer.decode(toks)
        if text.endswith("\ufffd"):
            continue
        if text.startswith(emitted):
            piece = text[len(emitted) :]
            if piece:
                yield piece
            emitted = text
    final = tokenizer.decode(toks)
    if final.startswith(emitted) and len(final) > len(emitted):
        yield final[len(emitted) :]
