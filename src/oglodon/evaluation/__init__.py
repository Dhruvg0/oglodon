"""Evaluation: perplexity, log-likelihood multiple choice, and generation sanity checks.

Everything works on any causal LM (``oglodon.Oglodon``, a Hugging Face model, or your own module) through
:class:`oglodon.verify.CausalLMAdapter`; only token ids are required.
"""

from .generation import GenerationReport, check_generation
from .multiple_choice import MultipleChoiceResult, continuation_logprob, evaluate_multiple_choice, load_jsonl
from .perplexity import PerplexityResult, perplexity

__all__ = [
    "GenerationReport",
    "MultipleChoiceResult",
    "PerplexityResult",
    "check_generation",
    "continuation_logprob",
    "evaluate_multiple_choice",
    "load_jsonl",
    "perplexity",
]
