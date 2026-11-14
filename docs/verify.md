# `oglodon verify`: behavioural verification of any causal language model

`oglodon.validation` compares Oglodon against an independent float64 reference, so it needs to know Oglodon's internals.
`oglodon.verify` is the model-agnostic counterpart: it needs only a function from token ids to logits, and checks
**properties a correct causal LM must have**. This is metamorphic testing: you cannot know the right logits, but
you can know how they must *relate* when you change the input in a controlled way.

## The contract

A model is "a causal LM" if `model(input_ids)` with `input_ids` of shape `(B, T)` returns logits `(B, T, V)`, as a
tensor, an object with `.logits`, a dict with `"logits"`, or a tuple whose first item is the logits. The adapter
(`CausalLMAdapter`) reads the `forward` signature to learn whether the model also takes `attention_mask`,
`position_ids`, `labels`, and `use_cache` / `past_key_values`. Checks that need a capability the model lacks are
reported **skipped**, never passed.

## Checks

| name | invariant |
|---|---|
| structure.shape_and_vocab | `(B, T, V)`, `V` equals the declared vocabulary, finite for ids `0` and `V-1` and length-1 input |
| structure.determinism | two eval passes are identical |
| structure.state_dict_roundtrip | loading the state dict into a re-initialised copy reproduces the logits |
| structure.max_length | finite logits at the model's maximum length |
| causality.future_independence | logits before position *t* do not change when token *t* changes; logits from *t* on **do** change (otherwise the test would pass vacuously and the check fails) |
| causality.prefix_consistency | logits of a prefix are the head of the logits of the full sequence |
| causality.padding_invariance | left padding with a mask and mask-derived positions leaves real-token logits unchanged |
| causality.mask_is_used | the contents of masked-out slots have no effect |
| batching.sample_independence | row *i* of a batch equals row *i* run alone |
| batching.order_independence | permuting the batch permutes the logits |
| cache.single_token_steps | prefill + one-token cached steps equal the full forward |
| cache.chunked_steps | prefill + multi-token cached chunks equal the full forward (tests the cache offset in the mask) |
| numerics.float64_consistency | a float64 copy agrees with native precision |
| numerics.bfloat16_sanity | bfloat16 copy is finite and within 25% of the logit scale (WARNING beyond that) |
| numerics.large_activation_stability | 3x-scaled weights still give finite logits |
| gradients.finite_and_flowing | finite gradients; tensors with none/zero gradient are listed as a WARNING |
| gradients.finite_difference | autograd equals central finite differences in float64 on up to 8 parameter tensors |
| gradients.loss_convention | the model's own loss equals shifted next-token cross-entropy; unshifted loss is reported as a failure |

## Why two weight sets

Each check runs on the weights as given and on a stress copy whose matrices are redrawn from `N(0, 0.2)`
(1-D parameters such as norm gains are kept). With the usual 0.02 initialisation attention is almost uniform and
a leaking mask moves logits by less than a tolerance. The stress copy makes the bug loud. It says nothing about
trained behaviour and the report labels which weights each message refers to.

## Reading the results

* `fail`: an invariant is broken, or the check crashed.
* `warning`: suspicious, not provably wrong (dead parameters, fragile bfloat16).
* `skipped` / `unsupported`: not run (missing capability / environment cannot run it). Never counted as passed.
* `report.passed` means no failure and at least one pass; `report.complete` means nothing was skipped.

## Limits (be aware)

* Invariants are necessary, not sufficient: a model can satisfy all of them and still be wrong (for example a
  consistent but incorrect attention scale). Two things close that gap: `oglodon.validation` (independent float64
  reference, Oglodon models) and `verify_conversion` / `oglodon verify --cross-check` (a second implementation with identical
  weights; Hugging Face Llama, Mistral and Qwen2 are supported through `oglodon.convert.from_hf_llama`). A wrong RoPE
  re-ordering or a wrong softmax scale is caught by the cross-check (see `tests/test_convert_equivalence.py`).
* Trained, not downloaded: the tests also verify models trained for a few hundred steps in the test run (loss must
  fall by more than 1 nat), because trained weights behave differently from random ones. One pretrained
  checkpoint was run: `HuggingFaceTB/SmolLM2-135M` (see [Interoperability](interoperability.md)); no other.
* Padding invariance needs both `attention_mask` and `position_ids`; models that derive positions inside
  `generate()` only are reported skipped for that check.
* The finite-difference check tries three step sizes because many libraries compute part of their maths (for
  example the softmax) in float32 even for float64 models; the report says when the result is limited by that.
* Tested on CPU only, on Oglodon, a from-scratch PyTorch model, and Hugging Face GPT-2, Llama, Mistral, Qwen2, Gemma, Phi,
  GPT-NeoX, OPT, Falcon, Bloom and Mixtral with random weights. One pretrained checkpoint (SmolLM2-135M, CPU, float32) was run, and
  `verify` on an Oglodon model was run on CUDA. Other pretrained checkpoints, Mamba/state-space and encoder-decoder models,
  and `verify` on a non-Oglodon model on CUDA were not run. Bloom (ALiBi, no `position_ids`) skips the padding check; Mixtral's
  float64 checks are `unsupported` where its expert kernels reject float64.
