# Contributing

```bash
git clone <repo> && cd Oglodon
pip install -e ".[dev]"
ruff check . && ruff format --check . && mypy src && pytest
```

`pytest` skips tests marked `slow` (they compile with `torch.compile` or launch `torchrun`); run them with
`pytest -m slow`.

## Rules

* Keep core layers readable; add a test for every behaviour change.
* **Never change default behaviour silently.** New options default to the old behaviour and are tested to leave outputs
  unchanged.
* Prefer fixing a real bug over adding a feature, and write the regression test first (see `tests/test_audit_fixes.py`).
* Do not claim something was verified unless a test or run shows it; mark CUDA/GPU paths unverified until run.
* Do not fabricate benchmarks. Numbers in docs come from a command you can run.
* PyTorch stays the core dependency; everything else is an optional extra.

## Adding a model option (checklist)

1. Add the field to `OglodonConfig` (validation, docstring, default that keeps old behaviour).
2. Implement it in the layer or model.
3. Teach the independent float64 reference (`oglodon/validation/reference.py`) and the parameter-count check, so
   `oglodon validate` can check it: never loosen a tolerance to make a check pass.
4. Add the variant to `tests/test_variants.py` (validation and `verify`, eager and SDPA).
5. If it should be convertible, extend `oglodon/convert/hf.py` and the equivalence tests, or make the converter refuse it.
6. Document it in [configuration](configuration.md) and the changelog.

## Tests worth copying

* Compare against an independent naive implementation (`tests/test_evaluation.py`).
* Plant a bug and assert the checker catches it (`tests/test_validation.py`, `tests/test_verify.py`).
* Mutation-check a test: break the code on purpose and confirm the test fails (done for the data-parallel weighting).

## Releasing

See [Publishing](publishing.md). Publishing to PyPI is a deliberate maintainer step.
