#!/usr/bin/env bash
# One-command CUDA / pretrained-model check for Oglodon. Run it on a machine with an NVIDIA GPU (a free Colab or
# Kaggle GPU works). It never claims anything it did not run: every result is written to reports/*.json.
#
#   bash scripts/gpu_check.sh                              # Oglodon itself on CUDA
#   bash scripts/gpu_check.sh meta-llama/Llama-3.2-1B      # + a real Hugging Face model (any causal LM)
#   HF_MODEL=./my-local-model bash scripts/gpu_check.sh
#
# Afterwards send reports/ back (or attach it to an issue): the JSON records versions, device, dtype and seed.
set -u
HF_MODEL="${1:-${HF_MODEL:-}}"
mkdir -p reports
pip install -q -e ".[dev]" || exit 2
oglodon info | tee reports/env.json
python - <<'PY' || { echo "No CUDA device: stopping, because nothing here would be a CUDA result."; exit 2; }
import sys, torch
sys.exit(0 if torch.cuda.is_available() else 1)
PY

fail=0
echo "== 1/4 full test-suite (CUDA-dependent tests run now instead of skipping)"
python -m pytest -q 2>&1 | tee reports/pytest.txt | tail -3 || fail=1

echo "== 2/4 oglodon validate on CUDA (includes fp16 mixed precision)"
oglodon validate configs/tiny_gqa.yaml --device cuda --out reports/validate_cuda.json || fail=1

echo "== 3/4 oglodon verify on CUDA (Oglodon model, bfloat16 and float32)"
oglodon verify --config configs/tiny_gqa.yaml --device cuda --out reports/verify_oglodon_cuda.json || fail=1
oglodon verify --config configs/tiny_gqa.yaml --device cuda --dtype bfloat16 --out reports/verify_oglodon_cuda_bf16.json || fail=1

if [ -n "$HF_MODEL" ]; then
  echo "== 4/4 oglodon verify on $HF_MODEL (Llama/Mistral also get the cross-implementation check)"
  # Large models: skip the large-weight copy and cap float64 copies; the checks that need them report 'unsupported'.
  oglodon verify --hf "$HF_MODEL" --device cuda --no-stress --cross-check --out reports/verify_hf.json || fail=1
else
  echo "== 4/4 skipped (give a model id or path as the first argument)"
fi
echo "done. exit status $fail (0 = no failures; skipped/unsupported checks are listed in the reports)"
exit $fail
