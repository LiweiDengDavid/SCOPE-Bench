#!/usr/bin/env bash
# Train the primary SCOPE-Bench baselines with the recorded tuned configs.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

DATASET="${DATASET:-ShortVideoSampled}"
EPOCHS="${EPOCHS:-500}"
GPU="${GPU:-0}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
DRY_RUN="${DRY_RUN:-0}"
DEFAULT_MODELS="BPR LightGCN NCF FlowCF VBPR BM3 DiffMM GRCN REARM FREEDOM MGCN LGMRec LATTICE FITMM"

read -r -a MODEL_LIST <<< "${MODELS:-${DEFAULT_MODELS}}"

if [[ ! -f "configs/datasets/${DATASET}.yaml" ]]; then
  echo "Missing dataset config: configs/datasets/${DATASET}.yaml" >&2
  exit 1
fi

# Fail early when a requested model does not have a tuned override for this bundle.
"${PYTHON_BIN}" - "${DATASET}" "${MODEL_LIST[@]}" <<'PY'
import sys
from pathlib import Path

import yaml

dataset, *models = sys.argv[1:]
config_path = Path("configs/datasets") / f"{dataset}.yaml"
config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
tuned = config.get("model_overrides", {})
missing = [model for model in models if model not in tuned]
if missing:
    raise SystemExit(
        f"No tuned {dataset} config for: {', '.join(missing)}. "
        f"Available: {', '.join(tuned)}"
    )
PY

"${PYTHON_BIN}" scripts/validate_short_video_bundle.py --datasets "${DATASET}"

echo "Dataset: ${DATASET}"
echo "Epochs:  ${EPOCHS}"
echo "GPU:     ${GPU}"
echo "Models:  ${MODEL_LIST[*]}"

for model in "${MODEL_LIST[@]}"; do
  model_tag="$(printf '%s' "${model}" | tr '[:upper:]' '[:lower:]')"
  echo
  echo "Starting ${model} with configs/datasets/${DATASET}.yaml"
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "CUDA_VISIBLE_DEVICES=${GPU} ${PYTHON_BIN} main.py --model ${model} --dataset ${DATASET} --max_epochs ${EPOCHS} --gpu_id 0 --type benchmark --comment scope_${model_tag} --hyper_parameters []"
    continue
  fi
  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" main.py \
    --model "${model}" \
    --dataset "${DATASET}" \
    --max_epochs "${EPOCHS}" \
    --gpu_id 0 \
    --type benchmark \
    --comment "scope_${model_tag}" \
    --hyper_parameters '[]'
done

echo "All requested baselines finished."
