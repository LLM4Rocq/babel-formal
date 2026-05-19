#!/bin/bash

#SBATCH --job-name=babel-bench
#SBATCH --output=babel-bench-%j.out
#SBATCH --error=babel-bench-%j.err
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=96
#SBATCH --time=06:00:00
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --hint=nomultithread
#SBATCH --account=mmr@h100
# Optional array for model sweeps:
# SBATCH --array=0-3

set -euo pipefail

module purge
module load arch/h100
module load miniforge/26.1.1-3 cuda/12.8.0 cudnn/9.7.1.26-cuda nccl/2.25.1-1-cuda

CONDA_ENV="${CONDA_ENV:-nemo}"
conda activate "${CONDA_ENV}"

REPO_ROOT="${REPO_ROOT:-$PWD}"
cd "${REPO_ROOT}"

# Optional singularity/apptainer image (with rocq/lean servers and toolchains).
SIF_IMAGE="${SIF_IMAGE:-}"

# Optional model list file (one model path per line), used with array jobs.
MODEL_LIST_FILE="${MODEL_LIST_FILE:-}"
if [[ -n "${MODEL_LIST_FILE}" ]]; then
  mapfile -t MODELS < <(grep -v '^\s*$' "${MODEL_LIST_FILE}" | grep -v '^\s*#')
  if [[ ${#MODELS[@]} -eq 0 ]]; then
    echo "MODEL_LIST_FILE=${MODEL_LIST_FILE} is empty." >&2
    exit 1
  fi
  MODEL_INDEX="${SLURM_ARRAY_TASK_ID:-0}"
  MODEL_PATH="${MODELS[$MODEL_INDEX]}"
else
  MODEL_PATH="${MODEL_PATH:-}"
fi

if [[ -z "${MODEL_PATH:-}" ]]; then
  echo "Set MODEL_PATH or MODEL_LIST_FILE." >&2
  exit 1
fi

TOKENIZER_PATH="${TOKENIZER_PATH:-}"
DIRECTION="${DIRECTION:-both}"
INPUT_PATH="${INPUT_PATH:-benchmark/prepared/benchmark_pairs.json}"
BENCHMARK_ROOT="${BENCHMARK_ROOT:-new_benchmark}"
OUTPUT_DIR="${OUTPUT_DIR:-export/benchmark_runs}"
RUN_NAME="${RUN_NAME:-jz_${SLURM_JOB_ID}_$(date +%Y%m%d_%H%M%S)}"

PASS_K="${PASS_K:-32}"
MAX_ROUNDS="${MAX_ROUNDS:-2}"
FEEDBACK_LEVEL="${FEEDBACK_LEVEL:-goals}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-0.95}"
MAX_TOKENS="${MAX_TOKENS:-2048}"

TP_SIZE="${TP_SIZE:-4}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-12000}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.98}"

DISABLE_REASONING="${DISABLE_REASONING:-0}"
LEANKIMINA_URL="${LEANKIMINA_URL:-http://127.0.0.1:8000}"
LEANKIMINA_TIMEOUT="${LEANKIMINA_TIMEOUT:-300}"
LEANKIMINA_API_KEY_ENV="${LEANKIMINA_API_KEY_ENV:-LEAN_SERVER_API_KEY}"

ROCQ_BACKEND="${ROCQ_BACKEND:-pytanque}"      # pytanque | ml_server
ROCQ_HOST="${ROCQ_HOST:-127.0.0.1}"
ROCQ_PORT="${ROCQ_PORT:-8765}"
ROCQ_TIMEOUT_PER_STEP="${ROCQ_TIMEOUT_PER_STEP:-10}"
ROCQ_START_SERVER="${ROCQ_START_SERVER:-0}"

CMD=(python -m src.benchmark.run
  --direction "${DIRECTION}"
  --input "${INPUT_PATH}"
  --benchmark-root "${BENCHMARK_ROOT}"
  --model-path "${MODEL_PATH}"
  --output-dir "${OUTPUT_DIR}"
  --run-name "${RUN_NAME}"
  --pass-k "${PASS_K}"
  --max-rounds "${MAX_ROUNDS}"
  --feedback-level "${FEEDBACK_LEVEL}"
  --temperature "${TEMPERATURE}"
  --top-p "${TOP_P}"
  --max-tokens "${MAX_TOKENS}"
  --tensor-parallel-size "${TP_SIZE}"
  --max-model-len "${MAX_MODEL_LEN}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
  --lean-kimina-url "${LEANKIMINA_URL}"
  --lean-kimina-timeout "${LEANKIMINA_TIMEOUT}"
  --lean-kimina-api-key-env "${LEANKIMINA_API_KEY_ENV}"
  --rocq-backend "${ROCQ_BACKEND}"
  --rocq-host "${ROCQ_HOST}"
  --rocq-port "${ROCQ_PORT}"
  --rocq-timeout-per-step "${ROCQ_TIMEOUT_PER_STEP}"
)

if [[ -n "${TOKENIZER_PATH}" ]]; then
  CMD+=(--tokenizer-path "${TOKENIZER_PATH}")
fi
if [[ "${DISABLE_REASONING}" == "1" ]]; then
  CMD+=(--disable-reasoning)
fi
if [[ "${ROCQ_START_SERVER}" == "1" ]]; then
  CMD+=(--rocq-start-server)
fi

set -x
if [[ -n "${SIF_IMAGE}" ]]; then
  RUNTIME_BIN="$(command -v apptainer || command -v singularity || true)"
  if [[ -z "${RUNTIME_BIN}" ]]; then
    echo "SIF_IMAGE is set but neither apptainer nor singularity is available." >&2
    exit 1
  fi
  "${RUNTIME_BIN}" exec --nv -B "${REPO_ROOT}:${REPO_ROOT}" --pwd "${REPO_ROOT}" "${SIF_IMAGE}" "${CMD[@]}"
else
  "${CMD[@]}"
fi
