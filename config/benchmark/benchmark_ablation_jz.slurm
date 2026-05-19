#!/bin/bash

#SBATCH --job-name=babel-ablation
#SBATCH --output=babel-ablation-%A_%a.out
#SBATCH --error=babel-ablation-%A_%a.err
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=96
#SBATCH --time=06:00:00
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --hint=nomultithread
#SBATCH --account=mmr@h100
# REQUIRED: set array range to match MODEL_MATRIX_FILE
# SBATCH --array=0-5

set -euo pipefail

module purge
module load arch/h100
module load miniforge/26.1.1-3 cuda/12.8.0 cudnn/9.7.1.26-cuda nccl/2.25.1-1-cuda

CONDA_ENV="${CONDA_ENV:-nemo}"
conda activate "${CONDA_ENV}"

REPO_ROOT="${REPO_ROOT:-$PWD}"
cd "${REPO_ROOT}"

MODEL_MATRIX_FILE="${MODEL_MATRIX_FILE:-config/benchmark/model_matrix.txt}"
if [[ ! -f "${MODEL_MATRIX_FILE}" ]]; then
  echo "Missing MODEL_MATRIX_FILE=${MODEL_MATRIX_FILE}" >&2
  exit 1
fi

mapfile -t MATRIX < <(grep -v '^\s*$' "${MODEL_MATRIX_FILE}" | grep -v '^\s*#')
if [[ ${#MATRIX[@]} -eq 0 ]]; then
  echo "MODEL_MATRIX_FILE is empty: ${MODEL_MATRIX_FILE}" >&2
  exit 1
fi

IDX="${SLURM_ARRAY_TASK_ID:-0}"
if (( IDX < 0 || IDX >= ${#MATRIX[@]} )); then
  echo "SLURM_ARRAY_TASK_ID=${IDX} out of bounds [0, $(( ${#MATRIX[@]} - 1 ))]" >&2
  exit 1
fi

IFS='|' read -r MODEL_LABEL MODEL_PATH DISABLE_REASONING <<< "${MATRIX[$IDX]}"
MODEL_LABEL="${MODEL_LABEL:-model_${IDX}}"
DISABLE_REASONING="${DISABLE_REASONING:-0}"

if [[ -z "${MODEL_PATH:-}" ]]; then
  echo "Invalid matrix row (missing model path): ${MATRIX[$IDX]}" >&2
  exit 1
fi

DIRECTION="${DIRECTION:-both}"
INPUT_PATH="${INPUT_PATH:-benchmark/prepared/benchmark_pairs.json}"
BENCHMARK_ROOT="${BENCHMARK_ROOT:-new_benchmark}"
OUTPUT_DIR="${OUTPUT_DIR:-export/benchmark_runs_ablation}"
RUN_NAME="${RUN_NAME:-abl_${MODEL_LABEL}_${SLURM_JOB_ID}_${IDX}_$(date +%Y%m%d_%H%M%S)}"

PASS_K="${PASS_K:-32}"
MAX_ROUNDS="${MAX_ROUNDS:-2}"
FEEDBACK_LEVEL="${FEEDBACK_LEVEL:-goals}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-0.95}"
MAX_TOKENS="${MAX_TOKENS:-2048}"
TP_SIZE="${TP_SIZE:-4}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-12000}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.98}"

LEANKIMINA_URL="${LEANKIMINA_URL:-http://127.0.0.1:8000}"
LEANKIMINA_TIMEOUT="${LEANKIMINA_TIMEOUT:-300}"
LEANKIMINA_API_KEY_ENV="${LEANKIMINA_API_KEY_ENV:-LEAN_SERVER_API_KEY}"

ROCQ_BACKEND="${ROCQ_BACKEND:-pytanque}"
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

if [[ "${DISABLE_REASONING}" == "1" ]]; then
  CMD+=(--disable-reasoning)
fi
if [[ "${ROCQ_START_SERVER}" == "1" ]]; then
  CMD+=(--rocq-start-server)
fi

set -x
"${CMD[@]}"
