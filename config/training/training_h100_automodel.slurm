#!/bin/bash

#SBATCH --job-name=babel-am
#SBATCH --output=babel-am%j.out
#SBATCH --error=babel-am%j.err
#SBATCH --constraint=h100
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=96
#SBATCH --time=01:30:00
#SBATCH --qos=qos_gpu_h100-dev
#SBATCH --hint=nomultithread
#SBATCH --account=mmr@h100
# Optional sweep mode:
# SBATCH --array=0-4

set -euo pipefail

module purge
conda deactivate

module load arch/h100
module load miniforge cuda/12.4.1 cudnn/9.7.1.26-cuda nccl/2.25.1-1-cuda

# Keep your current env name if it already contains nemo-automodel.
conda activate nemo

export SCRATCH="/lustre/fsn1/projects/rech/tdm/ulu88xb"
export HF_HOME="$SCRATCH/HF"
export NEMO_HOME="$SCRATCH/NEMO"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd "$SCRATCH/new-babel"

# Qwen 3.5 family sweep.
MODELS=(
  "Qwen/Qwen3.5-0.8B"
  "Qwen/Qwen3.5-2B"
  "Qwen/Qwen3.5-4B"
  "Qwen/Qwen3.5-9B"
  "Qwen/Qwen3.5-27B"
)

MODEL_INDEX="${SLURM_ARRAY_TASK_ID:-0}"
MODEL_NAME="${MODEL_OVERRIDE:-${MODELS[$MODEL_INDEX]}}"

if [[ -z "${MODEL_NAME:-}" ]]; then
  echo "No model selected. Set MODEL_OVERRIDE or SLURM_ARRAY_TASK_ID." >&2
  exit 1
fi

MODEL_TAG=$(echo "$MODEL_NAME" | tr '/.' '__' | tr -c '[:alnum:]_-' '_')

# Default recipe parameters; override any of these with sbatch --export or env vars.
export MODEL_NAME
export BABEL_DATASET_PATH="${BABEL_DATASET_PATH:-training_set/lean_rocq.json}"
export TRAIN_SPLIT="${TRAIN_SPLIT:-train}"
export CHECKPOINT_DIR="${CHECKPOINT_DIR:-$SCRATCH/NEMO/checkpoints/babel_automodel_${MODEL_TAG}}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-32}"
export LOCAL_BATCH_SIZE="${LOCAL_BATCH_SIZE:-1}"
export MAX_STEPS="${MAX_STEPS:-320}"
export SEQ_LENGTH="${SEQ_LENGTH:-32768}"
export NUM_WORKERS="${NUM_WORKERS:-8}"

# Legacy-like parallel defaults from your previous NeMo recipe:
# TP=4, PP=2, CP=2 on 32 GPUs -> DP=2.
export TP_SIZE="${TP_SIZE:-4}"
export CP_SIZE="${CP_SIZE:-2}"
export PP_SIZE="${PP_SIZE:-2}"
export SEQUENCE_PARALLEL="${SEQUENCE_PARALLEL:-true}"
export PP_MICROBATCH_SIZE="${PP_MICROBATCH_SIZE:-1}"
export ACTIVATION_CHECKPOINTING="${ACTIVATION_CHECKPOINTING:-true}"

echo "Launching model: $MODEL_NAME"
echo "Checkpoint dir: $CHECKPOINT_DIR"

head_node=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)

srun torchrun \
  --nproc-per-node=4 \
  --nnodes="${SLURM_NNODES}" \
  --rdzv-backend=c10d \
  --rdzv-endpoint="${head_node}:29500" \
  --rdzv-id="${SLURM_JOB_ID:-babel_automodel}" \
  -m nemo_automodel.cli.app config/training/automodel_qwen35.yaml
