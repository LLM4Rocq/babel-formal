#!/bin/bash

#SBATCH --job-name=babel # job's name
#SBATCH --output=babel%j.out # output log (%j = job ID)
#SBATCH --error=babel%j.err # error log (%j = job ID)
#SBATCH --constraint=h100 # h100 gpus
#SBATCH --nodes=1 # number of nodes
#SBATCH --ntasks-per-node=1 # beware, idr_accelerate manage subtasks, don't change this parameter
#SBATCH --gres=gpu:4 # number of gpus/node
#SBATCH --cpus-per-task=96 # cpus/per_tasks
#SBATCH --time=01:30:00 # maximal duration "(HH:MM:SS)"
#SBATCH --qos=qos_gpu_h100-dev # QoS
#SBATCH --hint=nomultithread # no hyperthreading
#SBATCH --account=mmr@h100 # account

module purge
conda deactivate

module load arch/h100
module load miniforge cuda/12.4.1 cudnn/9.7.1.26-cuda nccl/2.25.1-1-cuda

source /lustre/fsn1/worksf/projects/rech/tdm/commun/.opam/opam-init/init.sh
source /lustre/fswork/projects/rech/tdm/commun/venv/crrrocq/bin/activate

TP_SIZE=4
MODEL_PORT=30000
export SCRATCH="/lustre/fsn1/projects/rech/tdm/ulu88xb"
export HF_HOME="$SCRATCH/HF"
export NEMO_HOME="$SCRATCH/NEMO"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd $SCRATCH/babel-formal

pet-server &
python -m sglang.launch_server --tp-size $TP_SIZE --mem-fraction-static 0.9 --model-path /lustre/fsn1/projects/rech/tdm/ulu88xb/babel-formal/babel-translate --host 0.0.0.0 --port $MODEL_PORT &
python -m script.generate_fb_rocq