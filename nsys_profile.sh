#!/bin/bash -l
#SBATCH --job-name=resnet_nsys
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus-per-node=4
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --partition=gpu
#SBATCH --time=00:30:00
#SBATCH --hint=nomultithread
#SBATCH --qos=short

set -euo pipefail

module purge
module load env/staging/2023.1
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
module load torchvision/
module load Nsight-Systems/2023.2.1
module load zlib/1.2.13

export NCCL_SOCKET_IFNAME=ib0
export NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=1

NUM_GPUS=4
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
PROFDIR="${PWD}/${TIMESTAMP}_nsys_resnet_output"
mkdir -p "${PROFDIR}"

srun \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=64 \
    --gpus=${NUM_GPUS} \
    --cpu-bind=cores \
    --kill-on-bad-exit=1 \
    nsys profile \
        --gpu-metrics-device=all \
        --force-overwrite=true \
        --output="${PROFDIR}/resnet_profile" \
        --trace=cuda,nvtx,osrt,cublas,cusparse \
        --capture-range=cudaProfilerApi \
        --capture-range-end=stop \
        --gpuctxsw=true \
        torchrun \
            --standalone \
            --nnodes=1 \
            --nproc-per-node=${NUM_GPUS} \
            resnet_profile.py \
                --warmup-epochs 2 \
                --epochs 3 \
                --batch-size-per-gpu 64 \
                --dataset-length 4096 \
                --num-workers 4 \
                --pin-memory 1 \
                --prefetch-factor 2 \
                --persistent-workers 1
