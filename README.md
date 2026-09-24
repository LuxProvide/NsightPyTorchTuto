# Nsight Systems PyTorch Profiling Example

This repository contains the source code used in the LuxProvide tutorial:

**PyTorch profiling with NVIDIA Nsight Systems**

The example demonstrates how to profile distributed PyTorch training with NVIDIA Nsight Systems using:

- PyTorch Distributed Data Parallel (DDP)
- ResNet18 training on 4 GPUs
- NVTX annotations for timeline analysis
- Synthetic data generation
- Slurm job submission with `torchrun` and `nsys profile`

## Repository contents

- `resnet_profile.py` – Distributed ResNet18 training script instrumented with NVTX ranges.
- `nsys_profile.sh` – Example Slurm submission script that launches the workload under NVIDIA Nsight Systems.

## Purpose

This code is intentionally simplified and is provided as a profiling playground rather than a realistic machine learning benchmark. It is designed to generate clear Nsight Systems timelines that can be used to study:

- GPU utilization
- GPU idle gaps (starvation)
- Host-to-device transfers
- DataLoader behavior
- DDP synchronization
- CPU and GPU overlap

## Running the example

Submit the profiling job with:

```bash
sbatch --account=p2xxxxx nsys_profile.sh
