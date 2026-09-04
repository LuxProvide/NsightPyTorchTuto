#!/bin/bash -l
#SBATCH --job-name=resnetPipelineExperiments
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --partition=gpu
#SBATCH --time=04:00:00
#SBATCH --hint=nomultithread
#SBATCH --qos=default

set -euo pipefail

module purge
module load env/staging/2023.1
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
module load torchvision/
module load Nsight-Systems/2023.2.1
module load zlib/1.2.13

# One DDP rank per A100.
export NGPUS_PER_NODE=4
# Avoid each DataLoader worker or DDP rank creating a large OpenMP pool.
# DataLoader concurrency is controlled explicitly through --num-workers.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

export NCCL_SOCKET_IFNAME=ib0
export NCCL_ASYNC_ERROR_HANDLING=1

# Reduce accidental CPU thread proliferation in PyTorch.
export TORCH_NUM_THREADS=1

NUM_GPUS=4
CPUS_PER_NODE=64

SCRIPT_NAME="script_real_bottlenecks"
SCRIPT_FILE="${SCRIPT_NAME}.py"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
PROFDIR="${PWD}/${TIMESTAMP}_nsys_${SCRIPT_NAME}"
LOGDIR="${PROFDIR}/logs"

mkdir -p "${PROFDIR}"
mkdir -p "${LOGDIR}"

if [[ ! -f "${SCRIPT_FILE}" ]]; then
    echo "ERROR: Could not find ${SCRIPT_FILE} in ${PWD}" >&2
    exit 1
fi

echo "============================================================"
echo "MeluXina pipeline optimization experiments"
echo "Output directory: ${PROFDIR}"
echo "GPUs: ${NUM_GPUS}"
echo "Physical CPUs requested: ${CPUS_PER_NODE}"
echo "============================================================"

# ---------------------------------------------------------------------------
# Case format
# ---------------------------------------------------------------------------
#
#  1 CASE_NAME
#  2 SCENARIO
#  3 MODEL
#  4 DATASET_LENGTH
#  5 IMAGE_SIZE
#  6 BATCH_SIZE_PER_GPU
#  7 EPOCHS
#  8 WARMUP_EPOCHS
#  9 NUM_WORKERS_PER_RANK
# 10 PIN_MEMORY
# 11 PREFETCH_FACTOR
# 12 PERSISTENT_WORKERS
# 13 SAMPLE_DELAY
# 14 BURST_DELAY
# 15 BURST_INTERVAL
# 16 CPU_TRANSFORM_REPEATS
# 17 GPU_TRANSFORM_REPEATS
# 18 GPU_COMPUTE_REPEATS
# 19 PRELOAD_BANK_SIZE
#
# Important:
# num_workers is PER DDP RANK.
#
# With four ranks:
#     num_workers=4  means 16 workers/node
#     num_workers=8  means 32 workers/node
#     num_workers=16 means 64 workers/node
#
# prefetch_factor is also per worker and per rank.
# Approximate maximum outstanding batches/node:
#
#     world_size * num_workers * prefetch_factor
#
# ---------------------------------------------------------------------------

CASES=(

    # =======================================================================
    # A. GPU-COMPUTE-BOUND CONTROL
    #
    # Status: already completed.
    #
    # Main comparisons:
    #   A01 -> A02: add 4 DataLoader workers/rank
    #   A02 -> A03: enable pin_memory
    #   A03 -> A04: increase prefetch_factor from 2 to 8
    #
    # Existing result:
    # DataLoader optimizations had relatively limited impact compared with
    # the input-starved scenarios. Prefetch factor 8 did not improve
    # throughput over factor 2.
    # =======================================================================

    # "A01_gpu_bound_workers0|gpu_bound|resnet50|4096|224|64|3|1|0|0|2|0|0.0|0.0|0|0|0|3|256"
    # "A02_gpu_bound_workers4|gpu_bound|resnet50|4096|224|64|3|1|4|0|2|1|0.0|0.0|0|0|0|3|256"
    # "A03_gpu_bound_workers4_pin|gpu_bound|resnet50|4096|224|64|3|1|4|1|2|1|0.0|0.0|0|0|0|3|256"
    # "A04_gpu_bound_prefetch8|gpu_bound|resnet50|4096|224|64|3|1|4|1|8|1|0.0|0.0|0|0|0|3|256"


    # =======================================================================
    # B. CONSTANT PER-SAMPLE LATENCY
    #
    # Status:
    # B01-B07 already completed.
    # B08 is a new required correction case.
    #
    # B01-B06:
    # Worker sweep at constant 10 ms per-sample latency.
    #
    # Existing result:
    # 16 workers/rank was the best tested value and was still appreciably
    # faster than 8 workers/rank.
    #
    # B07:
    # Valid pinning comparison at 8 workers:
    #   B05 -> B07
    #
    # B08:
    # New pinning comparison at the best tested worker count:
    #   B06 -> B08
    #
    # Only pin_memory changes between B06 and B08.
    # =======================================================================

    # "B01_latency_workers0|latency|resnet18|4096|224|64|3|1|0|0|2|0|0.010|0.0|0|0|0|1|128"
    # "B02_latency_workers1|latency|resnet18|4096|224|64|3|1|1|0|2|1|0.010|0.0|0|0|0|1|128"
    # "B03_latency_workers2|latency|resnet18|4096|224|64|3|1|2|0|2|1|0.010|0.0|0|0|0|1|128"
    # "B04_latency_workers4|latency|resnet18|4096|224|64|3|1|4|0|2|1|0.010|0.0|0|0|0|1|128"
    # "B05_latency_workers8|latency|resnet18|4096|224|64|3|1|8|0|2|1|0.010|0.0|0|0|0|1|128"
    # "B06_latency_workers16|latency|resnet18|4096|224|64|3|1|16|0|2|1|0.010|0.0|0|0|0|1|128"
    # "B07_latency_workers8_pin|latency|resnet18|4096|224|64|3|1|8|1|2|1|0.010|0.0|0|0|0|1|128"

    # NEW REQUIRED CASE:
    # "B08_latency_workers16_pin|latency|resnet18|4096|224|64|3|1|16|1|2|1|0.010|0.0|0|0|0|1|128"


    # =======================================================================
    # C. BURSTY PER-SAMPLE LATENCY / PREFETCH SWEEP
    #
    # Status: already completed.
    #
    # Main comparisons:
    #   C01 -> C02: prefetch_factor 1 -> 2
    #   C02 -> C03: prefetch_factor 2 -> 4
    #   C03 -> C04: prefetch_factor 4 -> 8
    #
    # Existing result:
    # Factor 1 was insufficient. Factor 2 produced most of the improvement.
    # Factors 4 and 8 provided little or no additional throughput.
    #
    # This group is already methodologically clean because only
    # prefetch_factor changes.
    # =======================================================================

    # "C01_bursty_prefetch1|bursty|resnet18|4096|224|64|4|1|4|1|1|1|0.001|0.100|32|0|0|1|128"
    # "C02_bursty_prefetch2|bursty|resnet18|4096|224|64|4|1|4|1|2|1|0.001|0.100|32|0|0|1|128"
    # "C03_bursty_prefetch4|bursty|resnet18|4096|224|64|4|1|4|1|4|1|0.001|0.100|32|0|0|1|128"
    # "C04_bursty_prefetch8|bursty|resnet18|4096|224|64|4|1|4|1|8|1|0.001|0.100|32|0|0|1|128"


    # =======================================================================
    # D. CPU-TRANSFORM-BOUND INPUT PIPELINE
    #
    # Status:
    # D01-D08 already completed.
    # D09 and D10 are new required correction cases.
    #
    # D01-D06:
    # Worker sweep with CPU transformations.
    #
    # Existing result:
    # 16 workers/rank was the best tested worker count.
    #
    # D07 and D08:
    # Valid comparisons at 8 workers/rank:
    #   D05 -> D07: enable pin_memory
    #   D07 -> D08: prefetch_factor 2 -> 8
    #
    # D09 and D10:
    # Repeat the optimization sequence at 16 workers/rank:
    #   D06 -> D09: enable pin_memory
    #   D09 -> D10: prefetch_factor 2 -> 8
    # =======================================================================

    # "D01_cpu_tf_workers0|cpu_transform|resnet18|4096|224|64|3|1|0|0|2|0|0.0|0.0|0|8|0|1|128"
    # "D02_cpu_tf_workers1|cpu_transform|resnet18|4096|224|64|3|1|1|0|2|1|0.0|0.0|0|8|0|1|128"
    # "D03_cpu_tf_workers2|cpu_transform|resnet18|4096|224|64|3|1|2|0|2|1|0.0|0.0|0|8|0|1|128"
    # "D04_cpu_tf_workers4|cpu_transform|resnet18|4096|224|64|3|1|4|0|2|1|0.0|0.0|0|8|0|1|128"
    # "D05_cpu_tf_workers8|cpu_transform|resnet18|4096|224|64|3|1|8|0|2|1|0.0|0.0|0|8|0|1|128"
    # "D06_cpu_tf_workers16|cpu_transform|resnet18|4096|224|64|3|1|16|0|2|1|0.0|0.0|0|8|0|1|128"
    # "D07_cpu_tf_workers8_pin|cpu_transform|resnet18|4096|224|64|3|1|8|1|2|1|0.0|0.0|0|8|0|1|128"
    # "D08_cpu_tf_workers8_prefetch8|cpu_transform|resnet18|4096|224|64|3|1|8|1|8|1|0.0|0.0|0|8|0|1|128"

    # NEW REQUIRED CASES:
    # "D09_cpu_tf_workers16_pin|cpu_transform|resnet18|4096|224|64|3|1|16|1|2|1|0.0|0.0|0|8|0|1|128"
    # "D10_cpu_tf_workers16_pin_prefetch8|cpu_transform|resnet18|4096|224|64|3|1|16|1|8|1|0.0|0.0|0|8|0|1|128"


    # =======================================================================
    # E. CPU VERSUS GPU TRANSFORM PLACEMENT
    #
    # Status: already completed.
    #
    # Main interpretation:
    # E01 performs synthetic transforms on CPU.
    # E02 moves the synthetic transform work to a relatively free GPU.
    # E03 moves transforms to a GPU that is also executing a heavier model.
    #
    # Important:
    # These are best presented as end-to-end configuration comparisons,
    # rather than strict one-factor-at-a-time comparisons, because the
    # CPU-transform and GPU-transform cases use different worker counts.
    # =======================================================================

    "E01_transform_on_cpu|cpu_transform|resnet18|4096|224|64|3|1|8|1|2|1|0.0|0.0|0|8|0|1|128"
    "E02_transform_on_gpu|gpu_transform|resnet18|4096|224|64|3|1|4|1|2|1|0.0|0.0|0|0|8|1|128"
    "E03_gpu_tf_gpu_busy|gpu_transform|resnet50|4096|224|64|3|1|4|1|2|1|0.0|0.0|0|0|8|3|128"


    # =======================================================================
    # F. H2D-TRANSFER-SENSITIVE WORKLOAD
    #
    # Status: already completed.
    #
    # Main comparisons:
    #   F01 -> F02: add workers
    #   F02 -> F03: enable pin_memory
    #   F03 -> F04: increase prefetch_factor from 2 to 8
    #
    # This group is already methodologically useful:
    #   F02 versus F03 isolates pin_memory.
    #   F03 versus F04 isolates prefetch_factor.
    # =======================================================================

    "F01_h2d_pageable_workers0|transfer|tiny|4096|384|32|4|1|0|0|2|0|0.0|0.0|0|0|0|1|64"
    "F02_h2d_pageable_workers4|transfer|tiny|4096|384|32|4|1|4|0|2|1|0.0|0.0|0|0|0|1|64"
    "F03_h2d_pinned_workers4|transfer|tiny|4096|384|32|4|1|4|1|2|1|0.0|0.0|0|0|0|1|64"
    "F04_h2d_pinned_prefetch8|transfer|tiny|4096|384|32|4|1|4|1|8|1|0.0|0.0|0|0|0|1|64"


    # =======================================================================
    # G. MIXED WORKLOAD
    #
    # Status:
    # G01-G06 already completed.
    # G07-G09 are new required correction cases.
    #
    # Original sequence:
    #   G01: no workers
    #   G02: 4 workers/rank
    #   G03: 8 workers/rank
    #   G04: 8 workers/rank + pin_memory
    #   G05: 8 workers/rank + pin_memory + prefetch factor 4
    #   G06: move synthetic CPU work to GPU
    #
    # Corrected 16-worker sequence:
    #   G03 -> G07: 8 -> 16 workers/rank
    #   G07 -> G08: enable pin_memory
    #   G08 -> G09: prefetch_factor 2 -> 4
    #
    # G07-G09 provide a clean end-to-end sequence based on the best worker
    # count observed in the isolated latency and CPU-transform experiments.
    # =======================================================================

    "G01_mixed_baseline|mixed|resnet18|4096|224|64|3|1|0|0|2|0|0.003|0.080|32|4|0|1|128"
    "G02_mixed_workers4|mixed|resnet18|4096|224|64|3|1|4|0|2|1|0.003|0.080|32|4|0|1|128"
    "G03_mixed_workers8|mixed|resnet18|4096|224|64|3|1|8|0|2|1|0.003|0.080|32|4|0|1|128"
    "G04_mixed_workers8_pin|mixed|resnet18|4096|224|64|3|1|8|1|2|1|0.003|0.080|32|4|0|1|128"
    "G05_mixed_prefetch4|mixed|resnet18|4096|224|64|3|1|8|1|4|1|0.003|0.080|32|4|0|1|128"
    "G06_mixed_gpu_tf|gpu_transform_bursty|resnet18|4096|224|64|3|1|4|1|4|1|0.003|0.080|32|0|4|1|128"

    # NEW REQUIRED CASES:
    # "G07_mixed_workers16|mixed|resnet18|4096|224|64|3|1|16|0|2|1|0.003|0.080|32|4|0|1|128"
    # "G08_mixed_workers16_pin|mixed|resnet18|4096|224|64|3|1|16|1|2|1|0.003|0.080|32|4|0|1|128"
    # "G09_mixed_workers16_pin_prefetch4|mixed|resnet18|4096|224|64|3|1|16|1|4|1|0.003|0.080|32|4|0|1|128"


    # =======================================================================
    # H. PERSISTENT WORKERS WITH SHORT EPOCHS
    #
    # Status: already completed.
    #
    # Main comparison:
    #   H01 -> H02: persistent_workers false -> true
    #
    # This is already a clean matched comparison.
    # =======================================================================

    "H01_short_epochs_nonpersistent|latency|resnet18|2560|224|64|10|0|8|1|2|0|0.002|0.0|0|0|0|1|128"
    "H02_short_epochs_persistent|latency|resnet18|2560|224|64|10|0|8|1|2|1|0.002|0.0|0|0|0|1|128"


    # =======================================================================
    # OPTIONAL EXPLORATORY CASES: WORKER COUNTS BEYOND PHYSICAL CORE COUNT
    #
    # These are disabled by default.
    #
    # MeluXina has 64 physical CPU cores/node under the current allocation.
    #
    # With four DDP ranks:
    #   16 workers/rank = 64 workers/node
    #   24 workers/rank = 96 workers/node
    #   32 workers/rank = 128 workers/node
    #
    # For the latency workload, 24 or 32 workers/rank may still help because
    # workers spend much of their time sleeping.
    #
    # For CPU transformations, oversubscription is more likely to hurt because
    # workers actively use CPU cores and memory bandwidth.
    #
    # Uncomment these only if you want to locate the worker-count plateau.
    # =======================================================================

    # "B09_latency_workers24|latency|resnet18|4096|224|64|3|1|24|0|2|1|0.010|0.0|0|0|0|1|128"
    # "B10_latency_workers32|latency|resnet18|4096|224|64|3|1|32|0|2|1|0.010|0.0|0|0|0|1|128"

    # "D11_cpu_tf_workers24|cpu_transform|resnet18|4096|224|64|3|1|24|0|2|1|0.0|0.0|0|8|0|1|128"

    # "G10_mixed_workers24|mixed|resnet18|4096|224|64|3|1|24|0|2|1|0.003|0.080|32|4|0|1|128"
)



# CASES=(

#     # =======================================================================
#     # A. GPU-COMPUTE-BOUND CONTROL
#     #
#     # Hypothesis:
#     # The GPU is already busy, so DataLoader optimizations should have little
#     # effect. gpu_compute_repeats makes GPU compute dominate.
#     # =======================================================================

#     "A01_gpu_bound_workers0|gpu_bound|resnet50|4096|224|64|3|1|0|0|2|0|0.0|0.0|0|0|0|3|256"
#     "A02_gpu_bound_workers4|gpu_bound|resnet50|4096|224|64|3|1|4|0|2|1|0.0|0.0|0|0|0|3|256"
#     "A03_gpu_bound_workers4_pin|gpu_bound|resnet50|4096|224|64|3|1|4|1|2|1|0.0|0.0|0|0|0|3|256"
#     "A04_gpu_bound_prefetch8|gpu_bound|resnet50|4096|224|64|3|1|4|1|8|1|0.0|0.0|0|0|0|3|256"


#     # =======================================================================
#     # B. CONSTANT PER-SAMPLE LATENCY
#     #
#     # Hypothesis:
#     # More workers hide independent sample latency.
#     # Pinning and very deep prefetching should not remove the latency itself.
#     # =======================================================================

#     "B01_latency_workers0|latency|resnet18|4096|224|64|3|1|0|0|2|0|0.010|0.0|0|0|0|1|128"
#     "B02_latency_workers1|latency|resnet18|4096|224|64|3|1|1|0|2|1|0.010|0.0|0|0|0|1|128"
#     "B03_latency_workers2|latency|resnet18|4096|224|64|3|1|2|0|2|1|0.010|0.0|0|0|0|1|128"
#     "B04_latency_workers4|latency|resnet18|4096|224|64|3|1|4|0|2|1|0.010|0.0|0|0|0|1|128"
#     "B05_latency_workers8|latency|resnet18|4096|224|64|3|1|8|0|2|1|0.010|0.0|0|0|0|1|128"
#     "B06_latency_workers16|latency|resnet18|4096|224|64|3|1|16|0|2|1|0.010|0.0|0|0|0|1|128"
#     "B07_latency_workers16_pin|latency|resnet18|4096|224|64|3|1|16|1|2|1|0.010|0.0|0|0|0|1|128"


#     # =======================================================================
#     # C. BURSTY PER-SAMPLE LATENCY
#     #
#     # Most samples take 1 ms, but every 32nd sample takes an extra 100 ms.
#     #
#     # Hypothesis:
#     # A modest prefetch factor may absorb input jitter.
#     # Increasing it indefinitely should plateau or regress.
#     # =======================================================================

#     "C01_bursty_prefetch1|bursty|resnet18|4096|224|64|4|1|4|1|1|1|0.001|0.100|32|0|0|1|128"
#     "C02_bursty_prefetch2|bursty|resnet18|4096|224|64|4|1|4|1|2|1|0.001|0.100|32|0|0|1|128"
#     "C03_bursty_prefetch4|bursty|resnet18|4096|224|64|4|1|4|1|4|1|0.001|0.100|32|0|0|1|128"
#     "C04_bursty_prefetch8|bursty|resnet18|4096|224|64|4|1|4|1|8|1|0.001|0.100|32|0|0|1|128"


#     # =======================================================================
#     # D. CPU-TRANSFORM-BOUND
#     #
#     # The transform is deterministic and preserves image dimensions.
#     #
#     # Hypothesis:
#     # Workers help until CPU or memory bandwidth saturates.
#     # Pinning and deep prefetching should not increase CPU transform capacity.
#     # =======================================================================

#     "D01_cpu_tf_workers0|cpu_transform|resnet18|4096|224|64|3|1|0|0|2|0|0.0|0.0|0|8|0|1|128"
#     "D02_cpu_tf_workers1|cpu_transform|resnet18|4096|224|64|3|1|1|0|2|1|0.0|0.0|0|8|0|1|128"
#     "D03_cpu_tf_workers2|cpu_transform|resnet18|4096|224|64|3|1|2|0|2|1|0.0|0.0|0|8|0|1|128"
#     "D04_cpu_tf_workers4|cpu_transform|resnet18|4096|224|64|3|1|4|0|2|1|0.0|0.0|0|8|0|1|128"
#     "D05_cpu_tf_workers8|cpu_transform|resnet18|4096|224|64|3|1|8|0|2|1|0.0|0.0|0|8|0|1|128"
#     "D06_cpu_tf_workers16|cpu_transform|resnet18|4096|224|64|3|1|16|0|2|1|0.0|0.0|0|8|0|1|128"
#     "D07_cpu_tf_workers16_pin|cpu_transform|resnet18|4096|224|64|3|1|16|1|2|1|0.0|0.0|0|8|0|1|128"
#     "D08_cpu_tf_workers16_prefetch8|cpu_transform|resnet18|4096|224|64|3|1|16|1|8|1|0.0|0.0|0|8|0|1|128"


#     # =======================================================================
#     # E. TRANSFORM PLACEMENT
#     #
#     # These cases do not apply CPU and GPU transforms simultaneously.
#     # CPU repeats and GPU repeats represent alternative placement.
#     #
#     # Hypothesis:
#     # Moving transforms to the GPU helps only if the CPU is restricting input
#     # delivery and the GPU has sufficient unused capacity.
#     # =======================================================================

#     "E01_transform_on_cpu|cpu_transform|resnet18|4096|224|64|3|1|8|1|2|1|0.0|0.0|0|8|0|1|128"
#     "E02_transform_on_gpu|gpu_transform|resnet18|4096|224|64|3|1|4|1|2|1|0.0|0.0|0|0|8|1|128"
#     "E03_gpu_tf_gpu_busy|gpu_transform|resnet50|4096|224|64|3|1|4|1|2|1|0.0|0.0|0|0|8|3|128"


#     # =======================================================================
#     # F. H2D-TRANSFER-SENSITIVE
#     #
#     # Uses large 384x384 inputs and a deliberately tiny model.
#     #
#     # Hypothesis:
#     # Pinned memory should matter more here than extra dataset workers.
#     # =======================================================================

#     "F01_h2d_pageable_workers0|transfer|tiny|4096|384|32|4|1|0|0|2|0|0.0|0.0|0|0|0|1|64"
#     "F02_h2d_pageable_workers4|transfer|tiny|4096|384|32|4|1|4|0|2|1|0.0|0.0|0|0|0|1|64"
#     "F03_h2d_pinned_workers4|transfer|tiny|4096|384|32|4|1|4|1|2|1|0.0|0.0|0|0|0|1|64"
#     "F04_h2d_pinned_prefetch8|transfer|tiny|4096|384|32|4|1|4|1|8|1|0.0|0.0|0|0|0|1|64"


#     # =======================================================================
#     # G. MIXED WORKLOAD
#     #
#     # Hypothesis:
#     # This requires a combination selected using results from earlier cases.
#     # =======================================================================

#     "G01_mixed_baseline|mixed|resnet18|4096|224|64|3|1|0|0|2|0|0.003|0.080|32|4|0|1|128"
#     "G02_mixed_workers4|mixed|resnet18|4096|224|64|3|1|4|0|2|1|0.003|0.080|32|4|0|1|128"
#     "G03_mixed_workers8|mixed|resnet18|4096|224|64|3|1|8|0|2|1|0.003|0.080|32|4|0|1|128"
#     "G04_mixed_workers8_pin|mixed|resnet18|4096|224|64|3|1|8|1|2|1|0.003|0.080|32|4|0|1|128"
#     "G05_mixed_prefetch4|mixed|resnet18|4096|224|64|3|1|8|1|4|1|0.003|0.080|32|4|0|1|128"
#     "G06_mixed_gpu_tf|gpu_transform_bursty|resnet18|4096|224|64|3|1|4|1|4|1|0.003|0.080|32|0|4|1|128"


#     # =======================================================================
#     # H. PERSISTENT WORKERS
#     #
#     # Many short epochs expose worker teardown/recreation costs.
#     # No warmup epoch is used because epoch-start behavior is the experiment.
#     # =======================================================================

#     "H01_short_epochs_nonpersistent|latency|resnet18|2560|224|64|10|0|8|1|2|0|0.002|0.0|0|0|0|1|128"
#     "H02_short_epochs_persistent|latency|resnet18|2560|224|64|10|0|8|1|2|1|0.002|0.0|0|0|0|1|128"
# )

run_case()
{
    local CASE="$1"

    IFS="|" read -r \
        CASE_NAME \
        SCENARIO \
        MODEL \
        DATASET_LENGTH \
        IMAGE_SIZE \
        BATCH_SIZE_PER_GPU \
        EPOCHS \
        WARMUP_EPOCHS \
        NUM_WORKERS \
        PIN_MEMORY \
        PREFETCH_FACTOR \
        PERSISTENT_WORKERS \
        SAMPLE_DELAY \
        BURST_DELAY \
        BURST_INTERVAL \
        CPU_TRANSFORM_REPEATS \
        GPU_TRANSFORM_REPEATS \
        GPU_COMPUTE_REPEATS \
        PRELOAD_BANK_SIZE \
        <<< "${CASE}"

    local TOTAL_WORKERS=$((NUM_WORKERS * NUM_GPUS))

    local RUN_NAME
    RUN_NAME="${CASE_NAME}"
    RUN_NAME+="_scenario-${SCENARIO}"
    RUN_NAME+="_model-${MODEL}"
    RUN_NAME+="_nw-${NUM_WORKERS}"
    RUN_NAME+="_pin-${PIN_MEMORY}"
    RUN_NAME+="_pf-${PREFETCH_FACTOR}"
    RUN_NAME+="_pw-${PERSISTENT_WORKERS}"

    echo
    echo "============================================================"
    echo "Running: ${RUN_NAME}"
    echo "Workers/rank: ${NUM_WORKERS}"
    echo "Total workers/node: ${TOTAL_WORKERS}"
    echo "Pin memory: ${PIN_MEMORY}"
    echo "Prefetch factor: ${PREFETCH_FACTOR}"
    echo "Persistent workers: ${PERSISTENT_WORKERS}"
    echo "============================================================"

    # --force-overwrite avoids an interactive prompt if a partial result exists.
    # The .nsys-rep suffix is added by nsys.
    srun \
        --nodes=1 \
        --ntasks=1 \
        --ntasks-per-node=1 \
        --cpus-per-task=${CPUS_PER_NODE} \
        --gpus=${NUM_GPUS} \
        --kill-on-bad-exit=1 \
        --cpu-bind=cores \
        nsys profile \
            --force-overwrite=true \
            --output="${PROFDIR}/${RUN_NAME}" \
            --trace=cuda,nvtx,osrt,cublas,cusparse \
            --capture-range=cudaProfilerApi \
            --capture-range-end=stop \
            --gpuctxsw=true \
            --sample=none \
            torchrun \
                --standalone \
                --nnodes=1 \
                --nproc-per-node=${NUM_GPUS} \
                "${SCRIPT_FILE}" \
                --case-name "${CASE_NAME}" \
                --scenario "${SCENARIO}" \
                --model "${MODEL}" \
                --dataset-length "${DATASET_LENGTH}" \
                --image-size "${IMAGE_SIZE}" \
                --batch-size-per-gpu "${BATCH_SIZE_PER_GPU}" \
                --epochs "${EPOCHS}" \
                --warmup-epochs "${WARMUP_EPOCHS}" \
                --num-workers "${NUM_WORKERS}" \
                --pin-memory "${PIN_MEMORY}" \
                --prefetch-factor "${PREFETCH_FACTOR}" \
                --persistent-workers "${PERSISTENT_WORKERS}" \
                --sample-delay "${SAMPLE_DELAY}" \
                --burst-delay "${BURST_DELAY}" \
                --burst-interval "${BURST_INTERVAL}" \
                --cpu-transform-repeats "${CPU_TRANSFORM_REPEATS}" \
                --gpu-transform-repeats "${GPU_TRANSFORM_REPEATS}" \
                --gpu-compute-repeats "${GPU_COMPUTE_REPEATS}" \
                --preload-bank-size "${PRELOAD_BANK_SIZE}" \
                --seed 12345 \
                2>&1 | tee "${LOGDIR}/${RUN_NAME}.log"
}

# ---------------------------------------------------------------------------
# Execute cases in the declared order.
#
# For publication-quality results, run the suite several times with different
# case orders. Fixed ordering is retained here because it is easier to teach
# and inspect in the first tutorial run.
# ---------------------------------------------------------------------------

for CASE in "${CASES[@]}"; do
    run_case "${CASE}"
done

echo
echo "============================================================"
echo "All experiments completed"
echo "Nsight reports: ${PROFDIR}"
echo "Logs: ${LOGDIR}"
echo "============================================================"
