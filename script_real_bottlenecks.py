#!/usr/bin/env python3

import argparse
import json
import os
import random
import statistics
import time
from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.cuda.nvtx as nvtx
import torch.cuda.profiler as cuda_profiler
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision import models


# =============================================================================
# Utility functions
# =============================================================================

@contextmanager
def nvtx_range(message: str):
    nvtx.range_push(message)
    try:
        yield
    finally:
        nvtx.range_pop()


def percentile(values: List[float], percentile_value: float) -> float:
    if not values:
        return float("nan")

    sorted_values = sorted(values)

    if len(sorted_values) == 1:
        return sorted_values[0]

    position = (len(sorted_values) - 1) * percentile_value / 100.0
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = position - lower

    return (
        sorted_values[lower] * (1.0 - fraction)
        + sorted_values[upper] * fraction
    )


def set_seed(seed: int, rank: int) -> None:
    rank_seed = seed + rank

    random.seed(rank_seed)
    np.random.seed(rank_seed)
    torch.manual_seed(rank_seed)
    torch.cuda.manual_seed_all(rank_seed)


def dataloader_worker_init(worker_id: int) -> None:
    """
    Make worker random streams reproducible but distinct.
    """
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.set_num_threads(1)


def ddp_setup() -> Tuple[int, int, int]:
    init_process_group(backend="nccl")

    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = dist.get_rank()
    world_size = dist.get_world_size()

    torch.cuda.set_device(local_rank)

    return world_size, global_rank, local_rank


def distributed_max(value: float, device: torch.device) -> float:
    tensor = torch.tensor(value, device=device, dtype=torch.float64)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return float(tensor.item())


def distributed_sum(value: float, device: torch.device) -> float:
    tensor = torch.tensor(value, device=device, dtype=torch.float64)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return float(tensor.item())


# =============================================================================
# Synthetic dataset
# =============================================================================

class SyntheticPipelineDataset(Dataset):
    """
    A controlled synthetic dataset.

    Important design choices:

    1. A small sample bank is allocated once and reused.
       We do not generate torch.randn() for every sample because that would
       accidentally introduce a CPU random-number generation bottleneck.

    2. All transforms preserve spatial dimensions.

    3. CPU and GPU transforms are mutually exclusive in the experiment cases.

    4. sleep() represents synthetic acquisition latency, not real filesystem
       bandwidth or image decoding.
    """

    def __init__(
        self,
        length: int,
        scenario: str,
        image_size: int,
        num_classes: int,
        sample_delay: float,
        burst_delay: float,
        burst_interval: int,
        cpu_transform_repeats: int,
        preload_bank_size: int,
        seed: int,
    ) -> None:
        self.length = length
        self.scenario = scenario
        self.image_size = image_size
        self.num_classes = num_classes

        self.sample_delay = sample_delay
        self.burst_delay = burst_delay
        self.burst_interval = burst_interval
        self.cpu_transform_repeats = cpu_transform_repeats

        self.bank_size = min(preload_bank_size, length)

        generator = torch.Generator()
        generator.manual_seed(seed)

        print(
            f"Creating reusable sample bank: "
            f"{self.bank_size} x 3 x {image_size} x {image_size}",
            flush=True,
        )

        # Use a contiguous tensor instead of a Python list of individual tensors.
        self.sample_bank = torch.rand(
            self.bank_size,
            3,
            image_size,
            image_size,
            generator=generator,
            dtype=torch.float32,
        )

        self.label_bank = torch.randint(
            low=0,
            high=num_classes,
            size=(self.bank_size,),
            generator=generator,
            dtype=torch.long,
        )

    def __len__(self) -> int:
        return self.length

    def _apply_latency(self, idx: int) -> None:
        if self.sample_delay > 0:
            time.sleep(self.sample_delay)

        if (
            self.burst_delay > 0
            and self.burst_interval > 0
            and idx % self.burst_interval == 0
        ):
            time.sleep(self.burst_delay)

    @staticmethod
    def _cpu_transform(image: torch.Tensor, repeats: int) -> torch.Tensor:
        """
        Deterministic CPU work that preserves C x H x W.

        avg_pool2d is used repeatedly to create controllable CPU and memory work.
        The operation has a GPU equivalent, allowing CPU-versus-GPU placement
        experiments without changing output size.
        """
        if repeats <= 0:
            return image

        transformed = image.unsqueeze(0)

        for _ in range(repeats):
            transformed = F.avg_pool2d(
                transformed,
                kernel_size=5,
                stride=1,
                padding=2,
            )

            # A pointwise operation stops the experiment from becoming only
            # one repeated pooling kernel.
            transformed = transformed.mul(1.001).clamp_(0.0, 1.0)

        return transformed.squeeze(0)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.scenario in {
            "latency",
            "bursty",
            "mixed",
            "gpu_transform_bursty",
        }:
            self._apply_latency(idx)

        bank_idx = idx % self.bank_size

        # clone() prevents in-place modifications from changing the sample bank.
        # It also gives each sample its own contiguous output storage.
        image = self.sample_bank[bank_idx].clone()
        label = self.label_bank[bank_idx].clone()

        if self.scenario in {"cpu_transform", "mixed"}:
            image = self._cpu_transform(
                image,
                repeats=self.cpu_transform_repeats,
            )

        return image, label


# =============================================================================
# Models
# =============================================================================

class TinyTransferModel(nn.Module):
    """
    Intentionally inexpensive model for transfer-sensitive experiments.

    The input tensor can be large, but GPU compute remains relatively small.
    """

    def __init__(self, num_classes: int) -> None:
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=8,
                kernel_size=1,
                stride=8,
                bias=False,
            ),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Linear(8, num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = self.features(inputs)
        output = torch.flatten(output, 1)
        return self.classifier(output)


def build_model(model_name: str, num_classes: int) -> nn.Module:
    if model_name == "resnet18":
        return models.resnet18(
            weights=None,
            num_classes=num_classes,
        )

    if model_name == "resnet50":
        return models.resnet50(
            weights=None,
            num_classes=num_classes,
        )

    if model_name == "tiny":
        return TinyTransferModel(num_classes=num_classes)

    raise ValueError(f"Unknown model: {model_name}")


# =============================================================================
# DataLoader
# =============================================================================

def prepare_dataloader(
    dataset: Dataset,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    prefetch_factor: int,
    persistent_workers: bool,
    seed: int,
) -> Tuple[DataLoader, DistributedSampler]:
    sampler = DistributedSampler(
        dataset,
        shuffle=True,
        drop_last=True,
        seed=seed,
    )

    generator = torch.Generator()
    generator.manual_seed(seed)

    kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "sampler": sampler,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "drop_last": True,
        "worker_init_fn": dataloader_worker_init,
        "generator": generator,
    }

    # These options are only valid/useful when worker processes exist.
    if num_workers > 0:
        kwargs["persistent_workers"] = persistent_workers
        kwargs["prefetch_factor"] = prefetch_factor

    dataloader = DataLoader(**kwargs)

    return dataloader, sampler


# =============================================================================
# Trainer
# =============================================================================

class Trainer:
    def __init__(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        sampler: DistributedSampler,
        optimizer: torch.optim.Optimizer,
        local_rank: int,
        global_rank: int,
        world_size: int,
        gpu_transform_repeats: int,
        gpu_compute_repeats: int,
    ) -> None:
        self.local_rank = local_rank
        self.global_rank = global_rank
        self.world_size = world_size

        self.device = torch.device("cuda", local_rank)

        self.dataloader = dataloader
        self.sampler = sampler
        self.optimizer = optimizer

        self.gpu_transform_repeats = gpu_transform_repeats
        self.gpu_compute_repeats = gpu_compute_repeats

        model = model.to(self.device)

        self.model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
        )

    @staticmethod
    def _gpu_transform(
        images: torch.Tensor,
        repeats: int,
    ) -> torch.Tensor:
        if repeats <= 0:
            return images

        transformed = images

        for _ in range(repeats):
            transformed = F.avg_pool2d(
                transformed,
                kernel_size=5,
                stride=1,
                padding=2,
            )
            transformed = transformed.mul(1.001).clamp_(0.0, 1.0)

        return transformed

    def _run_batch(
        self,
        source: torch.Tensor,
        targets: torch.Tensor,
    ) -> float:
        with nvtx_range("Optimizer zero_grad"):
            self.optimizer.zero_grad(set_to_none=True)

        accumulated_loss = 0.0

        # Multiple independent forward/backward calculations increase GPU work
        # without changing Dataset or DataLoader cost.
        for repeat in range(self.gpu_compute_repeats):
            with nvtx_range(f"Forward pass repeat {repeat}"):
                output = self.model(source)

            with nvtx_range(f"Loss computation repeat {repeat}"):
                loss = F.cross_entropy(output, targets)
                scaled_loss = loss / self.gpu_compute_repeats

            with nvtx_range(f"Backward pass repeat {repeat}"):
                scaled_loss.backward()

            accumulated_loss += float(loss.detach())

        with nvtx_range("Optimizer step"):
            self.optimizer.step()

        return accumulated_loss / self.gpu_compute_repeats

    def run_epoch(
        self,
        epoch: int,
        measured: bool,
    ) -> Dict[str, float]:
        self.sampler.set_epoch(epoch)

        if measured:
            phase_name = f"Measured epoch {epoch}"
        else:
            phase_name = f"Warmup epoch {epoch}"

        data_wait_times: List[float] = []
        iterator_creation_times: List[float] = []
        losses: List[float] = []

        # Synchronization is only placed at epoch boundaries.
        # Synchronizing every step would destroy normal CUDA overlap.
        torch.cuda.synchronize(self.device)
        dist.barrier()

        epoch_start = time.perf_counter()

        with nvtx_range(phase_name):
            iterator_start = time.perf_counter()

            with nvtx_range(f"{phase_name} - create DataLoader iterator"):
                data_iterator = iter(self.dataloader)

            iterator_creation_times.append(
                time.perf_counter() - iterator_start
            )

            for step in range(len(self.dataloader)):
                with nvtx_range(f"{phase_name} step {step} - total"):
                    data_start = time.perf_counter()

                    with nvtx_range(
                        f"{phase_name} step {step} - DataLoader next"
                    ):
                        source, targets = next(data_iterator)

                    data_wait_times.append(
                        time.perf_counter() - data_start
                    )

                    with nvtx_range(
                        f"{phase_name} step {step} - H2D enqueue"
                    ):
                        source = source.to(
                            self.device,
                            non_blocking=True,
                        )
                        targets = targets.to(
                            self.device,
                            non_blocking=True,
                        )

                    if self.gpu_transform_repeats > 0:
                        with nvtx_range(
                            f"{phase_name} step {step} - GPU transforms"
                        ):
                            source = self._gpu_transform(
                                source,
                                repeats=self.gpu_transform_repeats,
                            )

                    with nvtx_range(
                        f"{phase_name} step {step} - train batch"
                    ):
                        loss = self._run_batch(source, targets)

                    losses.append(loss)

        torch.cuda.synchronize(self.device)
        dist.barrier()

        local_epoch_seconds = time.perf_counter() - epoch_start
        global_epoch_seconds = distributed_max(
            local_epoch_seconds,
            self.device,
        )

        local_samples = (
            len(self.dataloader)
            * self.dataloader.batch_size
        )

        global_samples = local_samples * self.world_size
        global_throughput = global_samples / global_epoch_seconds

        local_data_wait = sum(data_wait_times)
        global_data_wait_sum = distributed_sum(
            local_data_wait,
            self.device,
        )

        global_data_wait_mean_per_rank = (
            global_data_wait_sum / self.world_size
        )

        result = {
            "epoch": epoch,
            "measured": measured,
            "steps_per_rank": len(self.dataloader),
            "samples_global": global_samples,
            "epoch_seconds": global_epoch_seconds,
            "samples_per_second": global_throughput,
            "iterator_creation_seconds_local": sum(
                iterator_creation_times
            ),
            "data_wait_total_seconds_local": local_data_wait,
            "data_wait_mean_seconds_local": (
                statistics.mean(data_wait_times)
                if data_wait_times else float("nan")
            ),
            "data_wait_p50_seconds_local": percentile(
                data_wait_times, 50
            ),
            "data_wait_p95_seconds_local": percentile(
                data_wait_times, 95
            ),
            "data_wait_max_seconds_local": (
                max(data_wait_times)
                if data_wait_times else float("nan")
            ),
            "data_wait_total_seconds_mean_rank":
                global_data_wait_mean_per_rank,
            "loss_mean_local": (
                statistics.mean(losses)
                if losses else float("nan")
            ),
        }

        if self.global_rank == 0:
            phase = "MEASURED" if measured else "WARMUP"

            print(
                f"[{phase}] "
                f"epoch={epoch} "
                f"time={global_epoch_seconds:.4f}s "
                f"throughput={global_throughput:.2f} samples/s "
                f"rank0_data_wait_mean="
                f"{result['data_wait_mean_seconds_local'] * 1000:.3f}ms "
                f"rank0_data_wait_p95="
                f"{result['data_wait_p95_seconds_local'] * 1000:.3f}ms "
                f"rank0_iterator_creation="
                f"{result['iterator_creation_seconds_local']:.4f}s",
                flush=True,
            )

        return result

    def train(
        self,
        warmup_epochs: int,
        measured_epochs: int,
    ) -> List[Dict[str, float]]:
        results: List[Dict[str, float]] = []

        # Warmup is outside the Nsight capture range.
        for epoch in range(warmup_epochs):
            result = self.run_epoch(
                epoch=epoch,
                measured=False,
            )
            results.append(result)

        dist.barrier()
        torch.cuda.synchronize(self.device)

        if self.global_rank == 0:
            print("Starting Nsight CUDA profiler capture", flush=True)

        cuda_profiler.start()

        try:
            with nvtx_range("Entire measured training"):
                for measured_epoch in range(measured_epochs):
                    absolute_epoch = warmup_epochs + measured_epoch

                    result = self.run_epoch(
                        epoch=absolute_epoch,
                        measured=True,
                    )

                    results.append(result)
        finally:
            torch.cuda.synchronize(self.device)
            cuda_profiler.stop()

        if self.global_rank == 0:
            print("Stopped Nsight CUDA profiler capture", flush=True)

        return results


# =============================================================================
# Result reporting
# =============================================================================

def print_summary(
    args: argparse.Namespace,
    results: List[Dict[str, float]],
    rank: int,
) -> None:
    if rank != 0:
        return

    measured = [
        result
        for result in results
        if result["measured"]
    ]

    if not measured:
        print("No measured epochs were recorded.", flush=True)
        return

    throughput_values = [
        result["samples_per_second"]
        for result in measured
    ]

    epoch_times = [
        result["epoch_seconds"]
        for result in measured
    ]

    p95_wait_values_ms = [
        result["data_wait_p95_seconds_local"] * 1000.0
        for result in measured
    ]

    summary = {
        "case_name": args.case_name,
        "scenario": args.scenario,
        "model": args.model,
        "world_size": dist.get_world_size(),
        "workers_per_rank": args.num_workers,
        "workers_per_node": (
            args.num_workers * dist.get_world_size()
        ),
        "pin_memory": bool(args.pin_memory),
        "prefetch_factor": (
            args.prefetch_factor
            if args.num_workers > 0 else None
        ),
        "persistent_workers": (
            bool(args.persistent_workers)
            if args.num_workers > 0 else None
        ),
        "median_samples_per_second": statistics.median(
            throughput_values
        ),
        "mean_samples_per_second": statistics.mean(
            throughput_values
        ),
        "median_epoch_seconds": statistics.median(
            epoch_times
        ),
        "rank0_median_data_wait_p95_ms": statistics.median(
            p95_wait_values_ms
        ),
    }

    print("RESULT_JSON=" + json.dumps(summary, sort_keys=True))
    print(
        "SUMMARY "
        f"case={args.case_name} "
        f"scenario={args.scenario} "
        f"median_throughput="
        f"{summary['median_samples_per_second']:.2f} samples/s "
        f"median_epoch="
        f"{summary['median_epoch_seconds']:.4f}s "
        f"rank0_median_data_wait_p95="
        f"{summary['rank0_median_data_wait_p95_ms']:.3f}ms",
        flush=True,
    )


# =============================================================================
# Main
# =============================================================================

def main(args: argparse.Namespace) -> None:
    world_size, global_rank, local_rank = ddp_setup()

    try:
        set_seed(args.seed, global_rank)

        # Avoid internal CPU thread pools interacting with DataLoader workers.
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)

        if global_rank == 0:
            print("==================================================")
            print(f"Case: {args.case_name}")
            print(f"Scenario: {args.scenario}")
            print(f"Model: {args.model}")
            print(f"World size: {world_size}")
            print(f"Workers/rank: {args.num_workers}")
            print(
                f"Workers/node: {args.num_workers * world_size}"
            )
            print(f"Pin memory: {bool(args.pin_memory)}")
            print(f"Prefetch factor: {args.prefetch_factor}")
            print(
                "Persistent workers: "
                f"{bool(args.persistent_workers)}"
            )
            print(f"Image size: {args.image_size}")
            print(
                f"Batch size/GPU: {args.batch_size_per_gpu}"
            )
            print("==================================================")
            print(f"Arguments: {args}", flush=True)

        dataset = SyntheticPipelineDataset(
            length=args.dataset_length,
            scenario=args.scenario,
            image_size=args.image_size,
            num_classes=args.num_classes,
            sample_delay=args.sample_delay,
            burst_delay=args.burst_delay,
            burst_interval=args.burst_interval,
            cpu_transform_repeats=args.cpu_transform_repeats,
            preload_bank_size=args.preload_bank_size,
            seed=args.seed + global_rank,
        )

        dataloader, sampler = prepare_dataloader(
            dataset=dataset,
            batch_size=args.batch_size_per_gpu,
            num_workers=args.num_workers,
            pin_memory=bool(args.pin_memory),
            prefetch_factor=args.prefetch_factor,
            persistent_workers=bool(args.persistent_workers),
            seed=args.seed,
        )

        model = build_model(
            model_name=args.model,
            num_classes=args.num_classes,
        )

        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.learning_rate,
            momentum=0.9,
        )

        trainer = Trainer(
            model=model,
            dataloader=dataloader,
            sampler=sampler,
            optimizer=optimizer,
            local_rank=local_rank,
            global_rank=global_rank,
            world_size=world_size,
            gpu_transform_repeats=args.gpu_transform_repeats,
            gpu_compute_repeats=args.gpu_compute_repeats,
        )

        results = trainer.train(
            warmup_epochs=args.warmup_epochs,
            measured_epochs=args.epochs,
        )

        print_summary(
            args=args,
            results=results,
            rank=global_rank,
        )

        dist.barrier()

    finally:
        if dist.is_initialized():
            destroy_process_group()


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Controlled DDP input-pipeline bottleneck experiments"
        )
    )

    parser.add_argument(
        "--case-name",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--scenario",
        type=str,
        required=True,
        choices=[
            "gpu_bound",
            "latency",
            "bursty",
            "cpu_transform",
            "gpu_transform",
            "gpu_transform_bursty",
            "transfer",
            "mixed",
        ],
    )

    parser.add_argument(
        "--model",
        type=str,
        default="resnet18",
        choices=[
            "resnet18",
            "resnet50",
            "tiny",
        ],
    )

    parser.add_argument(
        "--dataset-length",
        type=int,
        default=4096,
    )

    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
    )

    parser.add_argument(
        "--num-classes",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--batch-size-per-gpu",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of measured epochs.",
    )

    parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=1,
        help="Warmup epochs executed before Nsight capture.",
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="DataLoader workers per DDP rank.",
    )

    parser.add_argument(
        "--pin-memory",
        type=int,
        choices=[0, 1],
        default=0,
    )

    parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--persistent-workers",
        type=int,
        choices=[0, 1],
        default=1,
    )

    parser.add_argument(
        "--sample-delay",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--burst-delay",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--burst-interval",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--cpu-transform-repeats",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--gpu-transform-repeats",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--gpu-compute-repeats",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--preload-bank-size",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1.0e-3,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=12345,
    )

    args = parser.parse_args()

    if args.dataset_length <= 0:
        parser.error("--dataset-length must be positive")

    if args.batch_size_per_gpu <= 0:
        parser.error("--batch-size-per-gpu must be positive")

    if args.epochs <= 0:
        parser.error("--epochs must be positive")

    if args.warmup_epochs < 0:
        parser.error("--warmup-epochs cannot be negative")

    if args.num_workers < 0:
        parser.error("--num-workers cannot be negative")

    if args.prefetch_factor <= 0:
        parser.error("--prefetch-factor must be positive")

    if args.gpu_compute_repeats <= 0:
        parser.error("--gpu-compute-repeats must be positive")

    if args.preload_bank_size <= 0:
        parser.error("--preload-bank-size must be positive")

    return args


if __name__ == "__main__":
    main(parse_arguments())
