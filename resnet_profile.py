import argparse
import os
from contextlib import contextmanager

import torch
import torch.cuda.profiler as cuda_profiler
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision.models import resnet18


@contextmanager
def nvtx_range(name):
    """Add a named region to the Nsight Systems timeline."""
    torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()


class SyntheticDataset(Dataset):
    """Return synthetic images without reading files from storage."""

    def __init__(self, length, image_size=224, num_classes=1000):
        self.length = length

        # Reuse one image to keep the example simple.
        self.image = torch.rand(3, image_size, image_size)
        self.num_classes = num_classes

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        label = index % self.num_classes
        return self.image, label


def train_epoch(model, dataloader, optimizer, device, epoch):
    """Run one training epoch."""

    model.train()
    dataloader.sampler.set_epoch(epoch)

    with nvtx_range(f"epoch_{epoch}"):
        for step, (images, labels) in enumerate(dataloader):
            with nvtx_range("data_to_gpu"):
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

            with nvtx_range("forward"):
                predictions = model(images)
                loss = F.cross_entropy(predictions, labels)

            with nvtx_range("backward"):
                optimizer.zero_grad(set_to_none=True)
                loss.backward()

            with nvtx_range("optimizer"):
                optimizer.step()

            if dist.get_rank() == 0 and step % 10 == 0:
                print(
                    f"epoch={epoch} step={step} "
                    f"loss={loss.item():.4f}",
                    flush=True,
                )


def main(args):
    # torchrun provides LOCAL_RANK, RANK and WORLD_SIZE.
    local_rank = int(os.environ["LOCAL_RANK"])

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device("cuda", local_rank)

    try:
        if rank == 0:
            print(
                f"Starting training with {world_size} GPUs",
                flush=True,
            )

        dataset = SyntheticDataset(
            length=args.dataset_length,
        )

        sampler = DistributedSampler(
            dataset,
            shuffle=True,
        )

        dataloader_options = {
            "dataset": dataset,
            "batch_size": args.batch_size_per_gpu,
            "sampler": sampler,
            "num_workers": args.num_workers,
            "pin_memory": bool(args.pin_memory),
            "drop_last": True,
        }

        # These arguments are only valid when workers are enabled.
        if args.num_workers > 0:
            dataloader_options["prefetch_factor"] = (
                args.prefetch_factor
            )
            dataloader_options["persistent_workers"] = bool(
                args.persistent_workers
            )

        dataloader = DataLoader(**dataloader_options)

        model = resnet18(weights=None).to(device)

        model = DDP(
            model,
            device_ids=[local_rank],
        )

        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=0.001,
            momentum=0.9,
        )
        # Warmup is not recorded by Nsight Systems.
        for epoch in range(args.warmup_epochs):
            if rank == 0:
                print(f"Warmup epoch {epoch}", flush=True)

            train_epoch(
                model,
                dataloader,
                optimizer,
                device,
                epoch,
            )
        # Wait until every rank has completed warmup.
        dist.barrier()
        torch.cuda.synchronize(device)

        if rank == 0:
            print("Starting Nsight Systems capture", flush=True)

        cuda_profiler.start()

        with nvtx_range("profiled_training"):
            for epoch in range(
                args.warmup_epochs,
                args.warmup_epochs + args.epochs,
            ):
                train_epoch(
                    model,
                    dataloader,
                    optimizer,
                    device,
                    epoch,
                )

        # Make sure all CUDA work is complete before ending capture.
        dist.barrier()
        torch.cuda.synchronize(device)
        cuda_profiler.stop()

        if rank == 0:
            print("Nsight Systems capture completed", flush=True)

    finally:
        dist.destroy_process_group()


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Profile distributed ResNet18 training"
    )

    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument(
        "--batch-size-per-gpu",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--dataset-length",
        type=int,
        default=4096,
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--pin-memory",
        type=int,
        choices=[0, 1],
        default=1,
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

    return parser.parse_args()


if __name__ == "__main__":
    main(parse_arguments())
