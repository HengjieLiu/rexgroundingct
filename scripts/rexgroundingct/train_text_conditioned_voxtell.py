#!/usr/bin/env python3
"""Experimental challenge-valid text-conditioned VoxTell fine-tuning.

The default mode does not start training until the caller passes
--allow-experimental-train. This is deliberate because public VoxTell exposes an
nnU-Net encoder-transfer fine-tuning CLI, while this script trains the actual
text-conditioned VoxTell model on finding prompts.

Reason: the public VoxTell repository exposes an nnU-Net encoder-transfer
fine-tuning CLI, but the ReXGroundingCT challenge requires free-text conditioned
segmentation.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from nnunetv2.imageio.nibabel_reader_writer import NibabelIOWithReorient
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from voxtell.inference.predictor import VoxTellPredictor

from common import (
    CT_ROOT,
    EXP_ROOT,
    REX_METADATA,
    REX_SEG_DIR,
    command_string,
    ct_rate_abs_path,
    load_split_entries,
    sorted_prompts,
    write_json,
)
from poll_ct_subset import snapshot as ct_snapshot


PATCH_SIZE = (192, 192, 192)


def setup_distributed(args: argparse.Namespace) -> tuple[bool, int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = args.distributed or world_size > 1
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(args.local_rank if args.local_rank is not None else args.gpu)))
    if distributed:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        if not dist.is_initialized():
            dist.init_process_group(backend=backend)
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    return distributed, rank, world_size, local_rank


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank: int) -> bool:
    return rank == 0


def normalize_patch(patch: np.ndarray) -> np.ndarray:
    foreground = patch[np.abs(patch) > 0]
    if foreground.size == 0:
        foreground = patch.reshape(-1)
    mean = float(foreground.mean())
    std = float(foreground.std())
    if std < 1e-6:
        std = 1.0
    return ((patch - mean) / std).astype(np.float32, copy=False)


def patch_bounds(mask_4d: np.ndarray, spatial_shape: tuple[int, int, int]) -> tuple[list[int], list[int]]:
    union = mask_4d > 0
    if union.any():
        coords = np.argwhere(union)
        center = coords[:, 1:].mean(axis=0)
    else:
        center = np.asarray(spatial_shape, dtype=np.float32) / 2

    starts: list[int] = []
    ends: list[int] = []
    for dim, patch, c in zip(spatial_shape, PATCH_SIZE, center):
        if dim <= patch:
            start = 0
            end = dim
        else:
            start = int(round(float(c) - patch / 2))
            start = max(0, min(start, dim - patch))
            end = start + patch
        starts.append(start)
        ends.append(end)
    return starts, ends


def crop_or_pad(image: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    spatial_shape = tuple(int(x) for x in image.shape[1:])
    starts, ends = patch_bounds(target, spatial_shape)
    image_crop = image[
        :,
        starts[0]:ends[0],
        starts[1]:ends[1],
        starts[2]:ends[2],
    ]
    target_crop = target[
        :,
        starts[0]:ends[0],
        starts[1]:ends[1],
        starts[2]:ends[2],
    ]

    image_patch = np.zeros((image.shape[0], *PATCH_SIZE), dtype=np.float32)
    target_patch = np.zeros((target.shape[0], *PATCH_SIZE), dtype=np.float32)
    insert = tuple(slice(0, size) for size in image_crop.shape[1:])
    image_patch[(slice(None), *insert)] = image_crop
    target_patch[(slice(None), *insert)] = target_crop
    return normalize_patch(image_patch), (target_patch > 0).astype(np.float32)


def dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    dims = tuple(range(2, probs.ndim))
    intersection = (probs * target).sum(dim=dims)
    denominator = probs.sum(dim=dims) + target.sum(dim=dims)
    dice = (2 * intersection + eps) / (denominator + eps)
    return 1 - dice.mean()


def load_case(entry: dict, reader: NibabelIOWithReorient, ct_root: Path, seg_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    name = entry["name"]
    ct_path = ct_rate_abs_path(name, ct_root)
    gt_path = seg_dir / name
    if not ct_path.is_file():
        raise FileNotFoundError(f"Missing CT: {ct_path}")
    if not gt_path.is_file():
        raise FileNotFoundError(f"Missing segmentation: {gt_path}")
    image, _props = reader.read_images([str(ct_path)])
    target = np.asanyarray(nib.load(str(gt_path)).dataobj).astype(np.float32, copy=False)
    if tuple(image.shape[1:]) != tuple(target.shape[1:]):
        raise ValueError(f"{name}: image shape {image.shape[1:]} != target shape {target.shape[1:]}")
    return crop_or_pad(image.astype(np.float32, copy=False), target)


def save_checkpoint(path: Path, network: torch.nn.Module, optimizer: torch.optim.Optimizer, iteration: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    model = network.module if hasattr(network, "module") else network
    torch.save(
        {
            "iteration": iteration,
            "network_weights": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        path,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("/workspace/configs/experiments/002_voxtell_text_ft_miccai_train_val.json"))
    parser.add_argument("--exp-dir", type=Path, default=EXP_ROOT / "002_voxtell_text_ft_miccai_train_val")
    parser.add_argument("--metadata", type=Path, default=REX_METADATA)
    parser.add_argument("--ct-root", type=Path, default=CT_ROOT)
    parser.add_argument("--seg-dir", type=Path, default=REX_SEG_DIR)
    parser.add_argument("--model-dir", type=Path, default=None)
    parser.add_argument("--embeddings", type=Path, default=None)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--local-rank", type=int, default=None)
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--max-iterations", type=int, default=None)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--allow-experimental-train", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    distributed, rank, world_size, local_rank = setup_distributed(args)
    main_process = is_main_process(rank)
    args.exp_dir.mkdir(parents=True, exist_ok=True)
    for child in ["config", "logs", "checkpoints", "reports", "eval"]:
        (args.exp_dir / child).mkdir(exist_ok=True)

    config = json.loads(args.config.read_text()) if args.config.exists() else {}
    readiness = ct_snapshot(args.metadata, args.ct_root, ["train", "val"])
    train_entries = load_split_entries(args.metadata, ["train"])
    val_entries = load_split_entries(args.metadata, ["val"])
    base_train_entries = train_entries[:4] if args.smoke else train_entries
    rank_train_entries = base_train_entries[rank::world_size]
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": command_string(),
        "config": config,
        "train_cases": len(train_entries),
        "val_cases": len(val_entries),
        "ct_readiness": readiness,
        "mode": "smoke" if args.smoke else "full",
        "distributed": distributed,
        "rank": rank,
        "world_size": world_size,
        "local_rank": local_rank,
        "effective_batch_size": world_size * args.grad_accum,
        "status": "starting" if args.allow_experimental_train else "blocked_until_explicit_train_flag",
    }
    if main_process:
        write_json(args.exp_dir / "run_manifest.json", manifest)

    if not readiness["ready"]:
        if main_process:
            print("CT train+val data is not complete. Wrote manifest and exited.")
            print(f"Present {readiness['present_files']}/{readiness['expected_files']} files")
        cleanup_distributed()
        return 2

    if not args.allow_experimental_train:
        if main_process:
            print("Training is intentionally gated.")
            print("Pass --allow-experimental-train to run this text-conditioned training loop.")
        cleanup_distributed()
        return 3
    if not rank_train_entries:
        cleanup_distributed()
        raise RuntimeError(
            f"Rank {rank} has no training cases. Reduce nproc_per_node or increase smoke/full case count."
        )

    max_iterations = args.max_iterations or (20 if args.smoke else 10000)
    device = torch.device(f"cuda:{local_rank if distributed else args.gpu}" if torch.cuda.is_available() else "cpu")
    reader = NibabelIOWithReorient()
    if distributed and rank != 0:
        dist.barrier()
    predictor = VoxTellPredictor(
        model_dir=str(args.model_dir) if args.model_dir else None,
        device=device,
        embedding_bank=str(args.embeddings) if args.embeddings else None,
    )
    if distributed and rank == 0:
        dist.barrier()
    network: torch.nn.Module = predictor.network.to(device)
    if distributed:
        network = DDP(
            network,
            device_ids=[local_rank] if device.type == "cuda" else None,
            output_device=local_rank if device.type == "cuda" else None,
        )
    network.train()
    optimizer = torch.optim.AdamW(network.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    train_cycle = itertools.cycle(rank_train_entries)
    losses = []
    optimizer.zero_grad(set_to_none=True)
    for iteration in tqdm(range(1, max_iterations + 1), desc="Text FT iterations", disable=not main_process):
        entry = next(train_cycle)
        prompts = sorted_prompts(entry)
        image_patch, target_patch = load_case(entry, reader, args.ct_root, args.seg_dir)
        image_t = torch.from_numpy(image_patch).unsqueeze(0).to(device)
        target_t = torch.from_numpy(target_patch).unsqueeze(0).to(device)
        with torch.no_grad():
            text_embeddings = predictor.embed_text_prompts(prompts)
        if target_t.shape[1] != text_embeddings.shape[1]:
            raise ValueError(
                f"{entry['name']}: {target_t.shape[1]} GT masks but {text_embeddings.shape[1]} prompts"
            )

        with torch.autocast(device.type, enabled=device.type == "cuda"):
            logits = network(image_t, text_embeddings)
            loss = F.binary_cross_entropy_with_logits(logits.float(), target_t) + dice_loss(logits.float(), target_t)
            loss = loss / args.grad_accum

        scaler.scale(loss).backward()
        if iteration % args.grad_accum == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        loss_value = loss.detach() * args.grad_accum
        if distributed:
            dist.all_reduce(loss_value, op=dist.ReduceOp.SUM)
            loss_value /= world_size
        if main_process:
            losses.append(float(loss_value.cpu()))
        if main_process and iteration % args.checkpoint_every == 0:
            save_checkpoint(args.exp_dir / "checkpoints" / "checkpoint_latest.pth", network, optimizer, iteration)

    if distributed:
        dist.barrier()
    if main_process:
        save_checkpoint(args.exp_dir / "checkpoints" / "checkpoint_final.pth", network, optimizer, max_iterations)
        write_json(
            args.exp_dir / "reports" / "training_smoke_metrics.json",
            {
                "iterations": max_iterations,
                "mean_loss": float(np.mean(losses)) if losses else None,
                "last_loss": losses[-1] if losses else None,
                "losses": losses,
                "world_size": world_size,
            },
        )
        manifest["status"] = "completed"
        manifest["checkpoint_final"] = str(args.exp_dir / "checkpoints" / "checkpoint_final.pth")
        write_json(args.exp_dir / "run_manifest.json", manifest)
        print(f"Wrote final checkpoint to {args.exp_dir / 'checkpoints' / 'checkpoint_final.pth'}")
    cleanup_distributed()
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
