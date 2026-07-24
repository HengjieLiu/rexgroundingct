#!/usr/bin/env python3
"""Train a full-FOV, text-conditioned region proposal model for experiment 005."""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from collections import Counter, OrderedDict, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from common import REX_METADATA, load_split_entries, sha256_file, sorted_prompts, utc_now_iso, write_json
from global_proposal import (
    GLOBAL_SHAPE,
    GlobalProposalUNet,
    candidate_region_metrics,
    extract_candidate_starts,
    load_global_case,
    proposal_target_from_box,
)


class CaseLRU:
    def __init__(self, root: Path, max_size: int = 4):
        self.root = root
        self.max_size = max_size
        self.items: OrderedDict[str, tuple[np.ndarray, dict[str, Any]]] = OrderedDict()

    def load(self, name: str) -> tuple[np.ndarray, dict[str, Any]]:
        if name in self.items:
            value = self.items.pop(name)
            self.items[name] = value
            return value
        image, _targets, metadata = load_global_case(self.root, name, load_targets=False)
        value = (image, metadata)
        self.items[name] = value
        while len(self.items) > self.max_size:
            self.items.popitem(last=False)
        return value


def load_schedule(path: Path) -> list[dict]:
    events = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not events:
        raise ValueError(f"Empty schedule: {path}")
    return events


def load_embeddings(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        labels = [str(label).lower() for label in data["labels"]]
        embeddings = data["embeddings"]
    return {
        label: np.asarray(embeddings[index], dtype=np.float32)
        for index, label in enumerate(labels)
    }


def make_event_batch(
    event: dict,
    cache: CaseLRU,
    embeddings: dict[str, np.ndarray],
    margin_voxels: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Counter]:
    image, metadata = cache.load(event["case_name"])
    targets = []
    prompt_embeddings = []
    positive = []
    stats: Counter = Counter(samples=1)
    for slot in event["prompt_slots"]:
        prompt_embeddings.append(embeddings[str(slot["prompt"]).lower()])
        if slot.get("target_index") is None:
            targets.append(np.zeros(GLOBAL_SHAPE, dtype=np.uint8))
            positive.append(False)
            stats["negative_slots"] += 1
            continue
        target_info = metadata["targets"][int(slot["target_index"])]
        targets.append(
            proposal_target_from_box(
                target_info["proposal_box_global_zyx"],
                margin_voxels=margin_voxels,
                valid_bbox_zyx=metadata["valid_global_bbox_zyx"],
            )
        )
        positive.append(True)
        stats["positive_slots"] += 1
        stats[f"policy_{target_info['proposal_target_policy']}"] += 1
    image_tensor = torch.from_numpy(np.asarray(image, dtype=np.float32))[None].to(device)
    embedding_tensor = torch.from_numpy(np.stack(prompt_embeddings))[None].to(device)
    target_tensor = torch.from_numpy(np.stack(targets).astype(np.float32))[None].to(device)
    positive_tensor = torch.as_tensor(positive, dtype=torch.bool, device=device)[None]
    return image_tensor, embedding_tensor, target_tensor, positive_tensor, stats


def proposal_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    positive: torch.Tensor,
    background_weight: float,
    empty_weight: float,
    tversky_beta: float,
    hit_loss_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    probs = torch.sigmoid(logits.float())
    voxel_weights = torch.where(
        target > 0.5,
        torch.ones_like(target),
        torch.full_like(target, float(background_weight)),
    )
    bce = F.binary_cross_entropy_with_logits(logits.float(), target, reduction="none")
    bce_per_prompt = (bce * voxel_weights).flatten(2).mean(dim=2)
    dims = tuple(range(2, probs.ndim))
    tp = (probs * target).sum(dim=dims)
    fp = (probs * (1.0 - target)).sum(dim=dims)
    fn = ((1.0 - probs) * target).sum(dim=dims)
    alpha = 1.0 - float(tversky_beta)
    tversky = (tp + 1e-5) / (tp + alpha * fp + float(tversky_beta) * fn + 1e-5)
    tversky_loss = 1.0 - tversky
    target_mass = (probs * target).sum(dim=dims)
    total_mass = probs.sum(dim=dims).clamp(min=1e-6)
    target_mass_fraction = (target_mass / total_mass).clamp(min=1e-8, max=1.0)
    hit_loss = -torch.log(target_mass_fraction)
    positive_loss = bce_per_prompt + tversky_loss + float(hit_loss_weight) * hit_loss
    empty_loss = bce_per_prompt * float(empty_weight)
    loss = torch.where(positive, positive_loss, empty_loss).mean()
    return loss, {
        "bce": float(bce_per_prompt.mean().detach().cpu()),
        "tversky_loss": float(tversky_loss[positive].mean().detach().cpu()) if positive.any() else 0.0,
        "hit_loss": float(hit_loss[positive].mean().detach().cpu()) if positive.any() else 0.0,
        "target_mass_fraction": (
            float(target_mass_fraction[positive].mean().detach().cpu()) if positive.any() else 0.0
        ),
    }


def aggregate_rows(rows: list[dict]) -> dict[str, Any]:
    output: dict[str, Any] = {"findings": len(rows)}
    for k in (1, 3, 5):
        output[f"hit_at_{k}"] = float(np.mean([row[f"hit_at_{k}"] for row in rows]))
        output[f"full_inclusion_at_{k}"] = float(
            np.mean([row[f"full_inclusion_at_{k}"] for row in rows])
        )
        output[f"target_coverage_at_{k}"] = float(
            np.mean([row[f"target_coverage_at_{k}"] for row in rows])
        )
    output["mean_top1_score"] = float(np.mean([row["top1_score"] for row in rows]))
    return output


@torch.inference_mode()
def validate(
    model: GlobalProposalUNet,
    entries: list[dict],
    cache_root: Path,
    embeddings: dict[str, np.ndarray],
    device: torch.device,
    top_k: int,
    nms_radius: int,
) -> dict[str, Any]:
    model.eval()
    rows = []
    for entry in tqdm(entries, desc="proposal val", leave=False):
        image, native_targets, metadata = load_global_case(cache_root, entry["name"], load_targets=True)
        assert native_targets is not None
        prompts = sorted_prompts(entry)
        prompt_embeddings = torch.from_numpy(
            np.stack([embeddings[prompt.lower()] for prompt in prompts])
        )[None].to(device)
        image_tensor = torch.from_numpy(np.asarray(image, dtype=np.float32))[None].to(device)
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            logits = model(image_tensor, prompt_embeddings)
        probabilities = torch.sigmoid(logits.float()).cpu().numpy()[0]
        for index, prompt in enumerate(prompts):
            candidates = extract_candidate_starts(
                probabilities[index],
                metadata,
                top_k=top_k,
                nms_radius=nms_radius,
            )
            metrics = candidate_region_metrics(candidates, native_targets[index])
            rows.append(
                {
                    "name": entry["name"],
                    "finding_index": index,
                    "prompt": prompt,
                    "category": entry.get("categories", {}).get(str(index), "unknown"),
                    "native_voxels": int(native_targets[index].sum()),
                    "inclusion_possible": bool(metadata["targets"][index]["inclusion_possible"]),
                    "top1_score": float(candidates[0]["score"]),
                    "candidates": candidates,
                    **metrics,
                }
            )
    model.train()
    by_category: dict[str, Any] = {}
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["category"]].append(row)
    for label, items in sorted(grouped.items()):
        by_category[label] = aggregate_rows(items)
    return {
        "created_at_utc": utc_now_iso(),
        "summary": aggregate_rows(rows),
        "by_category": by_category,
        "rows": rows,
    }


def write_validation_report(path: Path, result: dict[str, Any], step: int) -> None:
    summary = result["summary"]
    lines = [
        f"# Global Proposal Validation at Step {step}",
        "",
        "| K | Hit | Full inclusion | Mean target coverage |",
        "| ---: | ---: | ---: | ---: |",
    ]
    for k in (1, 3, 5):
        lines.append(
            f"| {k} | {summary[f'hit_at_{k}']:.4f} | "
            f"{summary[f'full_inclusion_at_{k}']:.4f} | "
            f"{summary[f'target_coverage_at_{k}']:.4f} |"
        )
    lines.extend(
        [
            "",
            f"- Findings: `{summary['findings']}`",
            f"- Mean top-1 score: `{summary['mean_top1_score']:.6f}`",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def atomic_checkpoint(
    path: Path,
    model: GlobalProposalUNet,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    step: int,
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    torch.save(
        {
            "network_weights": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "grad_scaler": scaler.state_dict(),
            "step": step,
            "args": vars(args),
            "model": {
                "class": "GlobalProposalUNet",
                "base_channels": args.base_channels,
                "global_shape": list(GLOBAL_SHAPE),
            },
        },
        tmp,
    )
    os.replace(tmp, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=REX_METADATA)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--val-json", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--steps-per-epoch", type=int, default=100)
    parser.add_argument("--base-channels", type=int, default=8)
    parser.add_argument("--target-margin-voxels", type=int, default=0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--background-weight", type=float, default=0.05)
    parser.add_argument("--empty-weight", type=float, default=0.25)
    parser.add_argument("--tversky-beta", type=float, default=0.8)
    parser.add_argument("--hit-loss-weight", type=float, default=0.25)
    parser.add_argument("--clip-grad-norm", type=float, default=12.0)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--nms-radius", type=int, default=8)
    parser.add_argument("--stop-file", type=Path, default=None)
    parser.add_argument("--smoke-steps", type=int, default=None)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cudnn.benchmark = True
    args.run_dir.mkdir(parents=True, exist_ok=True)
    (args.run_dir / "reports").mkdir(exist_ok=True)
    (args.run_dir / "checkpoints").mkdir(exist_ok=True)
    write_json(
        args.run_dir / "config.json",
        {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    )

    events = load_schedule(args.schedule)
    total_steps = min(args.steps, args.smoke_steps) if args.smoke_steps else args.steps
    if len(events) < total_steps:
        raise ValueError(f"Schedule has {len(events)} events but {total_steps} are required")
    embeddings = load_embeddings(args.embeddings)
    train_entries = load_split_entries(args.metadata, ["train"])
    required_prompts = {prompt.lower() for entry in train_entries for prompt in sorted_prompts(entry)}
    missing = required_prompts - embeddings.keys()
    if missing:
        raise KeyError(f"Embedding bank missing {len(missing)} training prompts")
    val_entries = json.loads(args.val_json.read_text())["test"]
    cache = CaseLRU(args.cache_root, max_size=4)
    model = GlobalProposalUNet(base_channels=args.base_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    sample_stats: Counter = Counter()
    history_path = args.run_dir / "history.jsonl"
    start_time = time.perf_counter()
    stopped_early = False
    model.train()

    for step in tqdm(range(1, total_steps + 1), desc=args.run_dir.name):
        if step <= args.warmup_steps:
            factor = step / float(max(1, args.warmup_steps))
        else:
            progress = (step - args.warmup_steps) / float(max(1, total_steps - args.warmup_steps))
            factor = 0.5 * (1.0 + np.cos(np.pi * min(max(progress, 0.0), 1.0)))
        for group in optimizer.param_groups:
            group["lr"] = args.lr * factor
        image, text, target, positive, stats = make_event_batch(
            events[step - 1],
            cache,
            embeddings,
            args.target_margin_voxels,
            device,
        )
        sample_stats.update(stats)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            logits = model(image, text)
            loss, components = proposal_loss(
                logits,
                target,
                positive,
                args.background_weight,
                args.empty_weight,
                args.tversky_beta,
                args.hit_loss_weight,
            )
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at step {step}")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        record: dict[str, Any] = {
            "step": step,
            "epoch": step / args.steps_per_epoch,
            "loss": float(loss.detach().cpu()),
            "lr": optimizer.param_groups[0]["lr"],
            "grad_norm": float(grad_norm.detach().cpu()),
            "case": events[step - 1]["case_name"],
            **components,
        }
        if step == 1 or step % args.eval_every == 0 or step == total_steps:
            result = validate(
                model,
                val_entries,
                args.cache_root,
                embeddings,
                device,
                args.top_k,
                args.nms_radius,
            )
            result["step"] = step
            result["epoch"] = step / args.steps_per_epoch
            write_json(args.run_dir / "reports" / f"val20_step_{step:05d}.json", result)
            write_validation_report(
                args.run_dir / "reports" / f"val20_step_{step:05d}.md",
                result,
                step,
            )
            record["val20"] = result["summary"]
        with history_path.open("a") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        if step % args.checkpoint_every == 0 or step == total_steps:
            atomic_checkpoint(
                args.run_dir / "checkpoints" / f"checkpoint_step_{step:05d}.pth",
                model,
                optimizer,
                scaler,
                step,
                args,
            )
            atomic_checkpoint(
                args.run_dir / "checkpoints" / "checkpoint_latest.pth",
                model,
                optimizer,
                scaler,
                step,
                args,
            )
        if args.stop_file is not None and args.stop_file.exists():
            atomic_checkpoint(
                args.run_dir / "checkpoints" / f"checkpoint_stopped_step_{step:05d}.pth",
                model,
                optimizer,
                scaler,
                step,
                args,
            )
            stopped_early = True
            break

    summary = {
        "status": "stopped_early" if stopped_early else "completed",
        "created_at_utc": utc_now_iso(),
        "requested_steps": args.steps,
        "completed_step": step,
        "elapsed_seconds": time.perf_counter() - start_time,
        "schedule": str(args.schedule),
        "schedule_sha256": sha256_file(args.schedule),
        "sample_stats": dict(sample_stats),
        "peak_cuda_allocated_gib": (
            torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == "cuda" else 0
        ),
    }
    write_json(args.run_dir / "reports" / "training_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
