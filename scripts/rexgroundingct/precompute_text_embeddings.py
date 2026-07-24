#!/usr/bin/env python3
"""Precompute VoxTell text embeddings for ReXGroundingCT findings.

The output is a resumable `.npz` bank with lowercase prompt labels. Training
expects this bank and does not load the Qwen text backbone in the training loop.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from voxtell.inference.predictor import VoxTellPredictor
from voxtell.utils.text_embedding import last_token_pool, wrap_with_instruction

from common import EXP_ROOT, REX_METADATA, load_split_entries, sorted_prompts


def collect_unique_prompts(metadata: Path, splits: list[str]) -> list[str]:
    entries = load_split_entries(metadata, splits)
    prompts: list[str] = []
    seen: set[str] = set()
    for entry in entries:
        for prompt in sorted_prompts(entry):
            key = prompt.lower()
            if key not in seen:
                seen.add(key)
                prompts.append(key)
    return prompts


def load_existing_bank(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        return {}
    data = np.load(path)
    labels = [str(label).lower() for label in data["labels"]]
    embeddings = data["embeddings"]
    if len(labels) != int(embeddings.shape[0]):
        raise ValueError(f"{path}: labels and embeddings length mismatch")
    return {
        label: np.asarray(embeddings[index], dtype=np.float16)
        for index, label in enumerate(labels)
    }


def write_bank_atomic(path: Path, prompt_order: list[str], bank: dict[str, np.ndarray]) -> None:
    available = [prompt for prompt in prompt_order if prompt in bank]
    if not available:
        raise ValueError("Cannot write an empty embedding bank")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    labels = np.array(available)
    embeddings = np.stack([bank[prompt] for prompt in available], axis=0)
    with tmp.open("wb") as handle:
        np.savez_compressed(handle, labels=labels, embeddings=embeddings)
    os.replace(tmp, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=REX_METADATA)
    parser.add_argument("--splits", nargs="+", choices=["train", "val", "test"], default=["train", "val"])
    parser.add_argument(
        "--output",
        type=Path,
        default=EXP_ROOT / "002_voxtell_text_ft_miccai_train_val" / "config" / "rex_text_embeddings.npz",
    )
    parser.add_argument("--model-dir", type=Path, default=None)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--save-every-batches", type=int, default=10)
    args = parser.parse_args()

    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if args.save_every_batches < 1:
        raise ValueError("--save-every-batches must be >= 1")

    prompts = collect_unique_prompts(args.metadata, args.splits)
    bank = load_existing_bank(args.output)
    existing = sum(1 for prompt in prompts if prompt in bank)
    missing_prompts = [prompt for prompt in prompts if prompt not in bank]
    print(
        f"Prompt bank target={len(prompts)} existing={existing} "
        f"missing={len(missing_prompts)} output={args.output}",
        flush=True,
    )
    if not missing_prompts:
        write_bank_atomic(args.output, prompts, bank)
        print(f"Embedding bank already complete: {args.output}")
        return 0

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    predictor = VoxTellPredictor(
        model_dir=str(args.model_dir) if args.model_dir else None,
        device=device,
        use_precomputed_embeddings=False,
    )

    predictor._ensure_text_backbone()
    predictor.text_backbone = predictor.text_backbone.to(device)
    try:
        batch_starts = list(range(0, len(missing_prompts), args.batch_size))
        for batch_index, start in enumerate(tqdm(batch_starts, desc="Embedding prompt batches"), start=1):
            chunk = missing_prompts[start:start + args.batch_size]
            wrapped = wrap_with_instruction(chunk)
            text_tokens = predictor.tokenizer(
                wrapped,
                padding=True,
                truncation=True,
                max_length=predictor.max_text_length,
                return_tensors="pt",
            )
            text_tokens = {key: value.to(device) for key, value in text_tokens.items()}
            with torch.inference_mode():
                text_embed = predictor.text_backbone(**text_tokens)
                embeddings = last_token_pool(
                    text_embed.last_hidden_state,
                    text_tokens["attention_mask"],
                ).float()
            embeddings = embeddings.detach().cpu().numpy().astype(np.float16)
            for prompt, embedding in zip(chunk, embeddings):
                bank[prompt] = embedding
            if batch_index % args.save_every_batches == 0 or batch_index == len(batch_starts):
                write_bank_atomic(args.output, prompts, bank)
    finally:
        predictor.text_backbone = predictor.text_backbone.to("cpu")
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print(f"Wrote {len(prompts)} embeddings to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
