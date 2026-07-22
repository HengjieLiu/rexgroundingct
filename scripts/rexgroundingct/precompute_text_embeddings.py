#!/usr/bin/env python3
"""Precompute VoxTell text embeddings for ReXGroundingCT findings."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from voxtell.inference.predictor import VoxTellPredictor

from common import EXP_ROOT, REX_METADATA, load_split_entries, sorted_prompts


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
    args = parser.parse_args()

    entries = load_split_entries(args.metadata, args.splits)
    prompts: list[str] = []
    seen = set()
    for entry in entries:
        for prompt in sorted_prompts(entry):
            key = prompt.lower()
            if key not in seen:
                seen.add(key)
                prompts.append(prompt)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    predictor = VoxTellPredictor(model_dir=str(args.model_dir) if args.model_dir else None, device=device)
    embeddings = []
    for prompt in tqdm(prompts, desc="Embedding prompts"):
        emb = predictor.embed_text_prompts([prompt])[0, 0].detach().cpu().numpy().astype(np.float16)
        embeddings.append(emb)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        labels=np.array([prompt.lower() for prompt in prompts]),
        embeddings=np.stack(embeddings, axis=0),
    )
    print(f"Wrote {len(prompts)} embeddings to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
