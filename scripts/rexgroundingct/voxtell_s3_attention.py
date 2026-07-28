#!/usr/bin/env python3
"""S3 attention coupling wrappers for VoxTell continuation experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn
from voxtell.inference.predictor import VoxTellPredictor


MODEL_SPEC_FILENAME = "model_spec.json"
MODEL_TYPE = "voxtell_s3_attention_v1"
S3_SCALES = ("1/1", "1/2", "1/4")
S3_HALF_QUARTER_SCALES = ("1/2", "1/4")

VARIANTS: dict[str, dict[str, Any]] = {
    "s3v1_fixedrho_suppress_allscales": {
        "coupling": "fixedrho_suppress",
        "attention_scales": S3_SCALES,
        "rho_fixed": 0.25,
        "alpha_target": None,
        "beta_target": None,
    },
    "s3v2_balanced_feature_allscales": {
        "coupling": "balanced_feature",
        "attention_scales": S3_SCALES,
        "rho_fixed": None,
        "alpha_target": 0.25,
        "beta_target": None,
    },
    "s3v3_logit_residual_allscales": {
        "coupling": "logit_residual",
        "attention_scales": S3_SCALES,
        "rho_fixed": None,
        "alpha_target": None,
        "beta_target": 1.0,
    },
    "s3v1_fixedrho_suppress_half_quarter": {
        "coupling": "fixedrho_suppress",
        "attention_scales": S3_HALF_QUARTER_SCALES,
        "rho_fixed": 0.25,
        "alpha_target": None,
        "beta_target": None,
    },
    "s3v2_balanced_feature_half_quarter": {
        "coupling": "balanced_feature",
        "attention_scales": S3_HALF_QUARTER_SCALES,
        "rho_fixed": None,
        "alpha_target": 0.25,
        "beta_target": None,
    },
    "s3v3_logit_residual_half_quarter": {
        "coupling": "logit_residual",
        "attention_scales": S3_HALF_QUARTER_SCALES,
        "rho_fixed": None,
        "alpha_target": None,
        "beta_target": 1.0,
    },
}


def variant_config(variant: str) -> dict[str, Any]:
    if variant not in VARIANTS:
        raise ValueError(f"Unsupported S3 variant {variant!r}; choose from {sorted(VARIANTS)}")
    return dict(VARIANTS[variant])


def scale_key(scale: str) -> str:
    return str(scale).replace("/", "over")


def coupling_scale_for_update(global_update: int, ramp_updates: int) -> float:
    ramp_updates = int(ramp_updates)
    if ramp_updates <= 0:
        return 1.0
    return min(1.0, max(0.0, float(global_update) / float(ramp_updates)))


class S3VoxelTextAttentionGate3D(nn.Module):
    """Cosine voxel-text attention map used by the S3 variants."""

    def __init__(
        self,
        skip_channels: int,
        context_channels: int,
        query_dim: int,
        hidden_channels: int = 128,
        logit_scale_init: float = 10.0,
    ) -> None:
        super().__init__()
        if logit_scale_init <= 0:
            raise ValueError("logit_scale_init must be positive")
        self.skip_projection = nn.Conv3d(skip_channels, hidden_channels, 1)
        self.context_projection = nn.Conv3d(context_channels, hidden_channels, 1)
        self.visual_projection = nn.Conv3d(hidden_channels, hidden_channels, 1)
        self.text_projection = nn.Linear(query_dim, hidden_channels)
        self.logit_scale = nn.Parameter(torch.tensor(float(logit_scale_init)).log())
        self.bias = nn.Parameter(torch.zeros(()))

    def forward(
        self,
        skip: torch.Tensor,
        coarse_context: torch.Tensor,
        query: torch.Tensor,
    ) -> torch.Tensor:
        if query.ndim == 4 and query.shape[2] == 1:
            query = query.squeeze(2)
        if query.ndim != 3:
            raise ValueError(f"Expected query [B,P,C], got {tuple(query.shape)}")
        if coarse_context.shape[2:] != skip.shape[2:]:
            coarse_context = F.interpolate(
                coarse_context,
                size=skip.shape[2:],
                mode="trilinear",
                align_corners=False,
            )

        visual = torch.relu(self.skip_projection(skip) + self.context_projection(coarse_context))
        visual = F.normalize(self.visual_projection(visual), dim=1, eps=1e-6)
        text = F.normalize(self.text_projection(query), dim=-1, eps=1e-6)
        similarity = torch.einsum("bcdhw,bpc->bpdhw", visual, text)
        scale = self.logit_scale.exp().clamp(max=50.0)
        return torch.sigmoid(scale * similarity + self.bias)[:, :, None]


class ZeroInitLogitResidualHead3D(nn.Module):
    """Tiny per-prompt residual head whose output is exactly zero initially."""

    def __init__(self, input_channels: int, hidden_channels: int = 16) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv3d(input_channels, hidden_channels, 1),
            nn.GELU(),
            nn.Conv3d(hidden_channels, 1, 1),
        )
        nn.init.zeros_(self.layers[-1].weight)
        nn.init.zeros_(self.layers[-1].bias)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.layers(value)


class S3AttentionVoxTellModel(nn.Module):
    """VoxTell with S3 attention and variant-specific coupling."""

    is_s3_attention_voxtell = True

    def __init__(
        self,
        source_model: nn.Module,
        variant: str,
        deep_supervision: bool = False,
        attention_scales: tuple[str, ...] | None = None,
        attention_hidden_channels: int = 128,
        logit_scale_init: float = 10.0,
        logit_residual_hidden_channels: int = 16,
        coupling_ramp_updates: int = 2000,
    ) -> None:
        super().__init__()
        config = variant_config(variant)
        self.variant = variant
        self.coupling = str(config["coupling"])
        self.rho_fixed = None if config["rho_fixed"] is None else float(config["rho_fixed"])
        self.alpha_target = None if config["alpha_target"] is None else float(config["alpha_target"])
        self.beta_target = None if config["beta_target"] is None else float(config["beta_target"])
        if attention_scales is None:
            attention_scales = tuple(str(scale) for scale in config.get("attention_scales", S3_SCALES))
        self.attention_scales = tuple(str(scale) for scale in attention_scales)
        self.attention_hidden_channels = int(attention_hidden_channels)
        self.logit_scale_init = float(logit_scale_init)
        self.logit_residual_hidden_channels = int(logit_residual_hidden_channels)
        self.coupling_ramp_updates = int(coupling_ramp_updates)
        self.deep_supervision = bool(deep_supervision)

        self.selected_decoder_layer = int(source_model.selected_decoder_layer)
        self.query_dim = int(source_model.query_dim)
        self.num_heads = int(source_model.num_heads)
        self.project_to_decoder_hidden_dim = int(source_model.project_to_decoder_hidden_dim)
        self.text_embedding_dim = int(source_model.text_embedding_dim)

        self.encoder = source_model.encoder
        self.decoder = source_model.decoder
        self.decoder.deep_supervision = True
        self.project_bottleneck_embed = source_model.project_bottleneck_embed
        self.project_text_embed = source_model.project_text_embed
        self.transformer_decoder = source_model.transformer_decoder
        self.project_to_decoder_channels = source_model.project_to_decoder_channels
        self.register_buffer("pos_embed", source_model.pos_embed.detach().clone())
        self.register_buffer("s3_coupling_scale", torch.zeros(()))

        valid_scales = {"1/4": 2, "1/2": 1, "1/1": 0}
        unknown = sorted(set(self.attention_scales) - set(valid_scales))
        if unknown:
            raise ValueError(f"Unsupported S3 scale(s): {unknown}")
        n_stages = len(self.encoder.output_channels)
        self.s3_attention_decoder_stage_to_scale: dict[int, str] = {}
        self.s3_attention_gates = nn.ModuleDict()
        for scale in self.attention_scales:
            encoder_level = valid_scales[scale]
            if encoder_level >= n_stages - 1:
                raise ValueError(f"S3 scale {scale} is unavailable")
            decoder_stage = n_stages - 2 - encoder_level
            channels = int(self.encoder.output_channels[encoder_level])
            self.s3_attention_gates[scale_key(scale)] = S3VoxelTextAttentionGate3D(
                skip_channels=channels,
                context_channels=channels,
                query_dim=self.query_dim,
                hidden_channels=self.attention_hidden_channels,
                logit_scale_init=self.logit_scale_init,
            )
            self.s3_attention_decoder_stage_to_scale[decoder_stage] = scale

        self.logit_residual_head = None
        if self.coupling == "logit_residual":
            self.logit_residual_head = ZeroInitLogitResidualHead3D(
                input_channels=1 + len(self.attention_scales),
                hidden_channels=self.logit_residual_hidden_channels,
            )

        self.last_s3_attention_maps: dict[str, list[torch.Tensor]] = {}
        self.last_s3_effective_strength = 0.0

    def set_coupling_scale(self, value: float) -> None:
        clamped = min(1.0, max(0.0, float(value)))
        self.s3_coupling_scale.fill_(clamped)

    def model_spec(self, source_model_dir: str | None = None) -> dict[str, Any]:
        return {
            "model_type": MODEL_TYPE,
            "schema_version": 1,
            "variant": self.variant,
            "coupling": self.coupling,
            "attention_scales": list(self.attention_scales),
            "attention_hidden_channels": self.attention_hidden_channels,
            "logit_scale_init": self.logit_scale_init,
            "rho_fixed": self.rho_fixed,
            "alpha_target": self.alpha_target,
            "beta_target": self.beta_target,
            "logit_residual_hidden_channels": self.logit_residual_hidden_channels,
            "coupling_ramp_updates": self.coupling_ramp_updates,
            "source_model_dir": source_model_dir,
        }

    def _shared_fusion(
        self,
        image: torch.Tensor,
        text_embedding: torch.Tensor,
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        skips = self.encoder(image)
        selected_feature = skips[self.selected_decoder_layer]
        bottleneck_embed = rearrange(selected_feature, "b c d h w -> b h w d c")
        bottleneck_embed = self.project_bottleneck_embed(bottleneck_embed)
        bottleneck_embed = rearrange(bottleneck_embed, "b h w d c -> (h w d) b c")
        text_embedding = text_embedding.squeeze(2)
        text_embed = repeat(text_embedding, "b n dim -> n b dim")
        text_embed = self.project_text_embed(text_embed)
        mask_embedding, _ = self.transformer_decoder(
            tgt=text_embed,
            memory=bottleneck_embed,
            pos=self.pos_embed,
            memory_key_padding_mask=None,
        )
        mask_embedding = repeat(mask_embedding, "n b dim -> b n dim")
        return skips, mask_embedding

    def _apply_skip_coupling(
        self,
        skip: torch.Tensor,
        attention: torch.Tensor,
    ) -> torch.Tensor:
        attention_skip = attention[:, 0].to(dtype=skip.dtype)
        ramp = self.s3_coupling_scale.to(device=skip.device, dtype=skip.dtype)
        if self.coupling == "fixedrho_suppress":
            rho = float(self.rho_fixed) * ramp
            self.last_s3_effective_strength = float(rho.detach().cpu())
            return skip * ((1.0 - rho) + rho * attention_skip)
        if self.coupling == "balanced_feature":
            alpha = float(self.alpha_target) * ramp
            self.last_s3_effective_strength = float(alpha.detach().cpu())
            return skip * (1.0 + alpha * (2.0 * attention_skip - 1.0))
        if self.coupling == "logit_residual":
            self.last_s3_effective_strength = float((float(self.beta_target) * ramp).detach().cpu())
            return skip
        raise ValueError(f"Unsupported S3 coupling: {self.coupling}")

    def _apply_logit_residual(
        self,
        logits: torch.Tensor,
        prompt_attention: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        if self.logit_residual_head is None:
            return logits
        pieces = [logits.float()]
        for scale in self.attention_scales:
            if scale not in prompt_attention:
                raise RuntimeError(f"Missing {scale} attention map for logit residual")
            attention = prompt_attention[scale][:, 0].float()
            if tuple(attention.shape[2:]) != tuple(logits.shape[2:]):
                attention = F.interpolate(
                    attention,
                    size=logits.shape[2:],
                    mode="trilinear",
                    align_corners=False,
                )
            pieces.append(attention)
        residual_input = torch.cat(pieces, dim=1)
        ramp = self.s3_coupling_scale.to(device=logits.device, dtype=logits.dtype)
        beta = float(self.beta_target) * ramp
        residual = self.logit_residual_head(residual_input).to(dtype=logits.dtype)
        self.last_s3_effective_strength = float(beta.detach().cpu())
        return logits + beta * residual

    def _decode_prompt_with_s3(
        self,
        skips: list[torch.Tensor],
        mask_embeddings: list[torch.Tensor],
        query: torch.Tensor,
    ) -> list[torch.Tensor]:
        decoder = self.decoder
        lres_input = skips[-1]
        seg_outputs: list[torch.Tensor] = []
        mask_embeddings = list(mask_embeddings)[::-1]
        prompt_attention: dict[str, torch.Tensor] = {}

        for stage_idx in range(len(decoder.stages)):
            coarse_context = decoder.transpconvs[stage_idx](lres_input)
            skip = skips[-(stage_idx + 2)]
            scale = self.s3_attention_decoder_stage_to_scale.get(stage_idx)
            if scale is not None:
                attention = self.s3_attention_gates[scale_key(scale)](
                    skip,
                    coarse_context,
                    query,
                )
                self.last_s3_attention_maps[scale].append(attention)
                prompt_attention[scale] = attention
                skip = self._apply_skip_coupling(skip, attention)

            x = decoder.stages[stage_idx](torch.cat((coarse_context, skip), dim=1))
            if stage_idx == len(decoder.stages) - 1:
                seg_pred = torch.einsum(
                    "b c h w d, b n c -> b n h w d",
                    x,
                    mask_embeddings[-1],
                )
                seg_outputs.append(self._apply_logit_residual(seg_pred, prompt_attention))
            elif stage_idx >= len(decoder.stages) - len(mask_embeddings):
                mask_embedding = mask_embeddings.pop(0)
                batch_size, _, channels = mask_embedding.shape
                reshaped = mask_embedding.view(batch_size, decoder.num_heads, -1)
                fusion = torch.einsum("b c h w d, b n c -> b n h w d", x, reshaped)
                x = torch.cat((x, fusion), dim=1)
                seg_outputs.append(decoder.seg_layers[stage_idx](x))
            lres_input = x

        seg_outputs = seg_outputs[::-1]
        return seg_outputs if decoder.deep_supervision else seg_outputs[:1]

    def forward(
        self,
        image: torch.Tensor,
        text_embedding: torch.Tensor,
    ) -> torch.Tensor | list[torch.Tensor]:
        self.last_s3_attention_maps = {scale: [] for scale in self.attention_scales}
        self.last_s3_effective_strength = 0.0
        skips, mask_embedding = self._shared_fusion(image, text_embedding)
        mask_embeddings = [
            projection(mask_embedding)
            for projection in self.project_to_decoder_channels
        ]
        per_prompt_outputs = []
        for prompt_idx in range(mask_embedding.shape[1]):
            prompt_embeddings = [
                embedding[:, prompt_idx : prompt_idx + 1]
                for embedding in mask_embeddings
            ]
            per_prompt_outputs.append(
                self._decode_prompt_with_s3(
                    skips,
                    prompt_embeddings,
                    mask_embedding[:, prompt_idx : prompt_idx + 1],
                )
            )
        outputs = [
            torch.cat(scale_outputs, dim=1)
            for scale_outputs in zip(*per_prompt_outputs)
        ]
        if self.deep_supervision:
            return outputs
        return outputs[0]


def unwrap_s3_attention(network: nn.Module) -> S3AttentionVoxTellModel | None:
    model = network.module if hasattr(network, "module") else network
    if isinstance(model, S3AttentionVoxTellModel):
        return model
    return None


def is_s3_attention_model(network: nn.Module) -> bool:
    return unwrap_s3_attention(network) is not None


def model_spec_from_network(
    network: nn.Module,
    source_model_dir: str | None = None,
) -> dict[str, Any] | None:
    model = unwrap_s3_attention(network)
    if model is None:
        return None
    return model.model_spec(source_model_dir=source_model_dir)


def set_s3_coupling_scale(network: nn.Module, value: float) -> None:
    model = unwrap_s3_attention(network)
    if model is not None:
        model.set_coupling_scale(value)


def s3_attention_stats(network: nn.Module) -> dict[str, float]:
    model = unwrap_s3_attention(network)
    if model is None:
        return {}
    stats: dict[str, float] = {
        "s3_coupling_scale": float(model.s3_coupling_scale.detach().cpu()),
        "s3_effective_strength": float(model.last_s3_effective_strength),
    }
    for scale, maps in model.last_s3_attention_maps.items():
        if not maps:
            continue
        attention = torch.cat(maps, dim=1).detach().float()
        suffix = scale_key(scale)
        stats[f"s3_attention_mean_{suffix}"] = float(attention.mean().cpu())
        stats[f"s3_attention_std_{suffix}"] = float(attention.std().cpu())
        stats[f"s3_attention_min_{suffix}"] = float(attention.amin().cpu())
        stats[f"s3_attention_max_{suffix}"] = float(attention.amax().cpu())
        gate = model.s3_attention_gates[scale_key(scale)]
        stats[f"s3_logit_scale_{suffix}"] = float(gate.logit_scale.detach().exp().clamp(max=50.0).cpu())
        stats[f"s3_bias_{suffix}"] = float(gate.bias.detach().cpu())
    return stats


def write_model_spec(path: Path, spec: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n")


def load_model_spec(model_dir: Path) -> dict[str, Any] | None:
    path = Path(model_dir) / MODEL_SPEC_FILENAME
    if not path.is_file():
        return None
    spec = json.loads(path.read_text())
    if spec.get("model_type") != MODEL_TYPE:
        raise ValueError(f"Unsupported S3 model type in {path}: {spec.get('model_type')!r}")
    variant_config(str(spec["variant"]))
    return spec


class S3AttentionVoxTellPredictor(VoxTellPredictor):
    """VoxTell predictor that loads an S3 attention wrapper checkpoint."""

    def __init__(
        self,
        model_dir: str | Path,
        device: torch.device,
        embedding_bank: str | dict[str, Any] | None = None,
        use_precomputed_embeddings: bool = True,
    ) -> None:
        model_dir = Path(model_dir)
        spec = load_model_spec(model_dir)
        if spec is None:
            raise ValueError(f"Missing {MODEL_SPEC_FILENAME} under {model_dir}")
        source_model_dir = spec.get("source_model_dir")
        if not source_model_dir:
            raise ValueError(f"{model_dir / MODEL_SPEC_FILENAME}: source_model_dir is required")
        super().__init__(
            model_dir=str(source_model_dir),
            device=device,
            embedding_bank=embedding_bank,
            use_precomputed_embeddings=use_precomputed_embeddings,
        )
        network = S3AttentionVoxTellModel(
            source_model=self.network,
            variant=str(spec["variant"]),
            deep_supervision=False,
            attention_scales=tuple(str(scale) for scale in spec.get("attention_scales", S3_SCALES)),
            attention_hidden_channels=int(spec.get("attention_hidden_channels") or 128),
            logit_scale_init=float(spec.get("logit_scale_init") or 10.0),
            logit_residual_hidden_channels=int(spec.get("logit_residual_hidden_channels") or 16),
            coupling_ramp_updates=int(spec.get("coupling_ramp_updates") or 2000),
        )
        checkpoint = torch.load(
            model_dir / "fold_0" / "checkpoint_final.pth",
            map_location=torch.device("cpu"),
            weights_only=False,
        )
        checkpoint_spec = checkpoint.get("model_spec")
        if checkpoint_spec is not None and checkpoint_spec.get("variant") != spec.get("variant"):
            raise ValueError(
                f"Checkpoint variant {checkpoint_spec.get('variant')!r} does not "
                f"match model spec variant {spec.get('variant')!r}"
            )
        network.load_state_dict(checkpoint["network_weights"], strict=True)
        network.deep_supervision = False
        network.decoder.deep_supervision = False
        network.set_coupling_scale(
            coupling_scale_for_update(
                int(checkpoint.get("global_update", 0)),
                int(spec.get("coupling_ramp_updates") or 2000),
            )
        )
        network = network.to(device)
        network.eval()
        self.network = network
