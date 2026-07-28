#!/usr/bin/env python3
"""Dual-branch proposal/refinement extensions for the VoxTell model."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import torch
from einops import rearrange, repeat
from acvl_utils.cropping_and_padding.padding import pad_nd_image
from nnunetv2.inference.sliding_window_prediction import compute_gaussian
from nnunetv2.utilities.helpers import dummy_context, empty_cache
from torch import nn
from tqdm import tqdm
from voxtell.inference.predictor import VoxTellPredictor


MODEL_SPEC_FILENAME = "model_spec.json"
MODEL_TYPE = "voxtell_dual_branch_v1"

VARIANTS: dict[str, dict[str, Any]] = {
    "v1_sharedfusion_softguide": {
        "fusion_mode": "shared",
        "detach_proposal_guide": True,
        "final_precision_weight": 0.0,
    },
    "v1_dualfusion_softguide": {
        "fusion_mode": "dual_adapter",
        "detach_proposal_guide": True,
        "final_precision_weight": 0.0,
    },
    "v2_dualfusion_precision": {
        "fusion_mode": "dual_adapter",
        "detach_proposal_guide": True,
        "final_precision_weight": 0.1,
    },
    "v3_dualfusion_softguide_joint": {
        "fusion_mode": "dual_adapter",
        "detach_proposal_guide": False,
        "final_precision_weight": 0.0,
    },
}


def variant_config(variant: str) -> dict[str, Any]:
    if variant not in VARIANTS:
        raise ValueError(
            f"Unsupported dual-branch variant {variant!r}; choose from {sorted(VARIANTS)}"
        )
    return dict(VARIANTS[variant])


class ZeroResidualAdapter(nn.Module):
    """A residual MLP whose output is exactly zero at initialization."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )
        nn.init.zeros_(self.layers[-1].weight)
        nn.init.zeros_(self.layers[-1].bias)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.layers(self.norm(value))


class GuidedVoxTellDecoder(nn.Module):
    """A copied VoxTell decoder with zero-initialized multiscale soft guides."""

    def __init__(
        self,
        source_decoder: nn.Module,
        shared_encoder: nn.Module,
    ) -> None:
        super().__init__()
        self.decoder = copy.deepcopy(
            source_decoder,
            memo={id(source_decoder.encoder): shared_encoder},
        )
        self.decoder.deep_supervision = True
        output_channels = list(self.decoder.encoder.output_channels)
        stage_channels = [
            int(output_channels[-(stage_index + 2)])
            for stage_index in range(len(self.decoder.stages))
        ]
        conv_op = self.decoder.encoder.conv_op
        self.guide_adapters = nn.ModuleList(
            [conv_op(1, channels, 1, 1, 0, bias=True) for channels in stage_channels]
        )
        for adapter in self.guide_adapters:
            nn.init.zeros_(adapter.weight)
            nn.init.zeros_(adapter.bias)

    def forward(
        self,
        skips: list[torch.Tensor],
        mask_embeddings: list[torch.Tensor],
        proposal_outputs: list[torch.Tensor],
        detach_proposal: bool,
    ) -> list[torch.Tensor]:
        if len(proposal_outputs) != len(self.decoder.stages):
            raise ValueError(
                f"Expected {len(self.decoder.stages)} proposal scales, "
                f"got {len(proposal_outputs)}"
            )

        lres_input = skips[-1]
        seg_outputs: list[torch.Tensor] = []
        mask_embeddings = list(mask_embeddings)[::-1]
        proposal_stage_outputs = list(proposal_outputs)[::-1]

        for stage_idx in range(len(self.decoder.stages)):
            x = self.decoder.transpconvs[stage_idx](lres_input)
            x = torch.cat((x, skips[-(stage_idx + 2)]), dim=1)
            x = self.decoder.stages[stage_idx](x)

            proposal_probability = torch.sigmoid(proposal_stage_outputs[stage_idx].float())
            if detach_proposal:
                proposal_probability = proposal_probability.detach()
            proposal_probability = proposal_probability.to(dtype=x.dtype)
            if tuple(proposal_probability.shape[2:]) != tuple(x.shape[2:]):
                raise ValueError(
                    f"Guide shape {tuple(proposal_probability.shape)} does not match "
                    f"decoder stage shape {tuple(x.shape)}"
                )
            x = x + self.guide_adapters[stage_idx](proposal_probability)

            if stage_idx == len(self.decoder.stages) - 1:
                seg_pred = torch.einsum(
                    "b c h w d, b n c -> b n h w d",
                    x,
                    mask_embeddings[-1],
                )
                seg_outputs.append(seg_pred)
            elif stage_idx >= len(self.decoder.stages) - len(mask_embeddings):
                mask_embedding = mask_embeddings.pop(0)
                batch_size, _, channels = mask_embedding.shape
                mask_embedding_reshaped = mask_embedding.view(
                    batch_size,
                    self.decoder.num_heads,
                    channels // self.decoder.num_heads,
                )
                fusion_features = torch.einsum(
                    "b c h w d, b n c -> b n h w d",
                    x,
                    mask_embedding_reshaped,
                )
                x = torch.cat((x, fusion_features), dim=1)
                seg_outputs.append(self.decoder.seg_layers[stage_idx](x))

            lres_input = x

        return seg_outputs[::-1]


class DualBranchVoxTellModel(nn.Module):
    """VoxTell with an inclusive proposal decoder and soft-guided refiner."""

    is_dual_branch_voxtell = True

    def __init__(
        self,
        source_model: nn.Module,
        variant: str,
        deep_supervision: bool = False,
        proposal_adapter_hidden_dim: int = 256,
        refinement_adapter_hidden_dim: int = 512,
    ) -> None:
        super().__init__()
        config = variant_config(variant)
        self.variant = variant
        self.fusion_mode = str(config["fusion_mode"])
        self.detach_proposal_guide = bool(config["detach_proposal_guide"])
        self.final_precision_weight = float(config["final_precision_weight"])
        self.deep_supervision = bool(deep_supervision)

        self.selected_decoder_layer = int(source_model.selected_decoder_layer)
        self.query_dim = int(source_model.query_dim)
        self.num_heads = int(source_model.num_heads)
        self.project_to_decoder_hidden_dim = int(
            source_model.project_to_decoder_hidden_dim
        )
        self.text_embedding_dim = int(source_model.text_embedding_dim)

        self.encoder = source_model.encoder
        self.project_bottleneck_embed = source_model.project_bottleneck_embed
        self.project_text_embed = source_model.project_text_embed
        self.transformer_decoder = source_model.transformer_decoder
        self.register_buffer("pos_embed", source_model.pos_embed.detach().clone())

        self.proposal_decoder = copy.deepcopy(
            source_model.decoder,
            memo={id(source_model.decoder.encoder): self.encoder},
        )
        self.proposal_decoder.deep_supervision = True
        self.refinement_decoder = GuidedVoxTellDecoder(
            source_model.decoder,
            shared_encoder=self.encoder,
        )

        if self.fusion_mode == "shared":
            self.shared_projectors = source_model.project_to_decoder_channels
            self.proposal_projectors = None
            self.refinement_projectors = None
            self.proposal_fusion_adapter = None
            self.refinement_fusion_adapter = None
        elif self.fusion_mode == "dual_adapter":
            self.shared_projectors = None
            self.proposal_projectors = copy.deepcopy(
                source_model.project_to_decoder_channels
            )
            self.refinement_projectors = source_model.project_to_decoder_channels
            self.proposal_fusion_adapter = ZeroResidualAdapter(
                input_dim=self.query_dim,
                output_dim=self.query_dim,
                hidden_dim=proposal_adapter_hidden_dim,
            )
            self.refinement_fusion_adapter = ZeroResidualAdapter(
                input_dim=2 * self.query_dim,
                output_dim=self.query_dim,
                hidden_dim=refinement_adapter_hidden_dim,
            )
        else:
            raise ValueError(f"Unsupported fusion mode: {self.fusion_mode}")

    def model_spec(self, source_model_dir: str | None = None) -> dict[str, Any]:
        return {
            "model_type": MODEL_TYPE,
            "schema_version": 1,
            "variant": self.variant,
            "fusion_mode": self.fusion_mode,
            "detach_proposal_guide": self.detach_proposal_guide,
            "final_precision_weight": self.final_precision_weight,
            "source_model_dir": source_model_dir,
            "proposal_adapter_hidden_dim": (
                int(self.proposal_fusion_adapter.layers[0].out_features)
                if self.proposal_fusion_adapter is not None
                else None
            ),
            "refinement_adapter_hidden_dim": (
                int(self.refinement_fusion_adapter.layers[0].out_features)
                if self.refinement_fusion_adapter is not None
                else None
            ),
        }

    def _shared_fusion(
        self,
        image: torch.Tensor,
        text_embedding: torch.Tensor,
    ) -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor]:
        skips = self.encoder(image)
        selected_feature = skips[self.selected_decoder_layer]
        memory_volume = rearrange(selected_feature, "b c d h w -> b h w d c")
        memory_volume = self.project_bottleneck_embed(memory_volume)
        bottleneck_embed = rearrange(
            memory_volume,
            "b h w d c -> (h w d) b c",
        )

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
        return skips, memory_volume, mask_embedding

    @staticmethod
    def _project_mask_embeddings(
        mask_embedding: torch.Tensor,
        projectors: nn.ModuleList,
    ) -> list[torch.Tensor]:
        return [projection(mask_embedding) for projection in projectors]

    @staticmethod
    def _decode_prompts(
        decoder: nn.Module,
        skips: list[torch.Tensor],
        mask_embeddings: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        per_prompt_outputs = []
        num_prompts = int(mask_embeddings[0].shape[1])
        for prompt_idx in range(num_prompts):
            prompt_embeddings = [
                embedding[:, prompt_idx : prompt_idx + 1]
                for embedding in mask_embeddings
            ]
            per_prompt_outputs.append(decoder(skips, prompt_embeddings))
        return [
            torch.cat(scale_outputs, dim=1)
            for scale_outputs in zip(*per_prompt_outputs)
        ]

    def _proposal_context(
        self,
        proposal_low_resolution: torch.Tensor,
        memory_volume: torch.Tensor,
    ) -> torch.Tensor:
        probability = torch.sigmoid(proposal_low_resolution.float())
        if self.detach_proposal_guide:
            probability = probability.detach()
        probability = rearrange(probability, "b n d h w -> b n h w d")
        if tuple(probability.shape[2:]) != tuple(memory_volume.shape[1:4]):
            raise ValueError(
                f"Proposal context shape {tuple(probability.shape[2:])} does not "
                f"match memory volume shape {tuple(memory_volume.shape[1:4])}"
            )
        weights = rearrange(probability, "b n h w d -> b n (h w d)")
        memory = rearrange(memory_volume.float(), "b h w d c -> b (h w d) c")
        weighted = torch.einsum("b n s, b s c -> b n c", weights, memory)
        denominator = weights.sum(dim=2, keepdim=True).clamp_min(1e-6)
        return weighted / denominator

    def forward_branches(
        self,
        image: torch.Tensor,
        text_embedding: torch.Tensor,
        return_all_scales: bool = True,
    ) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        skips, memory_volume, base_mask_embedding = self._shared_fusion(
            image,
            text_embedding,
        )

        if self.fusion_mode == "shared":
            proposal_mask_embedding = base_mask_embedding
            proposal_projectors = self.shared_projectors
        else:
            proposal_mask_embedding = (
                base_mask_embedding
                + self.proposal_fusion_adapter(base_mask_embedding)
            )
            proposal_projectors = self.proposal_projectors

        proposal_embeddings = self._project_mask_embeddings(
            proposal_mask_embedding,
            proposal_projectors,
        )
        proposal_outputs = self._decode_prompts(
            self.proposal_decoder,
            skips,
            proposal_embeddings,
        )

        if self.fusion_mode == "shared":
            refinement_mask_embedding = base_mask_embedding
            refinement_projectors = self.shared_projectors
        else:
            context = self._proposal_context(proposal_outputs[-1], memory_volume)
            if self.detach_proposal_guide:
                context = context.detach()
            adapter_input = torch.cat((base_mask_embedding, context), dim=-1)
            refinement_mask_embedding = (
                base_mask_embedding
                + self.refinement_fusion_adapter(adapter_input)
            )
            refinement_projectors = self.refinement_projectors

        refinement_embeddings = self._project_mask_embeddings(
            refinement_mask_embedding,
            refinement_projectors,
        )
        per_prompt_outputs = []
        num_prompts = int(refinement_embeddings[0].shape[1])
        for prompt_idx in range(num_prompts):
            prompt_embeddings = [
                embedding[:, prompt_idx : prompt_idx + 1]
                for embedding in refinement_embeddings
            ]
            prompt_proposals = [
                output[:, prompt_idx : prompt_idx + 1]
                for output in proposal_outputs
            ]
            per_prompt_outputs.append(
                self.refinement_decoder(
                    skips,
                    prompt_embeddings,
                    prompt_proposals,
                    detach_proposal=self.detach_proposal_guide,
                )
            )
        refinement_outputs = [
            torch.cat(scale_outputs, dim=1)
            for scale_outputs in zip(*per_prompt_outputs)
        ]

        if return_all_scales:
            return {
                "proposal": proposal_outputs,
                "final": refinement_outputs,
            }
        return {
            "proposal": proposal_outputs[0],
            "final": refinement_outputs[0],
        }

    def forward(
        self,
        image: torch.Tensor,
        text_embedding: torch.Tensor,
        return_branches: bool = False,
    ) -> (
        torch.Tensor
        | list[torch.Tensor]
        | dict[str, torch.Tensor | list[torch.Tensor]]
    ):
        branches = self.forward_branches(
            image,
            text_embedding,
            return_all_scales=self.deep_supervision or return_branches,
        )
        if return_branches:
            return branches
        final = branches["final"]
        if self.deep_supervision:
            return final
        if isinstance(final, list):
            return final[0]
        return final


def unwrap_dual_branch(network: nn.Module) -> DualBranchVoxTellModel | None:
    model = network.module if hasattr(network, "module") else network
    if isinstance(model, DualBranchVoxTellModel):
        return model
    return None


def is_dual_branch_model(network: nn.Module) -> bool:
    return unwrap_dual_branch(network) is not None


def model_spec_from_network(
    network: nn.Module,
    source_model_dir: str | None = None,
) -> dict[str, Any] | None:
    model = unwrap_dual_branch(network)
    if model is None:
        return None
    return model.model_spec(source_model_dir=source_model_dir)


def write_model_spec(path: Path, spec: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n")


def load_model_spec(model_dir: Path) -> dict[str, Any] | None:
    path = Path(model_dir) / MODEL_SPEC_FILENAME
    if not path.is_file():
        return None
    spec = json.loads(path.read_text())
    if spec.get("model_type") != MODEL_TYPE:
        raise ValueError(
            f"Unsupported model type in {path}: {spec.get('model_type')!r}"
        )
    variant_config(str(spec["variant"]))
    return spec


class DualBranchVoxTellPredictor(VoxTellPredictor):
    """VoxTell predictor that loads and aggregates both dual-branch outputs."""

    def __init__(
        self,
        model_dir: str | Path,
        device: torch.device,
        embedding_bank: str | dict[str, Any] | None = None,
        use_precomputed_embeddings: bool = True,
        sliding_window_batch_size: int = 1,
    ) -> None:
        model_dir = Path(model_dir)
        spec = load_model_spec(model_dir)
        if spec is None:
            raise ValueError(f"Missing {MODEL_SPEC_FILENAME} under {model_dir}")
        source_model_dir = spec.get("source_model_dir")
        if not source_model_dir:
            raise ValueError(
                f"{model_dir / MODEL_SPEC_FILENAME}: source_model_dir is required"
            )
        super().__init__(
            model_dir=str(source_model_dir),
            device=device,
            embedding_bank=embedding_bank,
            use_precomputed_embeddings=use_precomputed_embeddings,
        )
        source_network = self.network
        network = DualBranchVoxTellModel(
            source_model=source_network,
            variant=str(spec["variant"]),
            deep_supervision=False,
            proposal_adapter_hidden_dim=int(
                spec.get("proposal_adapter_hidden_dim") or 256
            ),
            refinement_adapter_hidden_dim=int(
                spec.get("refinement_adapter_hidden_dim") or 512
            ),
        )
        checkpoint = torch.load(
            model_dir / "fold_0" / "checkpoint_final.pth",
            map_location=torch.device("cpu"),
            weights_only=False,
        )
        checkpoint_spec = checkpoint.get("model_spec")
        if checkpoint_spec is not None and checkpoint_spec.get("variant") != spec.get(
            "variant"
        ):
            raise ValueError(
                f"Checkpoint variant {checkpoint_spec.get('variant')!r} does not "
                f"match model spec variant {spec.get('variant')!r}"
            )
        network.load_state_dict(checkpoint["network_weights"], strict=True)
        network.deep_supervision = False
        network.eval()
        self.network = network
        self.sliding_window_batch_size = int(sliding_window_batch_size)
        if self.sliding_window_batch_size < 1:
            raise ValueError("sliding_window_batch_size must be >= 1")

    @torch.inference_mode()
    def predict_sliding_window_return_branch_logits(
        self,
        input_image: torch.Tensor,
        text_embeddings: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if not isinstance(input_image, torch.Tensor) or input_image.ndim != 4:
            raise ValueError(
                f"input_image must be a 4D torch.Tensor, got "
                f"{type(input_image)} shape={getattr(input_image, 'shape', None)}"
            )
        self.network = self.network.to(self.device)
        empty_cache(self.device)
        autocast_context = (
            torch.autocast(self.device.type, enabled=True)
            if self.device.type == "cuda"
            else dummy_context()
        )
        with autocast_context:
            data, slicer_revert_padding = pad_nd_image(
                input_image,
                self.patch_size,
                "constant",
                {"value": 0},
                True,
                None,
            )
            slicers = self._internal_get_sliding_window_slicers(data.shape[1:])
            branch_logits = self._predict_branch_slicers(
                data,
                text_embeddings,
                slicers,
                self.perform_everything_on_device,
            )
            branch_logits = {
                branch: logits[(slice(None), *slicer_revert_padding[1:])]
                for branch, logits in branch_logits.items()
            }
        empty_cache(self.device)
        return branch_logits

    @torch.inference_mode()
    def predict_sliding_window_return_logits(
        self,
        input_image: torch.Tensor,
        text_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        return self.predict_sliding_window_return_branch_logits(
            input_image,
            text_embeddings,
        )["final"]

    def _predict_branch_slicers(
        self,
        data: torch.Tensor,
        text_embeddings: torch.Tensor,
        slicers: list[tuple],
        do_on_device: bool,
    ) -> dict[str, torch.Tensor]:
        results_device = self.device if do_on_device else torch.device("cpu")
        data = data.to(results_device)
        num_prompts = int(text_embeddings.shape[1])
        predicted = {
            branch: torch.zeros(
                (num_prompts, *data.shape[1:]),
                dtype=torch.half,
                device=results_device,
            )
            for branch in ("proposal", "final")
        }
        n_predictions = torch.zeros(
            data.shape[1:],
            dtype=torch.half,
            device=results_device,
        )
        gaussian = compute_gaussian(
            tuple(self.patch_size),
            sigma_scale=1.0 / 8,
            value_scaling_factor=10,
            device=results_device,
        )

        batch_size = self.sliding_window_batch_size
        with tqdm(total=len(slicers), desc=None) as progress:
            for start in range(0, len(slicers), batch_size):
                tile_slices = slicers[start : start + batch_size]
                patches = torch.cat(
                    [
                        torch.clone(
                            data[tile_slice][None],
                            memory_format=torch.contiguous_format,
                        )
                        for tile_slice in tile_slices
                    ],
                    dim=0,
                ).to(self.device)
                batch_embeddings = text_embeddings
                if int(text_embeddings.shape[0]) == 1 and len(tile_slices) > 1:
                    batch_embeddings = text_embeddings.expand(
                        len(tile_slices),
                        *text_embeddings.shape[1:],
                    )
                elif int(text_embeddings.shape[0]) != len(tile_slices):
                    raise ValueError(
                        f"Text embedding batch {text_embeddings.shape[0]} cannot "
                        f"serve window batch {len(tile_slices)}"
                    )
                outputs = self.network(
                    patches,
                    batch_embeddings,
                    return_branches=True,
                )
                for branch in ("proposal", "final"):
                    branch_output = outputs[branch]
                    if isinstance(branch_output, list):
                        branch_output = branch_output[0]
                    branch_output = branch_output.to(results_device)
                    for batch_index, tile_slice in enumerate(tile_slices):
                        predicted[branch][tile_slice] += (
                            branch_output[batch_index] * gaussian
                        )
                for tile_slice in tile_slices:
                    n_predictions[tile_slice[1:]] += gaussian
                progress.update(len(tile_slices))

        for branch in predicted:
            torch.div(predicted[branch], n_predictions, out=predicted[branch])
            if torch.any(torch.isinf(predicted[branch])) or torch.any(
                torch.isnan(predicted[branch])
            ):
                raise RuntimeError(
                    f"Encountered non-finite {branch} sliding-window logits"
                )
        return predicted
