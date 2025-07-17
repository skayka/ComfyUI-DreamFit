# Copyright (c) 2024 Black Forest Labs and The XLabs-AI Team.
# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0 
#
# This file has been modified by Bytedance Ltd. and/or its affiliates on 2025/5/6.
#
# Original file was released under Apache-2.0, with the full license text
# available at https://github.com/XLabs-AI/x-flux/blob/main/LICENSE.
#
# This modified file is released under the same license.

import os
from dataclasses import dataclass
import importlib
import functools
from tqdm import tqdm
import gc

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy, transformer_auto_wrap_policy
from torch.distributed.fsdp.api import (
    ShardingStrategy,
    BackwardPrefetch,
    MixedPrecision,
)

import json
import cv2
import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from safetensors.torch import load_file as load_sft

from typing import Optional, OrderedDict, Tuple, TypeAlias, Union
from loguru import logger
# import rearrange
 
from .model_dreamfit import Flux, FluxParams
# from .model_ipa_kv import Flux, FluxParams
from .controlnet import ControlNetFlux
from .modules.autoencoder import AutoEncoder, AutoEncoderParams
from .modules.conditioner import HFEmbedder
from .annotator.dwpose import DWposeDetector
from .annotator.mlsd import MLSDdetector
from .annotator.canny import CannyDetector
from .annotator.midas import MidasDetector
from .annotator.hed import HEDdetector
from .annotator.tile import TileDetector

from .lora import LoRACompatibleConv, LoRACompatibleLinear, LoRAConv2dLayer, LoRALinearLayer
from diffusers.models.lora import LoRACompatibleConv as LoRACompatibleConv_
from diffusers.models.lora import LoRACompatibleLinear as LoRACompatibleLinear_

def get_class(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

def load_safetensors(path):
    tensors = {}
    with safe_open(path, framework="pt", device="cpu") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
    return tensors

def get_lora_rank(checkpoint):
    for k in checkpoint.keys():
        if k.endswith(".down.weight"):
            return checkpoint[k].shape[0]

def load_checkpoint(local_path, repo_id, name):
    if local_path is not None:
        if '.safetensors' in local_path:
            print(f"Loading .safetensors checkpoint from {local_path}")
            checkpoint = load_safetensors(local_path)
        else:
            print(f"Loading checkpoint from {local_path}")
            checkpoint = torch.load(local_path, map_location='cpu')
    elif repo_id is not None and name is not None:
        print(f"Loading checkpoint {name} from repo id {repo_id}")
        checkpoint = load_from_repo_id(repo_id, name)
    else:
        raise ValueError(
            "LOADING ERROR: you must specify local_path or repo_id with name in HF to download"
        )
    return checkpoint

StateDict: TypeAlias = OrderedDict[str, torch.Tensor]

class LoraWeights:
    def __init__(
        self,
        weights: StateDict,
        path: str,
        name: str = None,
        scale: float = 1.0,
    ) -> None:
        self.path = path
        self.weights = weights
        self.name = name if name else path_regex.split(path)[-1]
        self.scale = scale

def get_lora_weights(lora_path: str | StateDict):
    if isinstance(lora_path, (dict, LoraWeights)):
        return lora_path, True
    else:
        return load_sft(lora_path, "cpu"), False

def get_module_for_key(
    key: str, model
):
    parts = key.split(".")
    module = model
    for part in parts:
        module = getattr(module, part)
    return module

def resolve_lora_state_dict(lora_weights, has_guidance: bool = True):
    check_if_starts_with_transformer = [
        k for k in lora_weights.keys() if k.startswith("transformer.")
    ]
    if len(check_if_starts_with_transformer) > 0:
        lora_weights = convert_diffusers_to_flux_transformer_checkpoint(
            lora_weights, 19, 38, has_guidance=has_guidance, prefix="transformer."
        )
    else:
        lora_weights = convert_from_original_flux_checkpoint(lora_weights)
    logger.info("LoRA weights loaded")
    logger.debug("Extracting keys")
    keys_without_ab = list(
        set(
            [
                key.replace(".lora_A.weight", "")
                .replace(".lora_B.weight", "")
                .replace(".lora_A", "")
                .replace(".lora_B", "")
                .replace(".alpha", "")
                for key in lora_weights.keys()
            ]
        )
    )
    logger.debug("Keys extracted")
    return keys_without_ab, lora_weights

def convert_diffusers_to_flux_transformer_checkpoint(
    diffusers_state_dict,
    num_layers,
    num_single_layers,
    has_guidance=True,
    prefix="",
):
    original_state_dict = {}

    # time_text_embed.timestep_embedder -> time_in
    original_state_dict, diffusers_state_dict = convert_if_lora_exists(
        original_state_dict,
        diffusers_state_dict,
        f"{prefix}time_text_embed.timestep_embedder.linear_1.weight",
        "time_in.in_layer.weight",
    )
    # time_text_embed.text_embedder -> vector_in
    original_state_dict, diffusers_state_dict = convert_if_lora_exists(
        original_state_dict,
        diffusers_state_dict,
        f"{prefix}time_text_embed.text_embedder.linear_1.weight",
        "vector_in.in_layer.weight",
    )

    original_state_dict, diffusers_state_dict = convert_if_lora_exists(
        original_state_dict,
        diffusers_state_dict,
        f"{prefix}time_text_embed.text_embedder.linear_2.weight",
        "vector_in.out_layer.weight",
    )

    if has_guidance:
        original_state_dict, diffusers_state_dict = convert_if_lora_exists(
            original_state_dict,
            diffusers_state_dict,
            f"{prefix}time_text_embed.guidance_embedder.linear_1.weight",
            "guidance_in.in_layer.weight",
        )

        original_state_dict, diffusers_state_dict = convert_if_lora_exists(
            original_state_dict,
            diffusers_state_dict,
            f"{prefix}time_text_embed.guidance_embedder.linear_2.weight",
            "guidance_in.out_layer.weight",
        )

    # context_embedder -> txt_in
    original_state_dict, diffusers_state_dict = convert_if_lora_exists(
        original_state_dict,
        diffusers_state_dict,
        f"{prefix}context_embedder.weight",
        "txt_in.weight",
    )

    # x_embedder -> img_in
    original_state_dict, diffusers_state_dict = convert_if_lora_exists(
        original_state_dict,
        diffusers_state_dict,
        f"{prefix}x_embedder.weight",
        "img_in.weight",
    )
    # double transformer blocks
    for i in range(num_layers):
        block_prefix = f"transformer_blocks.{i}."
        # norms
        original_state_dict, diffusers_state_dict = convert_if_lora_exists(
            original_state_dict,
            diffusers_state_dict,
            f"{prefix}{block_prefix}norm1.linear.weight",
            f"double_blocks.{i}.img_mod.lin.weight",
        )

        original_state_dict, diffusers_state_dict = convert_if_lora_exists(
            original_state_dict,
            diffusers_state_dict,
            f"{prefix}{block_prefix}norm1_context.linear.weight",
            f"double_blocks.{i}.txt_mod.lin.weight",
        )

        # Q, K, V
        temp_dict = {}

        expected_shape_qkv_a = None
        expected_shape_qkv_b = None
        expected_shape_add_qkv_a = None
        expected_shape_add_qkv_b = None
        dtype = None
        device = None

        for component in [
            "to_q",
            "to_k",
            "to_v",
            "add_q_proj",
            "add_k_proj",
            "add_v_proj",
        ]:

            sample_component_A_key = (
                f"{prefix}{block_prefix}attn.{component}.lora_A.weight"
            )
            sample_component_B_key = (
                f"{prefix}{block_prefix}attn.{component}.lora_B.weight"
            )
            if (
                sample_component_A_key in diffusers_state_dict
                and sample_component_B_key in diffusers_state_dict
            ):
                sample_component_A = diffusers_state_dict.pop(sample_component_A_key)
                sample_component_B = diffusers_state_dict.pop(sample_component_B_key)
                temp_dict[f"{component}"] = [sample_component_A, sample_component_B]
                if expected_shape_qkv_a is None and not component.startswith("add_"):
                    expected_shape_qkv_a = sample_component_A.shape
                    expected_shape_qkv_b = sample_component_B.shape
                    dtype = sample_component_A.dtype
                    device = sample_component_A.device
                if expected_shape_add_qkv_a is None and component.startswith("add_"):
                    expected_shape_add_qkv_a = sample_component_A.shape
                    expected_shape_add_qkv_b = sample_component_B.shape
                    dtype = sample_component_A.dtype
                    device = sample_component_A.device
            else:
                logger.info(
                    f"Skipping layer {i} since no LoRA weight is available for {sample_component_A_key}"
                )
                temp_dict[f"{component}"] = [None, None]

        if device is not None:
            if expected_shape_qkv_a is not None:

                if (sq := temp_dict["to_q"])[0] is not None:
                    sample_q_A, sample_q_B = sq
                else:
                    sample_q_A, sample_q_B = [
                        torch.zeros(expected_shape_qkv_a, dtype=dtype, device=device),
                        torch.zeros(expected_shape_qkv_b, dtype=dtype, device=device),
                    ]
                if (sq := temp_dict["to_k"])[0] is not None:
                    sample_k_A, sample_k_B = sq
                else:
                    sample_k_A, sample_k_B = [
                        torch.zeros(expected_shape_qkv_a, dtype=dtype, device=device),
                        torch.zeros(expected_shape_qkv_b, dtype=dtype, device=device),
                    ]
                if (sq := temp_dict["to_v"])[0] is not None:
                    sample_v_A, sample_v_B = sq
                else:
                    sample_v_A, sample_v_B = [
                        torch.zeros(expected_shape_qkv_a, dtype=dtype, device=device),
                        torch.zeros(expected_shape_qkv_b, dtype=dtype, device=device),
                    ]
                original_state_dict[f"double_blocks.{i}.img_attn.qkv.lora_A.weight"] = (
                    torch.cat([sample_q_A, sample_k_A, sample_v_A], dim=0)
                )
                original_state_dict[f"double_blocks.{i}.img_attn.qkv.lora_B.weight"] = (
                    torch.cat([sample_q_B, sample_k_B, sample_v_B], dim=0)
                )
            if expected_shape_add_qkv_a is not None:

                if (sq := temp_dict["add_q_proj"])[0] is not None:
                    context_q_A, context_q_B = sq
                else:
                    context_q_A, context_q_B = [
                        torch.zeros(
                            expected_shape_add_qkv_a, dtype=dtype, device=device
                        ),
                        torch.zeros(
                            expected_shape_add_qkv_b, dtype=dtype, device=device
                        ),
                    ]
                if (sq := temp_dict["add_k_proj"])[0] is not None:
                    context_k_A, context_k_B = sq
                else:
                    context_k_A, context_k_B = [
                        torch.zeros(
                            expected_shape_add_qkv_a, dtype=dtype, device=device
                        ),
                        torch.zeros(
                            expected_shape_add_qkv_b, dtype=dtype, device=device
                        ),
                    ]
                if (sq := temp_dict["add_v_proj"])[0] is not None:
                    context_v_A, context_v_B = sq
                else:
                    context_v_A, context_v_B = [
                        torch.zeros(
                            expected_shape_add_qkv_a, dtype=dtype, device=device
                        ),
                        torch.zeros(
                            expected_shape_add_qkv_b, dtype=dtype, device=device
                        ),
                    ]

                original_state_dict[f"double_blocks.{i}.txt_attn.qkv.lora_A.weight"] = (
                    torch.cat([context_q_A, context_k_A, context_v_A], dim=0)
                )
                original_state_dict[f"double_blocks.{i}.txt_attn.qkv.lora_B.weight"] = (
                    torch.cat([context_q_B, context_k_B, context_v_B], dim=0)
                )

        # qk_norm
        original_state_dict, diffusers_state_dict = convert_if_lora_exists(
            original_state_dict,
            diffusers_state_dict,
            f"{prefix}{block_prefix}attn.norm_q.weight",
            f"double_blocks.{i}.img_attn.norm.query_norm.scale",
        )

        original_state_dict, diffusers_state_dict = convert_if_lora_exists(
            original_state_dict,
            diffusers_state_dict,
            f"{prefix}{block_prefix}attn.norm_k.weight",
            f"double_blocks.{i}.img_attn.norm.key_norm.scale",
        )
        original_state_dict, diffusers_state_dict = convert_if_lora_exists(
            original_state_dict,
            diffusers_state_dict,
            f"{prefix}{block_prefix}attn.norm_added_q.weight",
            f"double_blocks.{i}.txt_attn.norm.query_norm.scale",
        )
        original_state_dict, diffusers_state_dict = convert_if_lora_exists(
            original_state_dict,
            diffusers_state_dict,
            f"{prefix}{block_prefix}attn.norm_added_k.weight",
            f"double_blocks.{i}.txt_attn.norm.key_norm.scale",
        )

        # ff img_mlp

        original_state_dict, diffusers_state_dict = convert_if_lora_exists(
            original_state_dict,
            diffusers_state_dict,
            f"{prefix}{block_prefix}ff.net.0.proj.weight",
            f"double_blocks.{i}.img_mlp.0.weight",
        )
        original_state_dict, diffusers_state_dict = convert_if_lora_exists(
            original_state_dict,
            diffusers_state_dict,
            f"{prefix}{block_prefix}ff.net.2.weight",
            f"double_blocks.{i}.img_mlp.2.weight",
        )
        original_state_dict, diffusers_state_dict = convert_if_lora_exists(
            original_state_dict,
            diffusers_state_dict,
            f"{prefix}{block_prefix}ff_context.net.0.proj.weight",
            f"double_blocks.{i}.txt_mlp.0.weight",
        )

        original_state_dict, diffusers_state_dict = convert_if_lora_exists(
            original_state_dict,
            diffusers_state_dict,
            f"{prefix}{block_prefix}ff_context.net.2.weight",
            f"double_blocks.{i}.txt_mlp.2.weight",
        )
        # output projections
        original_state_dict, diffusers_state_dict = convert_if_lora_exists(
            original_state_dict,
            diffusers_state_dict,
            f"{prefix}{block_prefix}attn.to_out.0.weight",
            f"double_blocks.{i}.img_attn.proj.weight",
        )

        original_state_dict, diffusers_state_dict = convert_if_lora_exists(
            original_state_dict,
            diffusers_state_dict,
            f"{prefix}{block_prefix}attn.to_add_out.weight",
            f"double_blocks.{i}.txt_attn.proj.weight",
        )

    # single transformer blocks
    for i in range(num_single_layers):
        block_prefix = f"single_transformer_blocks.{i}."
        # norm.linear -> single_blocks.0.modulation.lin
        key_norm = f"{prefix}{block_prefix}norm.linear.weight"
        original_state_dict, diffusers_state_dict = convert_if_lora_exists(
            original_state_dict,
            diffusers_state_dict,
            key_norm,
            f"single_blocks.{i}.modulation.lin.weight",
        )

        has_q, has_k, has_v, has_mlp = False, False, False, False
        shape_qkv_a = None
        shape_qkv_b = None
        # Q, K, V, mlp
        q_A = diffusers_state_dict.pop(f"{prefix}{block_prefix}attn.to_q.lora_A.weight")
        q_B = diffusers_state_dict.pop(f"{prefix}{block_prefix}attn.to_q.lora_B.weight")
        if q_A is not None and q_B is not None:
            has_q = True
            shape_qkv_a = q_A.shape
            shape_qkv_b = q_B.shape
        k_A = diffusers_state_dict.pop(f"{prefix}{block_prefix}attn.to_k.lora_A.weight")
        k_B = diffusers_state_dict.pop(f"{prefix}{block_prefix}attn.to_k.lora_B.weight")
        if k_A is not None and k_B is not None:
            has_k = True
            shape_qkv_a = k_A.shape
            shape_qkv_b = k_B.shape
        v_A = diffusers_state_dict.pop(f"{prefix}{block_prefix}attn.to_v.lora_A.weight")
        v_B = diffusers_state_dict.pop(f"{prefix}{block_prefix}attn.to_v.lora_B.weight")
        if v_A is not None and v_B is not None:
            has_v = True
            shape_qkv_a = v_A.shape
            shape_qkv_b = v_B.shape
        mlp_A = diffusers_state_dict.pop(
            f"{prefix}{block_prefix}proj_mlp.lora_A.weight"
        )
        mlp_B = diffusers_state_dict.pop(
            f"{prefix}{block_prefix}proj_mlp.lora_B.weight"
        )
        if mlp_A is not None and mlp_B is not None:
            has_mlp = True
            shape_qkv_a = mlp_A.shape
            shape_qkv_b = mlp_B.shape
        if any([has_q, has_k, has_v, has_mlp]):
            if not has_q:
                q_A, q_B = [
                    torch.zeros(shape_qkv_a, dtype=dtype, device=device),
                    torch.zeros(shape_qkv_b, dtype=dtype, device=device),
                ]
            if not has_k:
                k_A, k_B = [
                    torch.zeros(shape_qkv_a, dtype=dtype, device=device),
                    torch.zeros(shape_qkv_b, dtype=dtype, device=device),
                ]
            if not has_v:
                v_A, v_B = [
                    torch.zeros(shape_qkv_a, dtype=dtype, device=device),
                    torch.zeros(shape_qkv_b, dtype=dtype, device=device),
                ]
            if not has_mlp:
                mlp_A, mlp_B = [
                    torch.zeros(shape_qkv_a, dtype=dtype, device=device),
                    torch.zeros(shape_qkv_b, dtype=dtype, device=device),
                ]
            original_state_dict[f"single_blocks.{i}.linear1.lora_A.weight"] = torch.cat(
                [q_A, k_A, v_A, mlp_A], dim=0
            )
            original_state_dict[f"single_blocks.{i}.linear1.lora_B.weight"] = torch.cat(
                [q_B, k_B, v_B, mlp_B], dim=0
            )

        # output projections
        original_state_dict, diffusers_state_dict = convert_if_lora_exists(
            original_state_dict,
            diffusers_state_dict,
            f"{prefix}{block_prefix}proj_out.weight",
            f"single_blocks.{i}.linear2.weight",
        )

    original_state_dict, diffusers_state_dict = convert_if_lora_exists(
        original_state_dict,
        diffusers_state_dict,
        f"{prefix}proj_out.weight",
        "final_layer.linear.weight",
    )
    original_state_dict, diffusers_state_dict = convert_if_lora_exists(
        original_state_dict,
        diffusers_state_dict,
        f"{prefix}proj_out.bias",
        "final_layer.linear.bias",
    )
    original_state_dict, diffusers_state_dict = convert_if_lora_exists(
        original_state_dict,
        diffusers_state_dict,
        f"{prefix}norm_out.linear.weight",
        "final_layer.adaLN_modulation.1.weight",
    )
    if len(list(diffusers_state_dict.keys())) > 0:
        logger.warning("Unexpected keys:", diffusers_state_dict.keys())

    return original_state_dict

def get_lora_for_key(
    key: str, lora_weights: dict
) -> Optional[Tuple[torch.Tensor, torch.Tensor, Optional[float]]]:
    """
    Get LoRA weights for a specific key.

    Args:
        key (str): The key to look up in the LoRA weights.
        lora_weights (dict): Dictionary containing LoRA weights.

    Returns:
        Optional[Tuple[torch.Tensor, torch.Tensor, Optional[float]]]: A tuple containing lora_A, lora_B, and alpha if found, None otherwise.
    """
    prefix = key.split(".lora")[0]
    lora_A = lora_weights.get(f"{prefix}.lora_A.weight")
    lora_B = lora_weights.get(f"{prefix}.lora_B.weight")
    alpha = lora_weights.get(f"{prefix}.alpha")

    if lora_A is None or lora_B is None:
        return None
    return lora_A, lora_B, alpha

def extract_weight_from_linear(linear):
    dtype = linear.weight.dtype
    weight_is_f8 = False
    # if isinstance(linear, F8Linear):
    #     weight_is_f8 = True
    #     weight = (
    #         linear.float8_data.clone()
    #         .detach()
    #         .float()
    #         .mul(linear.scale_reciprocal)
    #         .to(linear.weight.device)
    #     )
    if isinstance(linear, torch.nn.Linear):
        weight = linear.weight.clone().detach().float()
    # elif isinstance(linear, CublasLinear):
    #     weight = linear.weight.clone().detach().float()
    return weight, weight_is_f8, dtype

def calculate_lora_weight(
    lora_weights: Tuple[torch.Tensor, torch.Tensor, Union[torch.Tensor, float]],
    rank: Optional[int] = None,
    lora_scale: float = 1.0,
    device: Optional[Union[torch.device, int, str]] = None,
):
    lora_A, lora_B, alpha = lora_weights
    if device is None:
        device = lora_A.device

    uneven_rank = lora_B.shape[1] != lora_A.shape[0]
    rank_diff = lora_A.shape[0] / lora_B.shape[1]

    if rank is None:
        rank = lora_B.shape[1]
    if alpha is None:
        alpha = rank

    dtype = torch.float32
    w_up = lora_A.to(dtype=dtype, device=device)
    w_down = lora_B.to(dtype=dtype, device=device)

    if alpha != rank:
        w_up = w_up * alpha / rank
    if uneven_rank:
        # Fuse each lora instead of repeat interleave for each individual lora,
        # seems to fuse more correctly.
        fused_lora = torch.zeros(
            (lora_B.shape[0], lora_A.shape[1]), device=device, dtype=dtype
        )
        w_up = w_up.chunk(int(rank_diff), dim=0)
        for w_up_chunk in w_up:
            fused_lora = fused_lora + (lora_scale * torch.mm(w_down, w_up_chunk))
    else:
        fused_lora = lora_scale * torch.mm(w_down, w_up)
    return fused_lora

@torch.inference_mode()
def apply_lora_weight_to_module(
    module_weight: torch.Tensor,
    lora_weights: dict,
    rank: int = None,
    lora_scale: float = 1.0,
):
    w_dtype = module_weight.dtype
    dtype = torch.float32
    device = module_weight.device

    fused_lora = calculate_lora_weight(lora_weights, rank, lora_scale, device=device)
    fused_weight = module_weight.to(dtype=dtype) + fused_lora
    return fused_weight.to(dtype=w_dtype, device=device)

@torch.inference_mode()
def apply_lora_to_model(
    model,
    lora_path: str | StateDict,
    lora_scale: float = 1.0,
    return_lora_resolved: bool = False,
):
    has_guidance = model.params.guidance_embed
    logger.info(f"Loading LoRA weights for {lora_path}")
    lora_weights, already_loaded = get_lora_weights(lora_path)

    if not already_loaded:
        keys_without_ab, lora_weights = resolve_lora_state_dict(
            lora_weights, has_guidance
        )
    elif isinstance(lora_weights, LoraWeights):
        b_ = lora_weights
        lora_weights = b_.weights
        keys_without_ab = list(
            set(
                [
                    key.replace(".lora_A.weight", "")
                    .replace(".lora_B.weight", "")
                    .replace(".lora_A", "")
                    .replace(".lora_B", "")
                    .replace(".alpha", "")
                    for key in lora_weights.keys()
                ]
            )
        )
    else:
        lora_weights = lora_weights
        keys_without_ab = list(
            set(
                [
                    key.replace(".lora_A.weight", "")
                    .replace(".lora_B.weight", "")
                    .replace(".lora_A", "")
                    .replace(".lora_B", "")
                    .replace(".alpha", "")
                    for key in lora_weights.keys()
                ]
            )
        )
    for key in tqdm(keys_without_ab, desc="Applying LoRA", total=len(keys_without_ab)):
        module = get_module_for_key(key, model)
        weight, is_f8, dtype = extract_weight_from_linear(module)
        lora_sd = get_lora_for_key(key, lora_weights)
        if lora_sd is None:
            # Skipping LoRA application for this module
            continue
        weight = apply_lora_weight_to_module(weight, lora_sd, lora_scale=lora_scale)
        if is_f8:
            module.set_weight_tensor(weight.type(dtype))
        else:
            module.weight.data = weight.type(dtype)
    logger.success("Lora applied")
    if return_lora_resolved:
        return model, lora_weights
    return model

def c_crop(image):
    width, height = image.size
    new_size = min(width, height)
    left = (width - new_size) / 2
    top = (height - new_size) / 2
    right = (width + new_size) / 2
    bottom = (height + new_size) / 2
    return image.crop((left, top, right, bottom))

def pad64(x):
    return int(np.ceil(float(x) / 64.0) * 64 - x)

def HWC3(x):
    assert x.dtype == np.uint8
    if x.ndim == 2:
        x = x[:, :, None]
    assert x.ndim == 3
    H, W, C = x.shape
    assert C == 1 or C == 3 or C == 4
    if C == 3:
        return x
    if C == 1:
        return np.concatenate([x, x, x], axis=2)
    if C == 4:
        color = x[:, :, 0:3].astype(np.float32)
        alpha = x[:, :, 3:4].astype(np.float32) / 255.0
        y = color * alpha + 255.0 * (1.0 - alpha)
        y = y.clip(0, 255).astype(np.uint8)
        return y

def safer_memory(x):
    # Fix many MAC/AMD problems
    return np.ascontiguousarray(x.copy()).copy()

#https://github.com/Mikubill/sd-webui-controlnet/blob/main/scripts/processor.py#L17
#Added upscale_method, mode params
def resize_image_with_pad(input_image, resolution, skip_hwc3=False, mode='edge'):
    if skip_hwc3:
        img = input_image
    else:
        img = HWC3(input_image)
    H_raw, W_raw, _ = img.shape
    if resolution == 0:
        return img, lambda x: x
    k = float(resolution) / float(min(H_raw, W_raw))
    H_target = int(np.round(float(H_raw) * k))
    W_target = int(np.round(float(W_raw) * k))
    img = cv2.resize(img, (W_target, H_target), interpolation=cv2.INTER_AREA)
    H_pad, W_pad = pad64(H_target), pad64(W_target)
    img_padded = np.pad(img, [[0, H_pad], [0, W_pad], [0, 0]], mode=mode)

    def remove_pad(x):
        return safer_memory(x[:H_target, :W_target, ...])

    return safer_memory(img_padded), remove_pad

class Annotator:
    def __init__(self, name: str, device: str):
        if name == "canny":
            processor = CannyDetector()
        elif name == "openpose":
            processor = DWposeDetector(device)
        elif name == "depth":
            processor = MidasDetector()
        elif name == "hed":
            processor = HEDdetector()
        elif name == "hough":
            processor = MLSDdetector()
        elif name == "tile":
            processor = TileDetector()
        self.name = name
        self.processor = processor

    def __call__(self, image: Image, width: int, height: int):
        image = np.array(image)
        detect_resolution = max(width, height)
        image, remove_pad = resize_image_with_pad(image, detect_resolution)

        image = np.array(image)
        if self.name == "canny":
            result = self.processor(image, low_threshold=100, high_threshold=200)
        elif self.name == "hough":
            result = self.processor(image, thr_v=0.05, thr_d=5)
        elif self.name == "depth":
            result = self.processor(image)
            result, _ = result
        else:
            result = self.processor(image)

        result = HWC3(remove_pad(result))
        result = cv2.resize(result, (width, height))
        return result


@dataclass
class ModelSpec:
    params: FluxParams
    ae_params: AutoEncoderParams
    ckpt_path: str | None
    ae_path: str | None
    repo_id: str | None
    repo_flow: str | None
    repo_ae: str | None
    repo_id_ae: str | None


configs = {
    "flux-dev": ModelSpec(
        repo_id="black-forest-labs/FLUX.1-dev",
        repo_id_ae="black-forest-labs/FLUX.1-dev",
        repo_flow="flux1-dev.safetensors",
        repo_ae="ae.safetensors",
        ckpt_path=os.getenv("FLUX_DEV"),
        params=FluxParams(
            in_channels=64,
            vec_in_dim=768,
            context_in_dim=4096,
            hidden_size=3072,
            mlp_ratio=4.0,
            num_heads=24,
            depth=19,
            depth_single_blocks=38,
            axes_dim=[16, 56, 56],
            theta=10_000,
            qkv_bias=True,
            guidance_embed=True,
        ),
        ae_path=os.getenv("AE"),
        ae_params=AutoEncoderParams(
            resolution=256,
            in_channels=3,
            ch=128,
            out_ch=3,
            ch_mult=[1, 2, 4, 4],
            num_res_blocks=2,
            z_channels=16,
            scale_factor=0.3611,
            shift_factor=0.1159,
        ),
    ),
    "flux-dev-fp8": ModelSpec(
        repo_id="XLabs-AI/flux-dev-fp8",
        repo_id_ae="black-forest-labs/FLUX.1-dev",
        repo_flow="flux-dev-fp8.safetensors",
        repo_ae="ae.safetensors",
        ckpt_path=os.getenv("FLUX_DEV_FP8"),
        params=FluxParams(
            in_channels=64,
            vec_in_dim=768,
            context_in_dim=4096,
            hidden_size=3072,
            mlp_ratio=4.0,
            num_heads=24,
            depth=19,
            depth_single_blocks=38,
            axes_dim=[16, 56, 56],
            theta=10_000,
            qkv_bias=True,
            guidance_embed=True,
        ),
        ae_path=os.getenv("AE"),
        ae_params=AutoEncoderParams(
            resolution=256,
            in_channels=3,
            ch=128,
            out_ch=3,
            ch_mult=[1, 2, 4, 4],
            num_res_blocks=2,
            z_channels=16,
            scale_factor=0.3611,
            shift_factor=0.1159,
        ),
    ),
    "flux-schnell": ModelSpec(
        repo_id="black-forest-labs/FLUX.1-schnell",
        repo_id_ae="black-forest-labs/FLUX.1-dev",
        repo_flow="flux1-schnell.safetensors",
        repo_ae="ae.safetensors",
        ckpt_path=os.getenv("FLUX_SCHNELL"),
        params=FluxParams(
            in_channels=64,
            vec_in_dim=768,
            context_in_dim=4096,
            hidden_size=3072,
            mlp_ratio=4.0,
            num_heads=24,
            depth=19,
            depth_single_blocks=38,
            axes_dim=[16, 56, 56],
            theta=10_000,
            qkv_bias=True,
            guidance_embed=False,
        ),
        ae_path=os.getenv("AE"),
        ae_params=AutoEncoderParams(
            resolution=256,
            in_channels=3,
            ch=128,
            out_ch=3,
            ch_mult=[1, 2, 4, 4],
            num_res_blocks=2,
            z_channels=16,
            scale_factor=0.3611,
            shift_factor=0.1159,
        ),
    ),
}

def swap_scale_shift(weight):
    scale, shift = weight.chunk(2, dim=0)
    new_weight = torch.cat([shift, scale], dim=0)
    return new_weight


def check_if_lora_exists(state_dict, lora_name):
    subkey = lora_name.split(".lora_A")[0].split(".lora_B")[0].split(".weight")[0]
    for key in state_dict.keys():
        if subkey in key:
            return subkey
    return False


def convert_if_lora_exists(new_state_dict, state_dict, lora_name, flux_layer_name):
    if (original_stubkey := check_if_lora_exists(state_dict, lora_name)) != False:
        weights_to_pop = [k for k in state_dict.keys() if original_stubkey in k]
        for key in weights_to_pop:
            key_replacement = key.replace(
                original_stubkey, flux_layer_name.replace(".weight", "")
            )
            new_state_dict[key_replacement] = state_dict.pop(key)
        return new_state_dict, state_dict
    else:
        return new_state_dict, state_dict


def print_load_warning(missing: list[str], unexpected: list[str]) -> None:
    if len(missing) > 0 and len(unexpected) > 0:
        print(f"Got {len(missing)} missing keys:\n\t" + "\n\t".join(missing))
        print("\n" + "-" * 79 + "\n")
        print(f"Got {len(unexpected)} unexpected keys:\n\t" + "\n\t".join(unexpected))
    elif len(missing) > 0:
        print(f"Got {len(missing)} missing keys:\n\t" + "\n\t".join(missing))
    elif len(unexpected) > 0:
        print(f"Got {len(unexpected)} unexpected keys:\n\t" + "\n\t".join(unexpected))

def load_from_repo_id(repo_id, checkpoint_name):
    ckpt_path = hf_hub_download(repo_id, checkpoint_name)
    sd = load_sft(ckpt_path, device='cpu')
    return sd

def load_flow_model(name: str, device: str | torch.device = "cuda", hf_download: bool = True, lora_path=None):
    # Loading Flux
    print("Init model")
    ckpt_path = configs[name].ckpt_path
    if (
        ckpt_path is None
        and configs[name].repo_id is not None
        and configs[name].repo_flow is not None
        and hf_download
    ):
        ckpt_path = hf_hub_download(configs[name].repo_id, configs[name].repo_flow)

    # from .model import Flux, FluxParams
    with torch.device("meta" if ckpt_path is not None else device):
        model = Flux(configs[name].params).to(torch.bfloat16)

    if ckpt_path is not None:
        # load_sft doesn't support torch.device
        sd = load_sft(ckpt_path, device=str(device))
        
        if lora_path is not None:
            checkpoint = torch.load(lora_path, map_location=device)

            for k in checkpoint.keys():
                if "processor" not in k:
                    sd[k] = checkpoint[k]

        missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)

        print("Loading flow_model checkpoint")
        print_load_warning(missing, unexpected)

    return model

def load_flow_model_by_type(name: str, device: str | torch.device = "cuda", hf_download: bool = True, lora_path=None, model_type='src.flux.model.Flux'):
    # Loading Flux
    print("Init model")
    ckpt_path = configs[name].ckpt_path
    if (
        ckpt_path is None
        and configs[name].repo_id is not None
        and configs[name].repo_flow is not None
        and hf_download
    ):
        ckpt_path = hf_hub_download(configs[name].repo_id, configs[name].repo_flow)

    # with torch.device("meta" if ckpt_path is not None else device):
    with torch.device(device):
        model_cls = get_class(model_type)
        model = model_cls(configs[name].params).to(torch.bfloat16)

    if ckpt_path is not None:
        # load_sft doesn't support torch.device
        sd = load_sft(ckpt_path, device=str(device))
        
        if lora_path is not None:
            checkpoint = torch.load(lora_path, map_location=device)
            for k in checkpoint.keys():
                if "processor" not in k:
                    sd[k] = checkpoint[k]

        missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
        # missing, unexpected = model.load_state_dict(sd, strict=False)
         
        print("Loading flow_model checkpoint")
        print_load_warning(missing, unexpected)

    return model

def load_flow_model_quintized(name: str, device: str | torch.device = "cuda", hf_download: bool = True):
    # Loading Flux
    print("Init model")
    ckpt_path = configs[name].ckpt_path
    if (
        ckpt_path is None
        and configs[name].repo_id is not None
        and configs[name].repo_flow is not None
        and hf_download
    ):
        ckpt_path = hf_hub_download(configs[name].repo_id, configs[name].repo_flow)
    json_path = hf_hub_download(configs[name].repo_id, 'flux_dev_quantization_map.json')

    model = Flux(configs[name].params).to(torch.bfloat16)

    print("Loading checkpoint")
    # load_sft doesn't support torch.device
    sd = load_sft(ckpt_path, device='cpu')
    with open(json_path, "r") as f:
        quantization_map = json.load(f)
    print("Start a quantization process...")
    requantize(model, sd, quantization_map, device=device)
    print("Model is quantized!")
    return model

def load_controlnet(name, device, transformer=None):
    with torch.device(device):
        
        controlnet = ControlNetFlux(configs[name].params)

    if transformer is not None:
        controlnet.load_state_dict(transformer.state_dict(), strict=False)
        
    return controlnet


def load_t5(device: str | torch.device = "cuda", max_length: int = 512) -> HFEmbedder:
    # max length 64, 128, 256 and 512 should work (if your sequence is short enough)
    return HFEmbedder("xlabs-ai/xflux_text_encoders", max_length=max_length, torch_dtype=torch.bfloat16).to(device)


def load_clip(device: str | torch.device = "cuda") -> HFEmbedder:
    return HFEmbedder("openai/clip-vit-large-patch14", max_length=77, torch_dtype=torch.bfloat16).to(device)


def load_ae(name: str, device: str | torch.device = "cuda", hf_download: bool = True) -> AutoEncoder:
    ckpt_path = configs[name].ae_path
    if (
        ckpt_path is None
        and configs[name].repo_id is not None
        and configs[name].repo_ae is not None
        and hf_download
    ):
        ckpt_path = hf_hub_download(configs[name].repo_id_ae, configs[name].repo_ae)

    # Loading the autoencoder
    print("Init AE")
    with torch.device("meta" if ckpt_path is not None else device):
        ae = AutoEncoder(configs[name].ae_params)

    if ckpt_path is not None:
        sd = load_sft(ckpt_path, device=str(device))
        missing, unexpected = ae.load_state_dict(sd, strict=False, assign=True)
        print_load_warning(missing, unexpected)
    return ae


class WatermarkEmbedder:
    def __init__(self, watermark):
        self.watermark = watermark
        self.num_bits = len(WATERMARK_BITS)
        self.encoder = WatermarkEncoder()
        self.encoder.set_watermark("bits", self.watermark)

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """
        Adds a predefined watermark to the input image

        Args:
            image: ([N,] B, RGB, H, W) in range [-1, 1]

        Returns:
            same as input but watermarked
        """
        image = 0.5 * image + 0.5
        squeeze = len(image.shape) == 4
        if squeeze:
            image = image[None, ...]
        n = image.shape[0]
        image_np = rearrange((255 * image).detach().cpu(), "n b c h w -> (n b) h w c").numpy()[:, :, :, ::-1]
        # torch (b, c, h, w) in [0, 1] -> numpy (b, h, w, c) [0, 255]
        # watermarking libary expects input as cv2 BGR format
        for k in range(image_np.shape[0]):
            image_np[k] = self.encoder.encode(image_np[k], "dwtDct")
        image = torch.from_numpy(rearrange(image_np[:, :, :, ::-1], "(n b) h w c -> n b c h w", n=n)).to(
            image.device
        )
        image = torch.clamp(image / 255, min=0.0, max=1.0)
        if squeeze:
            image = image[0]
        image = 2 * image - 1
        return image

def set_full_lora(dit, mapped_network_alphas=8, rank=64, use_network_alpha=True, weight_dtype=torch.float32, set_name_list=['double_blocks', 'single_blocks'], scale=1.0):
    state_dict = dit.state_dict()

    for key, value_dict in state_dict.items():
        if not any([name in key for name in set_name_list]):
            continue

        attn_processor = dit
        for sub_key in key.split('.'):
            parent_module = attn_processor
            attn_processor = getattr(attn_processor, sub_key) 
            if (isinstance(attn_processor, nn.Conv2d) or isinstance(attn_processor, LoRACompatibleConv_)) and (not isinstance(attn_processor, LoRACompatibleConv)):
                in_features = attn_processor.in_channels
                out_features = attn_processor.out_channels
                kernel_size = attn_processor.kernel_size
                stride = attn_processor.stride
                padding = attn_processor.padding
                new_attn_processor = LoRACompatibleConv(in_features, out_features, kernel_size, stride, padding, scale=scale)
                lora = LoRAConv2dLayer(
                    in_features=in_features,
                    out_features=out_features,
                    rank=rank,
                    kernel_size=kernel_size,
                    stride=attn_processor.stride,
                    padding=attn_processor.padding,
                    network_alpha=mapped_network_alphas if use_network_alpha else None,
                ).to(dtype=weight_dtype, device=dit.device)
                new_attn_processor.load_state_dict(attn_processor.state_dict(), strict=False)
                new_attn_processor = new_attn_processor.to(dtype=weight_dtype, device=dit.device)
                new_attn_processor.set_lora_layer(lora)
                setattr(parent_module, sub_key, new_attn_processor)
            elif (isinstance(attn_processor, nn.Linear) or isinstance(attn_processor, LoRACompatibleLinear_)) and (not isinstance(attn_processor, LoRACompatibleLinear)):
                use_bias = isinstance(attn_processor.bias, torch.Tensor)
                new_attn_processor = LoRACompatibleLinear(attn_processor.in_features, attn_processor.out_features, bias=use_bias, scale=scale)
                lora = LoRALinearLayer(
                    attn_processor.in_features,
                    attn_processor.out_features,
                    rank,
                    mapped_network_alphas if use_network_alpha else None,
                ).to(dtype=weight_dtype, device=dit.device)
                new_attn_processor.load_state_dict(attn_processor.state_dict(), strict=False)
                new_attn_processor = new_attn_processor.to(dtype=weight_dtype, device=dit.device)
                new_attn_processor.set_lora_layer(lora)
                setattr(parent_module, sub_key, new_attn_processor)
    print('successfully set lora!!')
    return dit

def get_lora_layer(unet):
    lora_processor = []
    state_dict = unet.state_dict()
    for key, value_dict in state_dict.items():
        attn_processor = unet
        for sub_key in key.split('.'):
            attn_processor = getattr(attn_processor, sub_key)

            if isinstance(attn_processor, LoRACompatibleConv):
                lora_processor.append(attn_processor)
            elif isinstance(attn_processor, LoRACompatibleLinear):
                lora_processor.append(attn_processor)
    
    lora_processor = list(set(lora_processor))
    return lora_processor

def get_module_to_ignore_mixed_precision():
    try:
        from apex.normalization import FusedLayerNorm

        return [
            torch.nn.GroupNorm,
            torch.nn.modules.batchnorm._BatchNorm,
            torch.nn.LayerNorm,
            FusedLayerNorm,
        ]
    except:
        return [
            torch.nn.GroupNorm,
            torch.nn.modules.batchnorm._BatchNorm,
            torch.nn.LayerNorm,
        ]

def exclude_embedding_policy(module):
    return isinstance(module, torch.nn.Embedding)

def make_model_fsdp(
    model,
    param_dtype,
    device,
    force_leaf_modules={},
    sync_module_states=True,
    part_size=1e6,
    sharding_strategy=ShardingStrategy.FULL_SHARD,
):

    model = FSDP(
        model,
        sharding_strategy=sharding_strategy,
        process_group=None,
        forward_prefetch=True, # True
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        limit_all_gathers=True, # True
        use_orig_params=False, # default=False
        sync_module_states=sync_module_states,
        mixed_precision=MixedPrecision(
            param_dtype=param_dtype,
            reduce_dtype=torch.bfloat16, # float32
            buffer_dtype=torch.bfloat16, # float32
            keep_low_precision_grads=False,
            cast_forward_inputs=False, # True
            cast_root_forward_inputs=False, # True
            _module_classes_to_ignore=get_module_to_ignore_mixed_precision(),
        ),
        # auto_wrap_policy=ModuleWrapPolicy(force_leaf_modules),
        # auto_wrap_policy=functools.partial(
        #     size_based_auto_wrap_policy,
        #     min_num_params=part_size,
        #     force_leaf_modules=force_leaf_modules,
        # ),
        auto_wrap_policy=functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls=force_leaf_modules,
        ),
        device_id=device.index,
    )
    torch.cuda.empty_cache()
    gc.collect()
    return model

def make_model_dit_fsdp(
    model,
    param_dtype,
    device,
    force_leaf_modules={},
    sync_module_states=True,
    part_size=1e9,
    sharding_strategy=ShardingStrategy.FULL_SHARD,
):
    model = FSDP(
        model,
        # cpu_ram_efficient_loading=True,
        sharding_strategy=sharding_strategy,
        process_group=None,
        forward_prefetch=False, # True
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE, # BackwardPrefetch.BACKWARD_PRE
        limit_all_gathers=True,
        use_orig_params=True, # default=False
        sync_module_states=sync_module_states,
        mixed_precision=MixedPrecision(
            param_dtype=param_dtype,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
            keep_low_precision_grads=False,
            cast_forward_inputs=False, 
            cast_root_forward_inputs=False,
            _module_classes_to_ignore=get_module_to_ignore_mixed_precision(),
        ),
        # auto_wrap_policy=ModuleWrapPolicy(force_leaf_modules),
        auto_wrap_policy=functools.partial(
            size_based_auto_wrap_policy,
            min_num_params=part_size,
            force_leaf_modules=force_leaf_modules,
        ),
        # auto_wrap_policy=functools.partial(
        #     transformer_auto_wrap_policy,
        #     transformer_layer_cls=force_leaf_modules,
        # ),
        device_id=device.index,
    )
    torch.cuda.empty_cache()
    gc.collect()
    return model



# A fixed 48-bit message that was choosen at random
WATERMARK_MESSAGE = 0b001010101111111010000111100111001111010100101110
# bin(x)[2:] gives bits of x as str, use int to convert them to 0/1
WATERMARK_BITS = [int(bit) for bit in bin(WATERMARK_MESSAGE)[2:]]
