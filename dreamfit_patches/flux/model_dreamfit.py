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


from dataclasses import dataclass
import numpy as np

import torch
from torch import Tensor, nn

from .modules.layers_dreamfit import (DoubleStreamBlock, EmbedND, LastLayer,
                                MLPEmbedder, SingleStreamBlock, LoRALinearLayer,
                                timestep_embedding)

from diffusers.models import ModelMixin

@dataclass
class FluxParams:
    in_channels: int
    vec_in_dim: int
    context_in_dim: int
    hidden_size: int
    mlp_ratio: float
    num_heads: int
    depth: int
    depth_single_blocks: int
    axes_dim: list[int]
    theta: int
    qkv_bias: bool
    guidance_embed: bool


class Flux(ModelMixin):  #
    """
    Transformer model for flow matching on sequences.
    """
    _supports_gradient_checkpointing = True

    def __init__(self, params: FluxParams):
        super().__init__()

        self.params = params
        self.in_channels = params.in_channels
        self.out_channels = self.in_channels
        if params.hidden_size % params.num_heads != 0:
            raise ValueError(
                f"Hidden size {params.hidden_size} must be divisible by num_heads {params.num_heads}"
            )
        pe_dim = params.hidden_size // params.num_heads
        if sum(params.axes_dim) != pe_dim:
            raise ValueError(f"Got {params.axes_dim} but expected positional dim {pe_dim}")
        self.hidden_size = params.hidden_size
        self.num_heads = params.num_heads
        self.pe_embedder = EmbedND(dim=pe_dim, theta=params.theta, axes_dim=params.axes_dim)

        self.img_in = nn.Linear(self.in_channels, self.hidden_size, bias=True)
        # self.img_in_lora = LoRALinearLayer(self.in_channels, self.hidden_size, rank=64, device="cuda")

        self.time_in = MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size)

        self.vector_in = MLPEmbedder(params.vec_in_dim, self.hidden_size)
        
        self.guidance_in = (
            MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size) if params.guidance_embed else nn.Identity()
        )

        self.txt_in = nn.Linear(params.context_in_dim, self.hidden_size)

        self.double_blocks = nn.ModuleList(
            [
                DoubleStreamBlock(
                    self.hidden_size,
                    self.num_heads,
                    mlp_ratio=params.mlp_ratio,
                    qkv_bias=params.qkv_bias,
                )
                for _ in range(params.depth)
            ]
        )

        self.single_blocks = nn.ModuleList(
            [
                SingleStreamBlock(self.hidden_size, self.num_heads, mlp_ratio=params.mlp_ratio)
                for _ in range(params.depth_single_blocks)
            ]
        )

        self.final_layer = LastLayer(self.hidden_size, 1, self.out_channels)
        self.gradient_checkpointing = False

    def _set_gradient_checkpointing(self, module, value=False):
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value

    @property
    def attn_processors(self):
        # set recursively
        processors = {}

        def fn_recursive_add_processors(name: str, module: torch.nn.Module, processors):
            if hasattr(module, "set_processor"):
                processors[f"{name}.processor"] = module.processor

            for sub_name, child in module.named_children():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

            return processors

        for name, module in self.named_children():
            fn_recursive_add_processors(name, module, processors)

        return processors

    def set_attn_processor(self, processor):
        r"""
        Sets the attention processor to use to compute attention.

        Parameters:
            processor (`dict` of `AttentionProcessor` or only `AttentionProcessor`):
                The instantiated processor class or a dictionary of processor classes that will be set as the processor
                for **all** `Attention` layers.

                If `processor` is a dict, the key needs to define the path to the corresponding cross attention
                processor. This is strongly recommended when setting trainable attention processors.

        """
        count = len(self.attn_processors.keys())

        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"A dict of processors was passed, but the number of processors {len(processor)} does not match the"
                f" number of attention layers: {count}. Please make sure to pass {count} processor classes."
            )

        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
            if hasattr(module, "set_processor"):
                if not isinstance(processor, dict):
                    module.set_processor(processor)
                else:
                    module.set_processor(processor.pop(f"{name}.processor"))

            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

        for name, module in self.named_children():
            fn_recursive_attn_processor(name, module, processor)

    def forward(
        self,
        img: Tensor,
        img_ids: Tensor,
        txt: Tensor,
        txt_ids: Tensor,
        timesteps: Tensor,
        y: Tensor,
        block_controlnet_hidden_states=None,
        single_block_controlnet_hidden_states=None,
        guidance: Tensor | None = None,
        image_proj: Tensor | None = None, 
        ip_scale: Tensor | float = 1.0, 
        rw_mode: str = "write",
        ref_img_ids=None, # handle the situation when the reference picture is different from the latent noise
    ) -> Tensor:
        if img.ndim != 3 or txt.ndim != 3:
            raise ValueError("Input img and txt tensors must have 3 dimensions.")
        
        if ref_img_ids == None:
            ref_img_ids = img_ids

        # running on sequences img
        img = self.img_in(img)

        vec = self.time_in(timestep_embedding(timesteps, 256))#, rw_mode="read"

        if self.params.guidance_embed:
            if guidance is None:
                raise ValueError("Didn't get guidance strength for guidance distilled model.")
            vec = vec + self.guidance_in(timestep_embedding(guidance, 256)) #, rw_mode="read"

        vec = vec + self.vector_in(y) ## vec = clip_emb + time_emb + guidance_emb

        txt = self.txt_in(txt)
 
        if "write" in rw_mode:
            ids = torch.cat((txt_ids, img_ids), dim=1)
            single_ids = ids

        elif "read" in rw_mode:
            ids = torch.cat((txt_ids, img_ids, ref_img_ids), dim=1)  
            single_ids = torch.cat((txt_ids, img_ids, txt_ids, ref_img_ids), dim=1) 

        else:
            ids = torch.cat((txt_ids, img_ids), dim=1)
            single_ids = ids

        pe = self.pe_embedder(ids)
        single_pe = self.pe_embedder(single_ids)

        if block_controlnet_hidden_states is not None:
            controlnet_depth = len(block_controlnet_hidden_states)
        
        for index_block, block in enumerate(self.double_blocks):
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                # ckpt_kwargs: Dict[str, Any] = {"use_reentrant": True} #if is_torch_version(">=", "1.11.0") else {}
                img, txt = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    img,
                    txt,
                    vec,
                    pe,
                    image_proj,
                    ip_scale,
                )
            else:
                img, txt = block(
                    img=img, 
                    txt=txt, 
                    vec=vec, 
                    pe=pe, 
                    rw_mode=rw_mode,
                    image_proj=image_proj,
                    ip_scale=ip_scale,
                )

            # controlnet residual
            if block_controlnet_hidden_states is not None:
                interval_control = self.params.depth / len(
                    block_controlnet_hidden_states
                )
                interval_control = int(np.ceil(interval_control))
                img = img + block_controlnet_hidden_states[index_block // interval_control]

        img = torch.cat((txt, img), 1)

        for index_block, block in enumerate(self.single_blocks):
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                # ckpt_kwargs: Dict[str, Any] = {"use_reentrant": True} #if is_torch_version(">=", "1.11.0") else {}
                img = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    img,
                    vec,
                    pe,
                )
            else:
                img = block(img, vec=vec, pe=single_pe, rw_mode=rw_mode)

            if single_block_controlnet_hidden_states is not None:
                interval_control = self.params.depth_single_blocks / len(
                    single_block_controlnet_hidden_states
                )
                interval_control = int(np.ceil(interval_control))
                img[:, txt.shape[1] :, ...] = (img[:, txt.shape[1] :, ...] + single_block_controlnet_hidden_states[index_block // interval_control])


        img = img[:, txt.shape[1] :, ...]

        img = self.final_layer(img, vec)  # (N, T, patch_size ** 2 * out_channels)
        return img
