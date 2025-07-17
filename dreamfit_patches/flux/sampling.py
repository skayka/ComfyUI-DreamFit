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


import math
from typing import Callable

import torch
from einops import rearrange, repeat
from torch import Tensor
# from .model_ipa_kv import Flux
from .model_dreamfit import Flux
from .modules.conditioner import HFEmbedder
import numpy as np 
from diffusers import FluxControlNetModel
def get_noise(
    num_samples: int,
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
):
    return torch.randn(
        num_samples,
        16,
        # allow for packing
        2 * math.ceil(height / 16),
        2 * math.ceil(width / 16),
        device=device,
        dtype=dtype,
        generator=torch.Generator(device=device).manual_seed(seed),
    )

def prepare_img(img: Tensor, inp: dict[str, Tensor]) -> dict[str, Tensor]:
    bs, c, h, w = img.shape

    img = rearrange(img, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
    if img.shape[0] == 1 and bs > 1:
        img = repeat(img, "1 ... -> bs ...", bs=bs)

    img_ids = torch.zeros(h // 2, w // 2, 3)
    img_ids[..., 1] = img_ids[..., 1] + torch.arange(h // 2)[:, None]
    img_ids[..., 2] = img_ids[..., 2] + torch.arange(w // 2)[None, :]
    img_ids = repeat(img_ids, "h w c -> b (h w) c", b=bs)

    inp["img"] = img
    inp["img_ids"] = img_ids.to(img.device)

    return inp


def prepare_txt(t5: HFEmbedder, clip: HFEmbedder, prompt: str | list[str], device="cuda") -> dict[str, Tensor]:  ##, img: Tensor
    if isinstance(prompt, str):
        prompt = [prompt]
    
    bs = len(prompt)

    txt = t5(prompt)
    if txt.shape[0] == 1 and bs > 1:
        txt = repeat(txt, "1 ... -> bs ...", bs=bs)
    txt_ids = torch.zeros(bs, txt.shape[1], 3)

    vec = clip(prompt)
    if vec.shape[0] == 1 and bs > 1:
        vec = repeat(vec, "1 ... -> bs ...", bs=bs)

    return {
        "txt": txt.to(device),
        "txt_ids": txt_ids.to(device),
        "vec": vec.to(device),
    }


def prepare(t5: HFEmbedder, clip: HFEmbedder, img: Tensor, prompt: str | list[str]) -> dict[str, Tensor]:
    bs, c, h, w = img.shape
    if bs == 1 and not isinstance(prompt, str):
        bs = len(prompt)

    img = rearrange(img, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
    if img.shape[0] == 1 and bs > 1:
        img = repeat(img, "1 ... -> bs ...", bs=bs)

    img_ids = torch.zeros(h // 2, w // 2, 3)
    img_ids[..., 1] = img_ids[..., 1] + torch.arange(h // 2)[:, None]
    img_ids[..., 2] = img_ids[..., 2] + torch.arange(w // 2)[None, :]
    img_ids = repeat(img_ids, "h w c -> b (h w) c", b=bs)

    if isinstance(prompt, str):
        prompt = [prompt]
    txt = t5(prompt)
    if txt.shape[0] == 1 and bs > 1:
        txt = repeat(txt, "1 ... -> bs ...", bs=bs)
    txt_ids = torch.zeros(bs, txt.shape[1], 3)

    vec = clip(prompt)
    if vec.shape[0] == 1 and bs > 1:
        vec = repeat(vec, "1 ... -> bs ...", bs=bs)

    return {
        "img": img,
        "img_ids": img_ids.to(img.device),
        "txt": txt.to(img.device),
        "txt_ids": txt_ids.to(img.device),
        "vec": vec.to(img.device),
    }

def time_shift(mu: float, sigma: float, t: Tensor):
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


def get_lin_function(
    x1: float = 256, y1: float = 0.5, x2: float = 4096, y2: float = 1.15
) -> Callable[[float], float]:
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return lambda x: m * x + b


def get_schedule(
    num_steps: int,
    image_seq_len: int,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
    shift: bool = True,
) -> list[float]:
    # extra step for zero
    timesteps = torch.linspace(1, 0, num_steps + 1)

    # shifting the schedule to favor high timesteps for higher signal images
    if shift:
        # eastimate mu based on linear estimation between two points
        mu = get_lin_function(y1=base_shift, y2=max_shift)(image_seq_len)
        timesteps = time_shift(mu, 1.0, timesteps)

    return timesteps.tolist()

def denoise_origin(
    model: Flux,
    # model input
    img: Tensor,
    img_ids: Tensor,
    txt: Tensor,
    txt_ids: Tensor,
    vec: Tensor,
    neg_txt: Tensor,
    neg_txt_ids: Tensor,
    neg_vec: Tensor,
    # sampling parameters
    timesteps: list[float],
    guidance: float = 4.0,
    true_gs = 1,
    timestep_to_start_cfg=0,
    # ip-adapter parameters
    image_proj: Tensor=None, 
    neg_image_proj: Tensor=None, 
    ip_scale: Tensor | float = 1.0,
    neg_ip_scale: Tensor | float = 1.0,
    inpaint_mask=None,
    x0=None,
):
    i = 0
    # this is ignored for schnell
    guidance_vec = torch.full((img.shape[0],), guidance, device=img.device, dtype=img.dtype)
    for t_curr, t_prev in zip(timesteps[:-1], timesteps[1:]):
        t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)
        pred = model(
            img=img,
            img_ids=img_ids,
            txt=txt,
            txt_ids=txt_ids,
            y=vec,
            timesteps=t_vec,
            guidance=guidance_vec,
            image_proj=image_proj,
            ip_scale=ip_scale, 
        )
        if i >= timestep_to_start_cfg:
            neg_pred = model(
                img=img,
                img_ids=img_ids,
                txt=neg_txt,
                txt_ids=neg_txt_ids,
                y=neg_vec,
                timesteps=t_vec,
                guidance=guidance_vec, 
                image_proj=neg_image_proj,
                ip_scale=neg_ip_scale, 
            )     
            pred = neg_pred + true_gs * (pred - neg_pred)
        img = img + (t_prev - t_curr) * pred
        if inpaint_mask is not None:
            noise = torch.randn_like(img).to(img.device)
            x_t_from_0 = (1 - t_curr) * x0 + t_curr * noise
            # x_t_from_0 = x0
            img = img * inpaint_mask + x_t_from_0 * (1 - inpaint_mask)
        i += 1
    return img

def denoise(
    model: Flux,
    # model input
    inp_person: dict[str, Tensor],
    inp_cloth: dict[str, Tensor],
    # neg_inp_cond
    neg_inp_cond: dict[str, Tensor],
    
    # sampling parameters
    timesteps: list[float],
    guidance: float = 4.0,
    true_gs = 1,
    timestep_to_start_cfg=0,
    # ip-adapter parameters
    image_proj: Tensor=None, 
    neg_image_proj: Tensor=None, 
    ip_scale: Tensor | float = 1.0,
    neg_ip_scale: Tensor | float = 1.0,
    num_steps=50,
):
    i = 0
    img  = inp_person['img']
    # this is ignored for schnell
    guidance_vec = torch.full((img.shape[0],), guidance, device=img.device, dtype=img.dtype)
 
    # timesteps_infer = [timesteps[x] for x in np.linspace(0, 999, num_steps).astype(int).tolist()]
    # print("timesteps_infer ", timesteps_infer)

    for t_curr, t_prev in zip(timesteps[:-1], timesteps[1:]):
        t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)
        if i == 0:
            _ = model(
                img=inp_cloth['img'],
                img_ids=inp_cloth['img_ids'],
                txt=inp_cloth['txt'],
                txt_ids=inp_cloth['txt_ids'],
                y=inp_cloth['vec'],
                timesteps=torch.zeros_like(t_vec),
                guidance=guidance_vec,
                rw_mode="write",
            )
            _ = model(
                img=neg_inp_cond['img'],
                img_ids=neg_inp_cond['img_ids'],
                txt=neg_inp_cond['txt'],
                txt_ids=neg_inp_cond['txt_ids'],
                y=neg_inp_cond['vec'],
                timesteps=torch.zeros_like(t_vec),
                guidance=guidance_vec,
                rw_mode="neg_write",
            )

        ## Predict the noise residual 
        pred = model(
            img=img,
            img_ids=inp_person['img_ids'],
            txt=inp_person['txt'],
            txt_ids=inp_person['txt_ids'],
            y=inp_person['vec'],
            timesteps=t_vec,
            guidance=guidance_vec,
            rw_mode="read",
            ref_img_ids=inp_cloth['img_ids'],
        )
 
        if i >= timestep_to_start_cfg:
            neg_pred = model(
                img=img,
                img_ids=inp_person['img_ids'],
                txt=neg_inp_cond['txt'],
                txt_ids=neg_inp_cond['txt_ids'],
                y=neg_inp_cond['vec'],
                timesteps=t_vec,
                guidance=guidance_vec, 
                rw_mode="neg_read"
            )
            pred = neg_pred + true_gs * (pred - neg_pred)
 
        img = img + (t_prev - t_curr) * pred
        i += 1
    return img

def denoise_controlnet(
    model: Flux,
    controlnet:None,
    # model input
    img: Tensor,
    img_ids: Tensor,
    txt: Tensor,
    txt_ids: Tensor,
    vec: Tensor,
    neg_txt: Tensor,
    neg_txt_ids: Tensor,
    neg_vec: Tensor,
    controlnet_cond,
    # sampling parameters
    timesteps: list[float],
    guidance: float = 4.0,
    true_gs = 1,
    controlnet_gs=0.7,
    timestep_to_start_cfg=0,
    # ip-adapter parameters
    image_proj: Tensor=None, 
    neg_image_proj: Tensor=None, 
    ip_scale: Tensor | float = 1, 
    neg_ip_scale: Tensor | float = 1, 
    control_mode=None,
    # num_steps=50,
):
    # this is ignored for schnell
    i = 0
    guidance_vec = torch.full((img.shape[0],), guidance, device=img.device, dtype=img.dtype)
    # timesteps_infer = [timesteps[x] for x in np.linspace(0, 999, num_steps).astype(int).tolist()]
    for t_curr, t_prev in zip(timesteps[:-1], timesteps[1:]):

        t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)
        if isinstance(controlnet, FluxControlNetModel):
            (
                block_res_samples,
                controlnet_single_block_samples,
            ) = controlnet(
                hidden_states=img,
                controlnet_cond=controlnet_cond,
                conditioning_scale=controlnet_gs,
                controlnet_mode=control_mode,
                timestep=t_vec,
                guidance=guidance_vec,
                pooled_projections=vec,
                encoder_hidden_states=txt,
                txt_ids=txt_ids[0],
                img_ids=img_ids[0],
                joint_attention_kwargs=None,
                return_dict=False,
            )
        else:
            block_res_samples = controlnet(
                        img=img,
                        img_ids=img_ids,
                        controlnet_cond=controlnet_cond,
                        txt=txt,
                        txt_ids=txt_ids,
                        y=vec,
                        timesteps=t_vec,
                        guidance=guidance_vec,
                    )
            controlnet_single_block_samples = None
            
        pred = model(
            img=img,
            img_ids=img_ids,
            txt=txt,
            txt_ids=txt_ids,
            y=vec,
            timesteps=t_vec,
            guidance=guidance_vec,
            block_controlnet_hidden_states=block_res_samples,
            single_block_controlnet_hidden_states=controlnet_single_block_samples if controlnet_single_block_samples is not None else None,
            image_proj=image_proj,
            ip_scale=ip_scale,
        )

        if i >= timestep_to_start_cfg:
            if isinstance(controlnet, FluxControlNetModel):
                (
                    block_res_samples,
                    controlnet_single_block_samples,
                ) = controlnet(
                    hidden_states=img,
                    controlnet_cond=controlnet_cond,
                    conditioning_scale=controlnet_gs,
                    timestep=t_vec,
                    guidance=guidance_vec,
                    pooled_projections=vec,
                    encoder_hidden_states=neg_txt,
                    txt_ids=neg_txt_ids[0],
                    img_ids=img_ids[0],
                    joint_attention_kwargs=None,
                    return_dict=False,
                )
            else:
                neg_block_res_samples = controlnet(
                            img=img,
                            img_ids=img_ids,
                            controlnet_cond=controlnet_cond,
                            txt=neg_txt,
                            txt_ids=neg_txt_ids,
                            y=neg_vec,
                            timesteps=t_vec,
                            guidance=guidance_vec,
                        )
            controlnet_single_block_samples = None
            neg_pred = model(
                img=img,
                img_ids=img_ids,
                txt=neg_txt,
                txt_ids=neg_txt_ids,
                y=neg_vec,
                timesteps=t_vec,
                block_controlnet_hidden_states=neg_block_res_samples,
                image_proj=neg_image_proj,
                ip_scale=neg_ip_scale, 
            )     
            pred = neg_pred + true_gs * (pred - neg_pred)
   
        img = img + (t_prev - t_curr) * pred

        i += 1
    return img

def denoise_controlnet_dreamfit(
    model: Flux,
    controlnet:None,
    # model input
    img: Tensor,
    img_ids: Tensor,
    txt: Tensor,
    txt_ids: Tensor,
    vec: Tensor,
    neg_txt: Tensor,
    neg_txt_ids: Tensor,
    neg_vec: Tensor,
    controlnet_cond,
    # sampling parameters
    timesteps: list[float],
    guidance: float = 4.0,
    true_gs = 1,
    controlnet_gs=0.7,
    timestep_to_start_cfg=0,
    # ip-adapter parameters
    image_proj: Tensor=None, 
    neg_image_proj: Tensor=None, 
    ip_scale: Tensor | float = 1, 
    neg_ip_scale: Tensor | float = 1, 
    control_mode=None,
    inp_cloth=None,
    # num_steps=50,
):
    # this is ignored for schnell
    i = 0
    guidance_vec = torch.full((img.shape[0],), guidance, device=img.device, dtype=img.dtype)
    # timesteps_infer = [timesteps[x] for x in np.linspace(0, 999, num_steps).astype(int).tolist()]
    for t_curr, t_prev in zip(timesteps[:-1], timesteps[1:]):
        t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)
        if i == 0:
            _ = model(
                img=inp_cloth['img'],
                img_ids=inp_cloth['img_ids'],
                txt=inp_cloth['txt'],
                txt_ids=inp_cloth['txt_ids'],
                y=inp_cloth['vec'],
                timesteps=torch.zeros_like(t_vec),
                guidance=guidance_vec,
                rw_mode="write",
            )
            # _ = model(
            #     img=neg_inp_cond['img'],
            #     img_ids=neg_inp_cond['img_ids'],
            #     txt=neg_inp_cond['txt'],
            #     txt_ids=neg_inp_cond['txt_ids'],
            #     y=neg_inp_cond['vec'],
            #     timesteps=torch.zeros_like(t_vec),
            #     guidance=guidance_vec,
            #     rw_mode="neg_write",
            # )

        if isinstance(controlnet, FluxControlNetModel):
            (
                block_res_samples,
                controlnet_single_block_samples,
            ) = controlnet(
                hidden_states=img,
                controlnet_cond=controlnet_cond,
                conditioning_scale=controlnet_gs,
                controlnet_mode=control_mode,
                timestep=t_vec,
                guidance=guidance_vec,
                pooled_projections=vec,
                encoder_hidden_states=txt,
                txt_ids=txt_ids[0],
                img_ids=img_ids[0],
                joint_attention_kwargs=None,
                return_dict=False,
            )
        else:
            block_res_samples = controlnet(
                        img=img,
                        img_ids=img_ids,
                        controlnet_cond=controlnet_cond,
                        txt=txt,
                        txt_ids=txt_ids,
                        y=vec,
                        timesteps=t_vec,
                        guidance=guidance_vec,
                    )
            controlnet_single_block_samples = None
            
        pred = model(
            img=img,
            img_ids=img_ids,
            txt=txt,
            txt_ids=txt_ids,
            y=vec,
            timesteps=t_vec,
            guidance=guidance_vec,
            block_controlnet_hidden_states=block_res_samples,
            single_block_controlnet_hidden_states=controlnet_single_block_samples if controlnet_single_block_samples is not None else None,
            image_proj=image_proj,
            ip_scale=ip_scale,
            rw_mode="read",
        )

        # if i >= timestep_to_start_cfg:
        #     if isinstance(controlnet, FluxControlNetModel):
        #         (
        #             block_res_samples,
        #             controlnet_single_block_samples,
        #         ) = controlnet(
        #             hidden_states=img,
        #             controlnet_cond=controlnet_cond,
        #             conditioning_scale=controlnet_gs,
        #             timestep=t_vec,
        #             guidance=guidance_vec,
        #             pooled_projections=vec,
        #             encoder_hidden_states=neg_txt,
        #             txt_ids=neg_txt_ids[0],
        #             img_ids=img_ids[0],
        #             joint_attention_kwargs=None,
        #             return_dict=False,
        #         )
        #     else:
        #         neg_block_res_samples = controlnet(
        #                     img=img,
        #                     img_ids=img_ids,
        #                     controlnet_cond=controlnet_cond,
        #                     txt=neg_txt,
        #                     txt_ids=neg_txt_ids,
        #                     y=neg_vec,
        #                     timesteps=t_vec,
        #                     guidance=guidance_vec,
        #                 )
        #     controlnet_single_block_samples = None
        #     neg_pred = model(
        #         img=img,
        #         img_ids=img_ids,
        #         txt=neg_txt,
        #         txt_ids=neg_txt_ids,
        #         y=neg_vec,
        #         timesteps=t_vec,
        #         block_controlnet_hidden_states=neg_block_res_samples,
        #         image_proj=neg_image_proj,
        #         ip_scale=neg_ip_scale, 
        #     )     
        #     pred = neg_pred + true_gs * (pred - neg_pred)
   
        img = img + (t_prev - t_curr) * pred

        i += 1
    return img


def unpack(x: Tensor, height: int, width: int) -> Tensor:
    return rearrange(
        x,
        "b (h w) (c ph pw) -> b c (h ph) (w pw)",
        h=math.ceil(height / 16),
        w=math.ceil(width / 16),
        ph=2,
        pw=2,
    )
