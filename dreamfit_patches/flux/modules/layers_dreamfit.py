# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates. 
# Copyright (c) 2024 Black Forest Labs and The XLabs-AI Team. 

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import math
from dataclasses import dataclass

import torch
from einops import rearrange
from torch import Tensor, nn

from ..math import attention, rope
import torch.nn.functional as F

from diffusers.models import ModelMixin

class EmbedND(ModelMixin):
    def __init__(self, dim: int, theta: int, axes_dim: list[int]):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: Tensor) -> Tensor:
        n_axes = ids.shape[-1]
        emb = torch.cat(
            [rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(n_axes)],
            dim=-3,
        )

        return emb.unsqueeze(1)


def timestep_embedding(t: Tensor, dim, max_period=10000, time_factor: float = 1000.0):
    """
    Create sinusoidal timestep embeddings.
    :param t: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an (N, D) Tensor of positional embeddings.
    """
    t = time_factor * t
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(
        t.device
    )

    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    if torch.is_floating_point(t):
        embedding = embedding.to(t)
    return embedding


class RMSNorm(torch.nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm)
    """
    def __init__(self, dim: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor):
        x_dtype = x.dtype
        x = x.float()
        rrms = torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True) + 1e-6)
        return (x * rrms).to(dtype=x_dtype) * self.scale

class QKNorm(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.query_norm = RMSNorm(dim)
        self.key_norm = RMSNorm(dim)

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        q = self.query_norm(q)
        k = self.key_norm(k)
        return q.to(v), k.to(v)


class LoRALinearLayer(ModelMixin):
    def __init__(self, in_features, out_features, rank=4, network_alpha=None, device=None, dtype=None):
        super().__init__()

        self.down = nn.Linear(in_features, rank, bias=False, device=device, dtype=dtype)
        self.up = nn.Linear(rank, out_features, bias=False, device=device, dtype=dtype)
        # This value has the same meaning as the `--network_alpha` option in the kohya-ss trainer script.
        # See https://github.com/darkstorm2150/sd-scripts/blob/main/docs/train_network_README-en.md#execute-learning
        self.network_alpha = network_alpha
        self.rank = rank

        nn.init.normal_(self.down.weight, std=1 / rank)
        nn.init.zeros_(self.up.weight)

    def forward(self, hidden_states):
        orig_dtype = hidden_states.dtype
        dtype = self.down.weight.dtype

        down_hidden_states = self.down(hidden_states.to(dtype))
        up_hidden_states = self.up(down_hidden_states)

        if self.network_alpha is not None:
            up_hidden_states *= self.network_alpha / self.rank

        return up_hidden_states.to(orig_dtype)


class MLPEmbedder(ModelMixin):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.in_layer = nn.Linear(in_dim, hidden_dim, bias=True)

        self.silu = nn.SiLU()
        self.out_layer = nn.Linear(hidden_dim, hidden_dim, bias=True)


    def forward(self, x: Tensor) -> Tensor:
        x = self.in_layer(x)
        x = self.silu(x)
        x = self.out_layer(x)

        return x

class FLuxSelfAttnProcessor:
    def __call__(self, attn, x, pe, **attention_kwargs):
        print('2' * 30)

        qkv = attn.qkv(x)
        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        q, k = attn.norm(q, k, v)
        x = attention(q, k, v, pe=pe)
        x = attn.proj(x)
        return x

class LoraFluxAttnProcessor(ModelMixin):

    def __init__(self, dim: int, rank=4, network_alpha=None, lora_weight=1):
        super().__init__()
        self.qkv_lora = LoRALinearLayer(dim, dim * 3, rank, network_alpha)
        self.proj_lora = LoRALinearLayer(dim, dim, rank, network_alpha)
        self.lora_weight = lora_weight

    def __call__(self, attn, x, pe, **attention_kwargs):
        qkv = attn.qkv(x) + self.qkv_lora(x) * self.lora_weight
        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        q, k = attn.norm(q, k, v)
        x = attention(q, k, v, pe=pe)
        x = attn.proj(x) + self.proj_lora(x) * self.lora_weight
        print('1' * 30)
        print(x.norm(), (self.proj_lora(x) * self.lora_weight).norm(), 'norm')
        return x

class SelfAttention(ModelMixin):
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.norm = QKNorm(head_dim)
        self.proj = nn.Linear(dim, dim)
    def forward():
        pass

@dataclass
class ModulationOut:
    shift: Tensor
    scale: Tensor
    gate: Tensor


class Modulation(torch.nn.Module):
    def __init__(self, dim: int, double: bool):
        super().__init__()
        self.is_double = double
        self.multiplier = 6 if double else 3
        self.lin = nn.Linear(dim, self.multiplier * dim, bias=True) # [3072, 18432]
        self.lin_lora = LoRALinearLayer(dim, self.multiplier * dim, rank=32, network_alpha=16)

    def forward(self, vec: Tensor, rw_mode="normal") -> tuple[ModulationOut, ModulationOut | None]:
        x =  nn.functional.silu(vec)
        if "write" in rw_mode:
            out = (self.lin(x) + self.lin_lora(x))[:, None, :].chunk(self.multiplier, dim=-1)
        else:
            out = self.lin(x)[:, None, :].chunk(self.multiplier, dim=-1)

        return (
            ModulationOut(*out[:3]),
            ModulationOut(*out[3:]) if self.is_double else None,
        )

class DoubleStreamBlockLoraProcessor(ModelMixin):
    def __init__(self, dim: int, rank=4, network_alpha=None, lora_weight=1, device=None, dtype=None):
        super().__init__()
        self.ref_qkv_lora_q  = LoRALinearLayer(dim, dim, rank, network_alpha, device=device, dtype=dtype)
        self.ref_qkv_lora_k  = LoRALinearLayer(dim, dim, rank, network_alpha, device=device, dtype=dtype)
        self.ref_qkv_lora_v  = LoRALinearLayer(dim, dim, rank, network_alpha, device=device, dtype=dtype)

        self.ref_proj_lora1 = LoRALinearLayer(dim, dim, rank, network_alpha, device=device, dtype=dtype)
        self.lora_weight = lora_weight

        self.bank_img_q = None
        self.bank_img_k = None
        self.bank_img_v = None

        self.bank_neg_img_q = None
        self.bank_neg_img_k = None
        self.bank_neg_img_v = None

    def forward(self, attn, img, txt, vec, pe, rw_mode="write", **attention_kwargs):

        if "write" in rw_mode:
            img_mod1, img_mod2 = attn.img_mod(vec, rw_mode=rw_mode)
            txt_mod1, txt_mod2 = attn.txt_mod(vec, rw_mode="txt")

            # prepare image for attention
            img_modulated = attn.img_norm1(img)
            img_modulated = (1 + img_mod1.scale) * img_modulated + img_mod1.shift

            # use_lora
            img_qkv = attn.img_attn.qkv(img_modulated) 
            img_q, img_k, img_v = rearrange(img_qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads)

            ref_lora_q = self.ref_qkv_lora_q(img_modulated) * self.lora_weight
            img_q = img_q + rearrange(ref_lora_q, "B L (H D) -> B H L D", H=attn.num_heads)

            ref_lora_k = self.ref_qkv_lora_k(img_modulated) * self.lora_weight
            img_k = img_k + rearrange(ref_lora_k, "B L (H D) -> B H L D", H=attn.num_heads)

            ref_lora_v = self.ref_qkv_lora_v(img_modulated) * self.lora_weight
            img_v = img_v + rearrange(ref_lora_v, "B L (H D) -> B H L D", H=attn.num_heads)

            img_q, img_k = attn.img_attn.norm(img_q, img_k, img_v)

            if rw_mode == "write":
                self.bank_img_q = img_q
                self.bank_img_k = img_k
                self.bank_img_v = img_v
            else:
                self.bank_neg_img_q = img_q
                self.bank_neg_img_k = img_k
                self.bank_neg_img_v = img_v

            # prepare txt for attention
            txt_modulated = attn.txt_norm1(txt)
            txt_modulated = (1 + txt_mod1.scale) * txt_modulated + txt_mod1.shift
            txt_qkv = attn.txt_attn.qkv(txt_modulated)  ## + self.qkv_lora2(txt_modulated) * self.lora_weight
            txt_q, txt_k, txt_v = rearrange(txt_qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads)
            txt_q, txt_k = attn.txt_attn.norm(txt_q, txt_k, txt_v)

            # run actual attention
            q = torch.cat((txt_q, img_q), dim=2)
            k = torch.cat((txt_k, img_k), dim=2)
            v = torch.cat((txt_v, img_v), dim=2)

            attn1 = attention(q, k, v, pe=pe)
            txt_attn, img_attn = attn1[:, : txt.shape[1]], attn1[:, txt.shape[1] :]

            # calculate the img bloks
            # use_lora
            img = img + img_mod1.gate * attn.img_attn.proj(img_attn) + img_mod1.gate * self.ref_proj_lora1(img_attn) * self.lora_weight

            ### use_lora
            img_scale_shift  =  (1 + img_mod2.scale) * attn.img_norm2(img) + img_mod2.shift
            img_scale_shift = attn.img_mlp[0](img_scale_shift) + attn.img_mlp_lora_1(img_scale_shift)
            img_scale_shift = attn.img_mlp[1](img_scale_shift)
            img_scale_shift = attn.img_mlp[2](img_scale_shift) + attn.img_mlp_lora_2(img_scale_shift)
            img = img + img_mod2.gate * img_scale_shift
            
            # calculate the txt bloks
            txt = txt + txt_mod1.gate * attn.txt_attn.proj(txt_attn)  ##+ txt_mod1.gate * self.proj_lora2(txt_attn) * self.lora_weight
            txt = txt + txt_mod2.gate * attn.txt_mlp((1 + txt_mod2.scale) * attn.txt_norm2(txt) + txt_mod2.shift)

        elif "read" in rw_mode:
            img_mod1, img_mod2 = attn.img_mod(vec,rw_mode=rw_mode)
            txt_mod1, txt_mod2 = attn.txt_mod(vec,rw_mode="txt")

            # prepare image for attention
            img_modulated = attn.img_norm1(img)
            img_modulated = (1 + img_mod1.scale) * img_modulated + img_mod1.shift
            img_qkv = attn.img_attn.qkv(img_modulated)
            img_q, img_k, img_v = rearrange(img_qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads, D=attn.head_dim)
            img_q, img_k = attn.img_attn.norm(img_q, img_k, img_v)

            # prepare txt for attention
            txt_modulated = attn.txt_norm1(txt)
            txt_modulated = (1 + txt_mod1.scale) * txt_modulated + txt_mod1.shift
            txt_qkv = attn.txt_attn.qkv(txt_modulated)
            txt_q, txt_k, txt_v = rearrange(txt_qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads, D=attn.head_dim)
            txt_q, txt_k = attn.txt_attn.norm(txt_q, txt_k, txt_v)

            # run actual attention
            if rw_mode == "read":
                ref_q = self.bank_img_q
                ref_k = self.bank_img_k
                ref_v = self.bank_img_v
            else:
                ref_q = self.bank_neg_img_q
                ref_k = self.bank_neg_img_k
                ref_v = self.bank_neg_img_v

            q = torch.cat((txt_q, img_q, ref_q), dim=2)
            k = torch.cat((txt_k, img_k, ref_k), dim=2)
            v = torch.cat((txt_v, img_v, ref_v), dim=2)

            attn_mask = torch.ones((q.shape[0], q.shape[1], q.shape[2], q.shape[2]), dtype=torch.bool, device=q.device)
            attn_mask[:,:,:txt_q.shape[2], -ref_q.shape[2]:] = 0
            attn_mask[:,:,-ref_q.shape[2]:, :txt_q.shape[2]] = 0

            attn1 = attention(q, k, v, pe=pe, attn_mask=attn_mask)

            txt_attn, img_attn = attn1[:, :txt.shape[1]], attn1[:, txt.shape[1]:txt.shape[1]+img.shape[1]]

            # calculate the img bloks
            img = img + img_mod1.gate * attn.img_attn.proj(img_attn)
            img = img + img_mod2.gate * attn.img_mlp((1 + img_mod2.scale) * attn.img_norm2(img) + img_mod2.shift)

            # calculate the txt bloks
            txt = txt + txt_mod1.gate * attn.txt_attn.proj(txt_attn)
            txt = txt + txt_mod2.gate * attn.txt_mlp((1 + txt_mod2.scale) * attn.txt_norm2(txt) + txt_mod2.shift)

        else:
            img_mod1, img_mod2 = attn.img_mod(vec, rw_mode=rw_mode)
            txt_mod1, txt_mod2 = attn.txt_mod(vec, rw_mode=rw_mode)

            # prepare image for attention
            img_modulated = attn.img_norm1(img)
            img_modulated = (1 + img_mod1.scale) * img_modulated + img_mod1.shift
            img_qkv = attn.img_attn.qkv(img_modulated)
            img_q, img_k, img_v = rearrange(img_qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads, D=attn.head_dim)
            img_q, img_k = attn.img_attn.norm(img_q, img_k, img_v)

            # prepare txt for attention
            txt_modulated = attn.txt_norm1(txt)
            txt_modulated = (1 + txt_mod1.scale) * txt_modulated + txt_mod1.shift
            txt_qkv = attn.txt_attn.qkv(txt_modulated)
            txt_q, txt_k, txt_v = rearrange(txt_qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads, D=attn.head_dim)
            txt_q, txt_k = attn.txt_attn.norm(txt_q, txt_k, txt_v)

            # run actual attention
            q = torch.cat((txt_q, img_q), dim=2)
            k = torch.cat((txt_k, img_k), dim=2)
            v = torch.cat((txt_v, img_v), dim=2)

            attn1 = attention(q, k, v, pe=pe)

            txt_attn, img_attn = attn1[:, :txt.shape[1]], attn1[:, txt.shape[1]:txt.shape[1]+img.shape[1]]

            # calculate the img bloks
            img = img + img_mod1.gate * attn.img_attn.proj(img_attn)
            img = img + img_mod2.gate * attn.img_mlp((1 + img_mod2.scale) * attn.img_norm2(img) + img_mod2.shift)

            # calculate the txt bloks
            txt = txt + txt_mod1.gate * attn.txt_attn.proj(txt_attn)
            txt = txt + txt_mod2.gate * attn.txt_mlp((1 + txt_mod2.scale) * attn.txt_norm2(txt) + txt_mod2.shift)

        return img, txt

class IPDoubleStreamBlockProcessor(ModelMixin):
    """Attention processor for handling IP-adapter with double stream block."""

    def __init__(self, context_dim, hidden_dim):
        super().__init__()
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "IPDoubleStreamBlockProcessor requires PyTorch 2.0 or higher. Please upgrade PyTorch."
            )
        
        # Ensure context_dim matches the dimension of image_proj
        self.context_dim = context_dim
        self.hidden_dim = hidden_dim

        # Initialize projections for IP-adapter
        self.ip_adapter_double_stream_k_proj = nn.Linear(context_dim, hidden_dim, bias=True)
        self.ip_adapter_double_stream_v_proj = nn.Linear(context_dim, hidden_dim, bias=True)
        
        nn.init.zeros_(self.ip_adapter_double_stream_k_proj.weight)
        nn.init.zeros_(self.ip_adapter_double_stream_k_proj.bias)
        
        nn.init.zeros_(self.ip_adapter_double_stream_v_proj.weight)
        nn.init.zeros_(self.ip_adapter_double_stream_v_proj.bias)
        
    def __call__(self, attn, img, txt, vec, pe, image_proj, ip_scale=1.0, **attention_kwargs):

        # Prepare image for attention
        img_mod1, img_mod2 = attn.img_mod(vec)
        txt_mod1, txt_mod2 = attn.txt_mod(vec)

        img_modulated = attn.img_norm1(img)
        img_modulated = (1 + img_mod1.scale) * img_modulated + img_mod1.shift
        img_qkv = attn.img_attn.qkv(img_modulated) ## [1, 2400, 3072] -> [1, 2400, 9216], because hidden_dim = 3072, qkv = 3

        ## L = h/16 * w/16, H = num_heads, D = head_dim
        img_q, img_k, img_v = rearrange(img_qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads, D=attn.head_dim) 
        img_q, img_k = attn.img_attn.norm(img_q, img_k, img_v)

        txt_modulated = attn.txt_norm1(txt)
        txt_modulated = (1 + txt_mod1.scale) * txt_modulated + txt_mod1.shift
        txt_qkv = attn.txt_attn.qkv(txt_modulated)
        
        txt_q, txt_k, txt_v = rearrange(txt_qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads, D=attn.head_dim)
        txt_q, txt_k = attn.txt_attn.norm(txt_q, txt_k, txt_v)

        q = torch.cat((txt_q, img_q), dim=2)
        k = torch.cat((txt_k, img_k), dim=2)
        v = torch.cat((txt_v, img_v), dim=2)
        
        attn1 = attention(q, k, v, pe=pe)
        txt_attn, img_attn = attn1[:, :txt.shape[1]], attn1[:, txt.shape[1]:]
 
        img = img + img_mod1.gate * attn.img_attn.proj(img_attn)

        ### split
        x = (1 + img_mod2.scale) * attn.img_norm2(img) + img_mod2.shift
        x = attn.img_mlp[0](x)  
        x = attn.img_mlp[1](x)
        x = attn.img_mlp[2](x) 

        img = img + img_mod2.gate * x

        txt = txt + txt_mod1.gate * attn.txt_attn.proj(txt_attn)
        txt = txt + txt_mod2.gate * attn.txt_mlp((1 + txt_mod2.scale) * attn.txt_norm2(txt) + txt_mod2.shift)
        
        # IP-adapter processing
        ip_query = img_q  # latent sample query

        ip_key   = self.ip_adapter_double_stream_k_proj(image_proj)
        ip_value = self.ip_adapter_double_stream_v_proj(image_proj)
        
        # Reshape projections for multi-head attention
        ip_key = rearrange(ip_key, 'B L (H D) -> B H L D', H=attn.num_heads, D=attn.head_dim)
        ip_value = rearrange(ip_value, 'B L (H D) -> B H L D', H=attn.num_heads, D=attn.head_dim)

        # Compute attention between IP projections and the latent query
        ip_attention = F.scaled_dot_product_attention(
            ip_query, 
            ip_key, 
            ip_value, 
            dropout_p=0.0, 
            is_causal=False
        )
 
        ip_attention = rearrange(ip_attention, "B H L D -> B L (H D)", H=attn.num_heads, D=attn.head_dim)
        
        img = img + ip_scale * ip_attention 

        return img, txt
    
class DoubleStreamBlockProcessor:
    def __call__(self, attn, img, txt, vec, pe, **attention_kwargs):
        img_mod1, img_mod2 = attn.img_mod(vec)
        txt_mod1, txt_mod2 = attn.txt_mod(vec)

        # prepare image for attention
        img_modulated = attn.img_norm1(img)
        img_modulated = (1 + img_mod1.scale) * img_modulated + img_mod1.shift
        img_qkv = attn.img_attn.qkv(img_modulated)
        img_q, img_k, img_v = rearrange(img_qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads, D=attn.head_dim)
        img_q, img_k = attn.img_attn.norm(img_q, img_k, img_v)

        # prepare txt for attention
        txt_modulated = attn.txt_norm1(txt)
        txt_modulated = (1 + txt_mod1.scale) * txt_modulated + txt_mod1.shift
        txt_qkv = attn.txt_attn.qkv(txt_modulated)
        txt_q, txt_k, txt_v = rearrange(txt_qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads, D=attn.head_dim)
        txt_q, txt_k = attn.txt_attn.norm(txt_q, txt_k, txt_v)

        # run actual attention
        q = torch.cat((txt_q, img_q), dim=2)
        k = torch.cat((txt_k, img_k), dim=2)
        v = torch.cat((txt_v, img_v), dim=2)

        attn1 = attention(q, k, v, pe=pe)
        txt_attn, img_attn = attn1[:, : txt.shape[1]], attn1[:, txt.shape[1] :]

        # calculate the img bloks
        img = img + img_mod1.gate * attn.img_attn.proj(img_attn)
        img = img + img_mod2.gate * attn.img_mlp((1 + img_mod2.scale) * attn.img_norm2(img) + img_mod2.shift)

        # calculate the txt bloks
        txt = txt + txt_mod1.gate * attn.txt_attn.proj(txt_attn)
        txt = txt + txt_mod2.gate * attn.txt_mlp((1 + txt_mod2.scale) * attn.txt_norm2(txt) + txt_mod2.shift)
        return img, txt

class DoubleStreamBlock(ModelMixin):
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float, qkv_bias: bool = False):
        super().__init__()
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.head_dim = hidden_size // num_heads
        
        self.img_mod = Modulation(hidden_size, double=True)
        self.img_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_attn  = SelfAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias)

        self.img_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )

        self.img_mlp_lora_1 = LoRALinearLayer(hidden_size, mlp_hidden_dim, rank=32, network_alpha=16)
        self.img_mlp_lora_2 = LoRALinearLayer(mlp_hidden_dim, hidden_size, rank=32, network_alpha=16)

        self.txt_mod   = Modulation(hidden_size, double=True)
        self.txt_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_attn  = SelfAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias)

        self.txt_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )
        processor = DoubleStreamBlockProcessor()
        self.set_processor(processor)

    def set_processor(self, processor) -> None:
        self.processor = processor

    def get_processor(self):
        return self.processor

    def forward(
        self, 
        img: Tensor, 
        txt: Tensor, 
        vec: Tensor, 
        pe: Tensor, 
        rw_mode="write",
        image_proj: Tensor = None, 
        ip_scale: float =1.0,
    ) -> tuple[Tensor, Tensor]:

        if image_proj is None: 
            return self.processor(self, img, txt, vec, pe, rw_mode)
        else:
            return self.processor(self, img, txt, vec, pe, image_proj, ip_scale)


class IPSingleStreamBlockProcessor(ModelMixin): 
    """Attention processor for handling IP-adapter with single stream block."""
    def __init__(self, context_dim, hidden_dim):
        super().__init__()
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "IPSingleStreamBlockProcessor requires PyTorch 2.0 or higher. Please upgrade PyTorch."
            )

        # Ensure context_dim matches the dimension of image_proj
        self.context_dim = context_dim
        self.hidden_dim = hidden_dim

        # Initialize projections for IP-adapter
        self.ip_adapter_single_stream_k_proj = nn.Linear(context_dim, hidden_dim, bias=False)
        self.ip_adapter_single_stream_v_proj = nn.Linear(context_dim, hidden_dim, bias=False)

        nn.init.zeros_(self.ip_adapter_single_stream_k_proj.weight)
        nn.init.zeros_(self.ip_adapter_single_stream_v_proj.weight)
        
    def __call__(
        self, 
        attn: ModelMixin, 
        x: Tensor, 
        vec: Tensor, 
        pe: Tensor, 
        image_proj: Tensor | None = None, 
        ip_scale: float = 1.0
    ) -> Tensor:
        
        mod, _ = attn.modulation(vec)
        x_mod = (1 + mod.scale) * attn.pre_norm(x) + mod.shift
        qkv, mlp = torch.split(attn.linear1(x_mod), [3 * attn.hidden_size, attn.mlp_hidden_dim], dim=-1)

        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads, D=attn.head_dim)
        q, k = attn.norm(q, k, v)

        # compute attention
        attn_1 = attention(q, k, v, pe=pe)
        
        # IP-adapter processing
        ip_query = q  
        ip_key = self.ip_adapter_single_stream_k_proj(image_proj)
        ip_value = self.ip_adapter_single_stream_v_proj(image_proj)

        # Reshape projections for multi-head attention
        ip_key = rearrange(ip_key, 'B L (H D) -> B H L D', H=attn.num_heads, D=attn.head_dim)
        ip_value = rearrange(ip_value, 'B L (H D) -> B H L D', H=attn.num_heads, D=attn.head_dim)
        
        # Compute attention between IP projections and the latent query
        ip_attention = F.scaled_dot_product_attention(
            ip_query, 
            ip_key, 
            ip_value
        )
        ip_attention = rearrange(ip_attention, "B H L D -> B L (H D)")
        
        attn_out = attn_1 + ip_scale * ip_attention
        
        # compute activation in mlp stream, cat again and run second linear layer
        output = attn.linear2(torch.cat((attn_out, attn.mlp_act(mlp)), 2))
        out = x + mod.gate * output
        
        return out

class SingleStreamBlockLoraProcessor(ModelMixin):
    def __init__(self, dim: int, rank: int = 4, network_alpha = None, lora_weight: float = 1, ip_scale=1.0, device=None, dtype=None):
        super().__init__()

        self.ref_qkv_lora_q  = LoRALinearLayer(dim, dim, rank, network_alpha, device=device, dtype=dtype)
        self.ref_qkv_lora_k  = LoRALinearLayer(dim, dim, rank, network_alpha, device=device, dtype=dtype)
        self.ref_qkv_lora_v  = LoRALinearLayer(dim, dim, rank, network_alpha, device=device, dtype=dtype)

        self.ref_proj_lora = LoRALinearLayer(15360, dim, rank*2, network_alpha*2, device=device, dtype=dtype)

        self.lora_weight = lora_weight
        self.ip_scale = ip_scale

        self.bank_img_q = None
        self.bank_img_k = None
        self.bank_img_v = None
        
        self.bank_neg_img_q = None
        self.bank_neg_img_k = None
        self.bank_neg_img_v = None

    def forward(self, attn: ModelMixin, x: Tensor, vec: Tensor, pe: Tensor, rw_mode: str) -> Tensor: # [txt, img]
        if "write" in rw_mode:
            mod, _ = attn.modulation(vec, rw_mode=rw_mode)
            x_mod = (1 + mod.scale) * attn.pre_norm(x) + mod.shift
            qkv, mlp = torch.split(attn.linear1(x_mod), [3 * attn.hidden_size, attn.mlp_hidden_dim], dim=-1)

            ### use_lora
            q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads)

            ### use_lora
            ref_lora_q = self.ref_qkv_lora_q(x_mod) * self.lora_weight
            q = q + rearrange(ref_lora_q, "B L (H D) -> B H L D", H=attn.num_heads)

            ref_lora_k = self.ref_qkv_lora_k(x_mod) * self.lora_weight
            k = k + rearrange(ref_lora_k, "B L (H D) -> B H L D", H=attn.num_heads)

            ref_lora_v = self.ref_qkv_lora_v(x_mod) * self.lora_weight
            v = v + rearrange(ref_lora_v, "B L (H D) -> B H L D", H=attn.num_heads)

            q, k = attn.norm(q, k, v)

            if rw_mode == "write":
                self.bank_img_q = q # [txt, img]
                self.bank_img_k = k
                self.bank_img_v = v
            else:
                self.bank_neg_img_q = q
                self.bank_neg_img_k = k
                self.bank_neg_img_v = v

            # compute attention
            attn_1 = attention(q, k, v, pe=pe)

            # compute activation in mlp stream, cat again and run second linear layer
            output = attn.linear2(torch.cat((attn_1, attn.mlp_act(mlp)), 2))

            ### use_lora
            output = output + self.ref_proj_lora(torch.cat((attn_1, attn.mlp_act(mlp)), 2)) * self.lora_weight
            output = x + mod.gate * output
        
        elif "read" in rw_mode:
            mod, _ = attn.modulation(vec, rw_mode=rw_mode)
            x_mod = (1 + mod.scale) * attn.pre_norm(x) + mod.shift
            qkv, mlp = torch.split(attn.linear1(x_mod), [3 * attn.hidden_size, attn.mlp_hidden_dim], dim=-1)
        
            q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads)

            q, k = attn.norm(q, k, v)
 
            if rw_mode == "read":
                q = torch.cat((q, self.bank_img_q), dim=2) # cat([txt ,img, txt_ref, img_ref])
                k = torch.cat((k, self.bank_img_k), dim=2)
                v = torch.cat((v, self.bank_img_v), dim=2)
            else:
                q = torch.cat((q, self.bank_neg_img_q), dim=2)
                k = torch.cat((k, self.bank_neg_img_k), dim=2)
                v = torch.cat((v, self.bank_neg_img_v), dim=2)

            attn_cat = attention(q, k, v, pe=pe)

            attn_1 = attn_cat[:, :x.shape[1], :]

            # compute activation in mlp stream, cat again and run second linear layer
            output = attn.linear2(torch.cat((attn_1, attn.mlp_act(mlp)), 2))
            output = x + mod.gate * output

        else:
            mod, _ = attn.modulation(vec, rw_mode=rw_mode)
            x_mod = (1 + mod.scale) * attn.pre_norm(x) + mod.shift
            qkv, mlp = torch.split(attn.linear1(x_mod), [3 * attn.hidden_size, attn.mlp_hidden_dim], dim=-1)
        
            q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads)

            q, k = attn.norm(q, k, v)
 
            attn_cat = attention(q, k, v, pe=pe)

            attn_1 = attn_cat[:, :x.shape[1], :]

            # compute activation in mlp stream, cat again and run second linear layer
            output = attn.linear2(torch.cat((attn_1, attn.mlp_act(mlp)), 2))
            output = x + mod.gate * output            

        return output


class SingleStreamBlockProcessor: 
    def __call__(self, attn: ModelMixin, x: Tensor, vec: Tensor, pe: Tensor, rw_mode: str) -> Tensor:
        
        mod, _ = attn.modulation(vec)
        x_mod = (1 + mod.scale) * attn.pre_norm(x) + mod.shift
        qkv, mlp = torch.split(attn.linear1(x_mod), [3 * attn.hidden_size, attn.mlp_hidden_dim], dim=-1)

        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads)
        q, k = attn.norm(q, k, v)

        # compute attention
        attn_1 = attention(q, k, v, pe=pe)
        
        # compute activation in mlp stream, cat again and run second linear layer
        output = attn.linear2(torch.cat((attn_1, attn.mlp_act(mlp)), 2))
        output = x + mod.gate * output
        return output
        
class SingleStreamBlock(ModelMixin):
    """
    A DiT block with parallel linear layers as described in
    https://arxiv.org/abs/2302.05442 and adapted modulation interface.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qk_scale: float | None = None,
    ):
        super().__init__()
        self.hidden_dim = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = qk_scale or self.head_dim**-0.5

        self.mlp_hidden_dim = int(hidden_size * mlp_ratio)
        # qkv and mlp_in
        self.linear1 = nn.Linear(hidden_size, hidden_size * 3 + self.mlp_hidden_dim)
 
        # proj and mlp_out
        self.linear2 = nn.Linear(hidden_size + self.mlp_hidden_dim, hidden_size)
 
        self.norm = QKNorm(self.head_dim)

        self.hidden_size = hidden_size
        self.pre_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        self.mlp_act = nn.GELU(approximate="tanh")
        self.modulation = Modulation(hidden_size, double=False)
        
        processor = SingleStreamBlockProcessor()
        self.set_processor(processor)
    
    def set_processor(self, processor) -> None:
        self.processor = processor

    def get_processor(self):
        return self.processor
    
    def forward(
        self, 
        x: Tensor, 
        vec: Tensor, 
        pe: Tensor, 
        rw_mode="write",
        image_proj: Tensor | None = None, 
        ip_scale: float = 1.0
    ) -> Tensor:
        if image_proj is None: 
            return self.processor(self, x, vec, pe, rw_mode)
        else:
            return self.processor(self, x, vec, pe, image_proj, ip_scale)

class LastLayer(ModelMixin):
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))

    def forward(self, x: Tensor, vec: Tensor) -> Tensor:
        shift, scale = self.adaLN_modulation(vec).chunk(2, dim=1)

        x = (1 + scale[:, None, :]) * self.norm_final(x) + shift[:, None, :]
 
        x = self.linear(x)

        return x

class ImageProjModel(ModelMixin):
    """Projection Model
    https://github.com/tencent-ailab/IP-Adapter/blob/main/ip_adapter/ip_adapter.py#L28
    """

    def __init__(self, cross_attention_dim=1024, clip_embeddings_dim=1024, clip_extra_context_tokens=4):
        super().__init__()

        self.generator = None
        self.cross_attention_dim = cross_attention_dim
        self.clip_extra_context_tokens = clip_extra_context_tokens
        self.proj = torch.nn.Linear(clip_embeddings_dim, self.clip_extra_context_tokens * cross_attention_dim)
        self.norm = torch.nn.LayerNorm(cross_attention_dim)

    def forward(self, image_embeds):
        embeds = image_embeds

        if self.clip_extra_context_tokens == 1:
            clip_extra_context_tokens = self.proj(embeds)
        else:
            clip_extra_context_tokens = self.proj(embeds).reshape(
                -1, self.clip_extra_context_tokens, self.cross_attention_dim
            )
 
        clip_extra_context_tokens = self.norm(clip_extra_context_tokens)
        return clip_extra_context_tokens

