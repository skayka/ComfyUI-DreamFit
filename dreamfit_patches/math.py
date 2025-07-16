"""
Wrapper for math.py that makes flash_attn optional
Place this file at: DreamFit-official/src/flux/math.py
"""

import torch
from einops import rearrange
from torch import Tensor

# Try to import flash_attn, but make it optional
try:
    from flash_attn import (
        flash_attn_qkvpacked_func,
        flash_attn_func,
        flash_attn_varlen_func,
    )
    from flash_attn.bert_padding import pad_input, unpad_input
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    print("flash_attn not available, using PyTorch's built-in attention")
    FLASH_ATTN_AVAILABLE = False
    # Define dummy functions to avoid errors
    flash_attn_qkvpacked_func = None
    flash_attn_func = None
    flash_attn_varlen_func = None
    pad_input = None
    unpad_input = None


def attention(q: Tensor, k: Tensor, v: Tensor, pe: Tensor, attn_mask: Tensor=None, use_flash_attn=True) -> Tensor:
    q, k = apply_rope(q, k, pe)
    
    # Only use flash_attn if it's available AND requested
    if use_flash_attn and FLASH_ATTN_AVAILABLE:
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2) # B H L D -> B L H D
        
        x = flash_attn_func(
            q,
            k,
            v,
        )
        x = rearrange(x, "B L H D -> B L (H D)")
    else:
        # Use PyTorch's built-in attention (this is the fallback)
        x = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        x = rearrange(x, "B H L D -> B L (H D)")

    return x


def rope(pos: Tensor, dim: int, theta: int) -> Tensor:
    assert dim % 2 == 0
    scale = torch.arange(0, dim, 2, dtype=torch.float64, device=pos.device) / dim
    omega = 1.0 / (theta**scale)
    out = torch.einsum("...n,d->...nd", pos, omega)
    out = torch.stack([torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1)
    out = rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2)
    return out.float()


def apply_rope(xq: Tensor, xk: Tensor, freqs_cis: Tensor) -> tuple[Tensor, Tensor]:
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
    xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
    xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
    return xq_out.reshape(*xq.shape).type_as(xq), xk_out.reshape(*xk.shape).type_as(xk)