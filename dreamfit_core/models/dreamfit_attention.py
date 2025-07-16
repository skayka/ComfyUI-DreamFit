"""
DreamFit Attention Processors for ComfyUI
Based on the official DreamFit implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Any, Tuple
from einops import rearrange
import math
try:
    import comfy.model_management
except ImportError:
    # Fallback for testing without ComfyUI
    class MockModelManagement:
        @staticmethod
        def intermediate_device():
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    comfy = type('comfy', (), {'model_management': MockModelManagement})()


class DreamFitDoubleStreamProcessor(nn.Module):
    """
    Attention processor for Flux double-stream blocks with DreamFit read/write mechanism.
    Based on DoubleStreamBlockLoraProcessor from official implementation.
    """
    
    def __init__(self, hidden_size: int = 3072, num_heads: int = 24, rank: int = 32, network_alpha: Optional[float] = None, lora_weight: float = 1.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.rank = rank
        self.network_alpha = network_alpha or rank
        self.lora_weight = lora_weight
        
        # Mode for read/write mechanism
        self.current_mode = "normal"
        
        # LoRA layers for garment feature adaptation
        self.ref_qkv_lora_q = LoRALinearLayer(hidden_size, hidden_size, rank, network_alpha)
        self.ref_qkv_lora_k = LoRALinearLayer(hidden_size, hidden_size, rank, network_alpha)
        self.ref_qkv_lora_v = LoRALinearLayer(hidden_size, hidden_size, rank, network_alpha)
        self.ref_proj_lora = LoRALinearLayer(hidden_size, hidden_size, rank, network_alpha)
        
        # Storage for garment features (read/write mechanism)
        self.bank_img_q = None
        self.bank_img_k = None
        self.bank_img_v = None
        self.bank_neg_img_q = None
        self.bank_neg_img_k = None
        self.bank_neg_img_v = None
        
    def __call__(self, attn, img, txt, vec, pe, **kwargs):
        """
        Process attention with DreamFit read/write mechanism.
        
        Args:
            attn: The attention module
            img: Image features
            txt: Text features
            vec: Time/guidance embeddings
            pe: Positional encoding
        """
        # Use current_mode instead of rw_mode parameter
        rw_mode = getattr(self, 'current_mode', 'normal')
        
        if rw_mode in ["write", "neg_write"]:
            return self._forward_write_mode(attn, img, txt, vec, pe, rw_mode)
        elif rw_mode in ["read", "neg_read"]:
            return self._forward_read_mode(attn, img, txt, vec, pe, rw_mode)
        else:
            return self._forward_normal_mode(attn, img, txt, vec, pe)
    
    def _forward_write_mode(self, attn, img, txt, vec, pe, rw_mode):
        """Write mode: Store garment features"""
        # Get modulation
        img_mod1, img_mod2 = attn.img_mod(vec)
        txt_mod1, txt_mod2 = attn.txt_mod(vec)
        
        # Prepare image for attention
        img_modulated = attn.img_norm1(img)
        img_modulated = (1 + img_mod1.scale) * img_modulated + img_mod1.shift
        
        # Apply QKV projection
        img_qkv = attn.img_attn.qkv(img_modulated)
        img_q, img_k, img_v = rearrange(img_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        
        # Apply LoRA adaptation for garment features
        ref_lora_q = self.ref_qkv_lora_q(img_modulated) * self.lora_weight
        img_q = img_q + rearrange(ref_lora_q, "B L (H D) -> B H L D", H=self.num_heads)
        
        ref_lora_k = self.ref_qkv_lora_k(img_modulated) * self.lora_weight
        img_k = img_k + rearrange(ref_lora_k, "B L (H D) -> B H L D", H=self.num_heads)
        
        ref_lora_v = self.ref_qkv_lora_v(img_modulated) * self.lora_weight
        img_v = img_v + rearrange(ref_lora_v, "B L (H D) -> B H L D", H=self.num_heads)
        
        # Normalize Q, K
        img_q, img_k = attn.img_attn.norm(img_q, img_k, img_v)
        
        # Store features
        if rw_mode == "write":
            self.bank_img_q = img_q
            self.bank_img_k = img_k
            self.bank_img_v = img_v
            print(f"DreamFitProcessor: Stored features in write mode - Q shape: {img_q.shape}")
        else:  # neg_write
            self.bank_neg_img_q = img_q
            self.bank_neg_img_k = img_k
            self.bank_neg_img_v = img_v
            print(f"DreamFitProcessor: Stored features in neg_write mode - Q shape: {img_q.shape}")
        
        # Process text
        txt_modulated = attn.txt_norm1(txt)
        txt_modulated = (1 + txt_mod1.scale) * txt_modulated + txt_mod1.shift
        txt_qkv = attn.txt_attn.qkv(txt_modulated)
        txt_q, txt_k, txt_v = rearrange(txt_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        txt_q, txt_k = attn.txt_attn.norm(txt_q, txt_k, txt_v)
        
        # Run attention
        q = torch.cat((txt_q, img_q), dim=2)
        k = torch.cat((txt_k, img_k), dim=2)
        v = torch.cat((txt_v, img_v), dim=2)
        
        attn_out = attention(q, k, v, pe=pe)
        txt_attn, img_attn = attn_out[:, :txt.shape[1]], attn_out[:, txt.shape[1]:]
        
        # Apply output projections with LoRA
        img = img + img_mod1.gate * (attn.img_attn.proj(img_attn) + self.ref_proj_lora(img_attn) * self.lora_weight)
        img = img + img_mod2.gate * attn.img_mlp((1 + img_mod2.scale) * attn.img_norm2(img) + img_mod2.shift)
        
        txt = txt + txt_mod1.gate * attn.txt_attn.proj(txt_attn)
        txt = txt + txt_mod2.gate * attn.txt_mlp((1 + txt_mod2.scale) * attn.txt_norm2(txt) + txt_mod2.shift)
        
        return img, txt
    
    def _forward_read_mode(self, attn, img, txt, vec, pe, rw_mode):
        """Read mode: Use stored garment features"""
        # Get modulation
        img_mod1, img_mod2 = attn.img_mod(vec)
        txt_mod1, txt_mod2 = attn.txt_mod(vec)
        
        # Prepare current features
        img_modulated = attn.img_norm1(img)
        img_modulated = (1 + img_mod1.scale) * img_modulated + img_mod1.shift
        img_qkv = attn.img_attn.qkv(img_modulated)
        img_q, img_k, img_v = rearrange(img_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        img_q, img_k = attn.img_attn.norm(img_q, img_k, img_v)
        
        txt_modulated = attn.txt_norm1(txt)
        txt_modulated = (1 + txt_mod1.scale) * txt_modulated + txt_mod1.shift
        txt_qkv = attn.txt_attn.qkv(txt_modulated)
        txt_q, txt_k, txt_v = rearrange(txt_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        txt_q, txt_k = attn.txt_attn.norm(txt_q, txt_k, txt_v)
        
        # Get stored features
        if rw_mode == "read":
            ref_q = self.bank_img_q
            ref_k = self.bank_img_k
            ref_v = self.bank_img_v
            print(f"DreamFitProcessor: Using stored features in read mode - Q shape: {ref_q.shape if ref_q is not None else 'None'}")
        else:  # neg_read
            ref_q = self.bank_neg_img_q
            ref_k = self.bank_neg_img_k
            ref_v = self.bank_neg_img_v
            print(f"DreamFitProcessor: Using stored features in neg_read mode - Q shape: {ref_q.shape if ref_q is not None else 'None'}")
        
        # Validate stored features
        if ref_q is None or ref_k is None or ref_v is None:
            print(f"Warning: No stored features for {rw_mode} mode, falling back to normal mode")
            return self._forward_normal_mode(attn, img, txt, vec, pe)
        
        # Ensure same device and dtype
        device = img_q.device
        dtype = img_q.dtype
        
        # Concatenate with stored features
        q = torch.cat((txt_q, img_q, ref_q.to(device, dtype)), dim=2)
        k = torch.cat((txt_k, img_k, ref_k.to(device, dtype)), dim=2)
        v = torch.cat((txt_v, img_v, ref_v.to(device, dtype)), dim=2)
        
        # Create attention mask to prevent text from attending to reference
        B, H, L_txt, D = txt_q.shape
        L_img = img_q.shape[2]
        L_ref = ref_q.shape[2]
        L_total = L_txt + L_img + L_ref
        
        attn_mask = torch.ones((B, H, L_total, L_total), dtype=torch.bool, device=q.device)
        attn_mask[:, :, :L_txt, -L_ref:] = 0  # Text can't attend to reference
        attn_mask[:, :, -L_ref:, :L_txt] = 0  # Reference can't attend to text
        
        # Run attention with mask
        attn_out = attention(q, k, v, pe=pe, attn_mask=attn_mask)
        txt_attn = attn_out[:, :L_txt]
        img_attn = attn_out[:, L_txt:L_txt+L_img]
        
        # Apply output projections
        img = img + img_mod1.gate * attn.img_attn.proj(img_attn)
        img = img + img_mod2.gate * attn.img_mlp((1 + img_mod2.scale) * attn.img_norm2(img) + img_mod2.shift)
        
        txt = txt + txt_mod1.gate * attn.txt_attn.proj(txt_attn)
        txt = txt + txt_mod2.gate * attn.txt_mlp((1 + txt_mod2.scale) * attn.txt_norm2(txt) + txt_mod2.shift)
        
        return img, txt
    
    def _forward_normal_mode(self, attn, img, txt, vec, pe):
        """Normal mode: Standard attention without garment features"""
        # This is a simplified version - in practice, would call the original processor
        # For now, return inputs unchanged
        print("Warning: Using fallback normal mode in DreamFitDoubleStreamProcessor")
        return img, txt
    
    def reset(self):
        """Reset stored features"""
        self.bank_img_q = None
        self.bank_img_k = None
        self.bank_img_v = None
        self.bank_neg_img_q = None
        self.bank_neg_img_k = None
        self.bank_neg_img_v = None


class DreamFitSingleStreamProcessor(nn.Module):
    """
    Attention processor for Flux single-stream blocks with DreamFit read/write mechanism.
    Based on SingleStreamBlockLoraProcessor from official implementation.
    """
    
    def __init__(self, hidden_size: int = 3072, num_heads: int = 24, rank: int = 32, network_alpha: Optional[float] = None, lora_weight: float = 1.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.rank = rank
        self.network_alpha = network_alpha or rank
        self.lora_weight = lora_weight
        
        # Mode for read/write mechanism
        self.current_mode = "normal"
        
        # LoRA layers
        self.ref_qkv_lora_q = LoRALinearLayer(hidden_size, hidden_size, rank, network_alpha)
        self.ref_qkv_lora_k = LoRALinearLayer(hidden_size, hidden_size, rank, network_alpha)
        self.ref_qkv_lora_v = LoRALinearLayer(hidden_size, hidden_size, rank, network_alpha)
        self.ref_proj_lora = LoRALinearLayer(hidden_size * 5, hidden_size, rank * 2, network_alpha * 2)
        
        # Storage
        self.bank_img_q = None
        self.bank_img_k = None
        self.bank_img_v = None
        self.bank_neg_img_q = None
        self.bank_neg_img_k = None
        self.bank_neg_img_v = None
    
    def __call__(self, attn, x, vec, pe, **kwargs):
        """Process single-stream attention with read/write"""
        # Use current_mode instead of rw_mode parameter
        rw_mode = getattr(self, 'current_mode', 'normal')
        
        if rw_mode in ["write", "neg_write"]:
            return self._forward_write_mode(attn, x, vec, pe, rw_mode)
        elif rw_mode in ["read", "neg_read"]:
            return self._forward_read_mode(attn, x, vec, pe, rw_mode)
        else:
            return self._forward_normal_mode(attn, x, vec, pe)
    
    def _forward_write_mode(self, attn, x, vec, pe, rw_mode):
        """Write mode for single-stream"""
        mod, _ = attn.modulation(vec)
        x_mod = (1 + mod.scale) * attn.pre_norm(x) + mod.shift
        qkv, mlp = torch.split(attn.linear1(x_mod), [3 * self.hidden_size, attn.mlp_hidden_dim], dim=-1)
        
        # Apply QKV
        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        
        # Apply LoRA
        ref_lora_q = self.ref_qkv_lora_q(x_mod) * self.lora_weight
        q = q + rearrange(ref_lora_q, "B L (H D) -> B H L D", H=self.num_heads)
        
        ref_lora_k = self.ref_qkv_lora_k(x_mod) * self.lora_weight
        k = k + rearrange(ref_lora_k, "B L (H D) -> B H L D", H=self.num_heads)
        
        ref_lora_v = self.ref_qkv_lora_v(x_mod) * self.lora_weight
        v = v + rearrange(ref_lora_v, "B L (H D) -> B H L D", H=self.num_heads)
        
        q, k = attn.norm(q, k, v)
        
        # Store features
        if rw_mode == "write":
            self.bank_img_q = q
            self.bank_img_k = k
            self.bank_img_v = v
        else:
            self.bank_neg_img_q = q
            self.bank_neg_img_k = k
            self.bank_neg_img_v = v
        
        # Compute attention
        attn_out = attention(q, k, v, pe=pe)
        
        # Apply output with LoRA
        output = attn.linear2(torch.cat((attn_out, attn.mlp_act(mlp)), 2))
        output = output + self.ref_proj_lora(torch.cat((attn_out, attn.mlp_act(mlp)), 2)) * self.lora_weight
        
        return x + mod.gate * output
    
    def _forward_read_mode(self, attn, x, vec, pe, rw_mode):
        """Read mode for single-stream"""
        mod, _ = attn.modulation(vec)
        x_mod = (1 + mod.scale) * attn.pre_norm(x) + mod.shift
        qkv, mlp = torch.split(attn.linear1(x_mod), [3 * self.hidden_size, attn.mlp_hidden_dim], dim=-1)
        
        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        q, k = attn.norm(q, k, v)
        
        # Get stored features
        if rw_mode == "read":
            ref_q = self.bank_img_q
            ref_k = self.bank_img_k
            ref_v = self.bank_img_v
        else:
            ref_q = self.bank_neg_img_q
            ref_k = self.bank_neg_img_k
            ref_v = self.bank_neg_img_v
        
        # Validate stored features
        if ref_q is None or ref_k is None or ref_v is None:
            print(f"Warning: No stored features for {rw_mode} mode in single-stream, falling back to normal mode")
            return self._forward_normal_mode(attn, x, vec, pe)
        
        # Ensure same device and dtype
        device = q.device
        dtype = q.dtype
        
        # Concatenate stored features
        q = torch.cat((q, ref_q.to(device, dtype)), dim=2)
        k = torch.cat((k, ref_k.to(device, dtype)), dim=2)
        v = torch.cat((v, ref_v.to(device, dtype)), dim=2)
        
        # Attention
        attn_cat = attention(q, k, v, pe=pe)
        attn_out = attn_cat[:, :x.shape[1], :]
        
        # Output
        output = attn.linear2(torch.cat((attn_out, attn.mlp_act(mlp)), 2))
        return x + mod.gate * output
    
    def _forward_normal_mode(self, attn, x, vec, pe):
        """Normal mode"""
        print("Warning: Using fallback normal mode in DreamFitSingleStreamProcessor")
        return x
    
    def reset(self):
        """Reset stored features"""
        self.bank_img_q = None
        self.bank_img_k = None
        self.bank_img_v = None
        self.bank_neg_img_q = None
        self.bank_neg_img_k = None
        self.bank_neg_img_v = None


class LoRALinearLayer(nn.Module):
    """LoRA Linear layer implementation"""
    
    def __init__(self, in_features: int, out_features: int, rank: int = 4, network_alpha: Optional[float] = None):
        super().__init__()
        self.rank = rank
        self.network_alpha = network_alpha or rank
        
        self.down = nn.Linear(in_features, rank, bias=False)
        self.up = nn.Linear(rank, out_features, bias=False)
        
        # Initialize weights
        nn.init.normal_(self.down.weight, std=1 / rank)
        nn.init.zeros_(self.up.weight)
    
    def forward(self, x):
        down_hidden = self.down(x)
        up_hidden = self.up(down_hidden)
        
        if self.network_alpha is not None:
            up_hidden *= self.network_alpha / self.rank
        
        return up_hidden


def attention(q, k, v, pe, attn_mask=None):
    """
    Compute attention with positional encoding and optional mask.
    Based on the official DreamFit implementation.
    """
    # Apply RoPE (Rotary Position Embedding)
    q, k = apply_rope(q, k, pe)
    
    # Use PyTorch's built-in scaled dot-product attention
    # This is more efficient and handles various optimizations
    x = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
    
    # Reshape from B H L D -> B L (H D)
    x = rearrange(x, "B H L D -> B L (H D)")
    
    return x


def apply_rope(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary position embedding to query and key tensors.
    Based on the official Flux implementation.
    """
    if freqs_cis is None:
        return xq, xk
    
    # Reshape for RoPE application
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
    
    # Apply rotary embedding
    xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
    xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
    
    # Reshape back and preserve dtype
    return xq_out.reshape(*xq.shape).type_as(xq), xk_out.reshape(*xk.shape).type_as(xk)