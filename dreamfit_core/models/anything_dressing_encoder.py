"""
Anything-Dressing Encoder for DreamFit
A lightweight encoder (83.4M parameters) for garment feature extraction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass


@dataclass
class EncoderConfig:
    """Configuration for Anything-Dressing Encoder"""
    hidden_dim: int = 1024
    num_layers: int = 12
    num_heads: int = 16
    mlp_ratio: float = 4.0
    patch_size: int = 14
    image_size: int = 224
    dropout: float = 0.0
    use_checkpoint: bool = False
    # Cross-attention settings
    cross_attention_layers: List[int] = None
    cross_attention_dim: int = 768
    # Feature extraction settings
    feature_layers: List[int] = None
    pool_type: str = "avg"  # avg, max, cls
    
    def __post_init__(self):
        if self.cross_attention_layers is None:
            self.cross_attention_layers = [4, 7, 10]  # Default layers for cross-attention
        if self.feature_layers is None:
            self.feature_layers = [4, 8, 11]  # Default layers for feature extraction


class PatchEmbed(nn.Module):
    """Convert image to patch embeddings"""
    
    def __init__(self, img_size=224, patch_size=14, in_chans=3, embed_dim=1024):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)  # B, embed_dim, H/P, W/P
        x = x.flatten(2).transpose(1, 2)  # B, N, embed_dim
        return x


class Attention(nn.Module):
    """Multi-head self-attention with optional cross-attention"""
    
    def __init__(self, dim, num_heads=8, qkv_bias=True, attn_drop=0., proj_drop=0., 
                 cross_attention=False, cross_dim=None):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.cross_attention = cross_attention
        
        if cross_attention:
            self.q = nn.Linear(dim, dim, bias=qkv_bias)
            self.kv = nn.Linear(cross_dim or dim, dim * 2, bias=qkv_bias)
        else:
            self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
            
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
    def forward(self, x, context=None):
        B, N, C = x.shape
        
        if self.cross_attention and context is not None:
            # Cross-attention
            q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            kv = self.kv(context).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            k, v = kv[0], kv[1]
        else:
            # Self-attention
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x, attn


class Block(nn.Module):
    """Transformer block with optional cross-attention"""
    
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 use_cross_attention=False, cross_dim=None):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, 
                             attn_drop=attn_drop, proj_drop=drop)
        
        self.use_cross_attention = use_cross_attention
        if use_cross_attention:
            self.norm_cross = nn.LayerNorm(dim)
            self.cross_attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                       attn_drop=attn_drop, proj_drop=drop,
                                       cross_attention=True, cross_dim=cross_dim)
        
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop)
        )
        
    def forward(self, x, context=None):
        # Self-attention
        x = x + self.attn(self.norm1(x))[0]
        
        # Cross-attention if enabled
        if self.use_cross_attention and context is not None:
            x = x + self.cross_attn(self.norm_cross(x), context)[0]
        
        # MLP
        x = x + self.mlp(self.norm2(x))
        
        return x


class AnythingDressingEncoder(nn.Module):
    """
    Anything-Dressing Encoder for garment feature extraction
    Lightweight (83.4M parameters) encoder that extracts multi-scale features
    """
    
    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.config = config
        
        # Patch embedding
        self.patch_embed = PatchEmbed(
            img_size=config.image_size,
            patch_size=config.patch_size,
            embed_dim=config.hidden_dim
        )
        
        # Position embedding
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.patch_embed.num_patches + 1, config.hidden_dim)
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_dim))
        self.pos_drop = nn.Dropout(config.dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            Block(
                dim=config.hidden_dim,
                num_heads=config.num_heads,
                mlp_ratio=config.mlp_ratio,
                drop=config.dropout,
                attn_drop=config.dropout,
                use_cross_attention=(i in config.cross_attention_layers),
                cross_dim=config.cross_attention_dim
            )
            for i in range(config.num_layers)
        ])
        
        # Output norm
        self.norm = nn.LayerNorm(config.hidden_dim)
        
        # Feature projections for different scales
        self.feature_projections = nn.ModuleDict({
            f"layer_{i}": nn.Linear(config.hidden_dim, config.hidden_dim)
            for i in config.feature_layers
        })
        
        # Garment-specific heads
        self.garment_token = nn.Parameter(torch.zeros(1, 1, config.hidden_dim))
        self.garment_mlp = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim * 2, config.hidden_dim)
        )
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        # Initialize position embeddings
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.garment_token, std=0.02)
        
        # Initialize other layers
        self.apply(self._init_layer_weights)
        
    def _init_layer_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the encoder
        
        Args:
            x: Input image tensor [B, C, H, W]
            context: Optional context features for cross-attention [B, L, D]
            
        Returns:
            Dictionary containing:
                - features: Multi-scale features
                - garment_token: Global garment representation
                - attention_weights: Attention maps for visualization
        """
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)  # [B, N, D]
        
        # Add cls token and position embeddings
        cls_tokens = self.cls_token.expand(B, -1, -1)
        garment_tokens = self.garment_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, garment_tokens, x], dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # Extract features at different layers
        features = {}
        attention_weights = {}
        
        for i, block in enumerate(self.blocks):
            x = block(x, context)
            
            # Extract features at specified layers
            if i in self.config.feature_layers:
                feat = self.norm(x)
                # Project features
                feat = self.feature_projections[f"layer_{i}"](feat)
                features[f"layer_{i}"] = feat
                
                # Store attention weights for visualization
                if hasattr(block, 'attn'):
                    _, attn = block.attn(block.norm1(x))
                    attention_weights[f"layer_{i}"] = attn
        
        # Final normalization
        x = self.norm(x)
        
        # Extract different tokens
        cls_output = x[:, 0]  # CLS token
        garment_output = x[:, 1]  # Garment token
        patch_output = x[:, 2:]  # Patch tokens
        
        # Process garment token
        garment_features = self.garment_mlp(garment_output)
        
        # Pool patch features
        if self.config.pool_type == "avg":
            pooled_features = patch_output.mean(dim=1)
        elif self.config.pool_type == "max":
            pooled_features = patch_output.max(dim=1)[0]
        else:  # cls
            pooled_features = cls_output
        
        # Prepare output
        output = {
            "features": features,
            "garment_token": garment_features,
            "pooled_features": pooled_features,
            "patch_features": patch_output,
            "attention_weights": attention_weights,
        }
        
        return output
    
    def extract_garment_features(self, image: torch.Tensor, 
                               return_attention: bool = False) -> torch.Tensor:
        """
        Simplified interface for extracting garment features
        
        Args:
            image: Input garment image [B, C, H, W]
            return_attention: Whether to return attention maps
            
        Returns:
            Garment features [B, D] or tuple of (features, attention_maps)
        """
        output = self.forward(image)
        features = output["garment_token"]
        
        if return_attention:
            return features, output["attention_weights"]
        return features
    
    def get_multi_scale_features(self, image: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract multi-scale features for adaptive attention injection
        
        Args:
            image: Input garment image [B, C, H, W]
            
        Returns:
            Dictionary of features at different scales
        """
        output = self.forward(image)
        return output["features"]