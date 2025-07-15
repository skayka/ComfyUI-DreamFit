"""
Adaptive Attention Injection for DreamFit
Injects garment features into Flux attention layers
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass


@dataclass
class InjectionConfig:
    """Configuration for adaptive attention injection"""
    injection_layers: List[int] = None  # Which layers to inject into
    injection_strength: float = 0.5  # Strength of injection (0-1)
    injection_mode: str = "adaptive"  # adaptive, fixed, progressive
    cross_attention_dim: int = 768  # Dimension for cross-attention
    num_heads: int = 8
    dropout: float = 0.0
    
    def __post_init__(self):
        if self.injection_layers is None:
            # Default injection layers for Flux
            self.injection_layers = [3, 6, 9, 12, 15, 18]


class AdaptiveCrossAttention(nn.Module):
    """Adaptive cross-attention module for feature injection"""
    
    def __init__(self, hidden_dim: int, context_dim: int, num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.context_dim = context_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Query, Key, Value projections
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(context_dim, hidden_dim)
        self.v_proj = nn.Linear(context_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Adaptive gating
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim + context_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        context: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply adaptive cross-attention
        
        Args:
            hidden_states: Hidden states from Flux model [B, L, D]
            context: Garment features from encoder [B, L_c, D_c]
            attention_mask: Optional attention mask
            
        Returns:
            Updated hidden states
        """
        B, L, D = hidden_states.shape
        
        # Project to multi-head format
        q = self.q_proj(hidden_states).reshape(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(context).reshape(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(context).reshape(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if attention_mask is not None:
            scores = scores + attention_mask
        
        # Apply softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).reshape(B, L, D)
        attn_output = self.out_proj(attn_output)
        
        # Compute adaptive gate
        # Pool context for gate computation
        context_pooled = context.mean(dim=1, keepdim=True).expand(-1, L, -1)
        gate_input = torch.cat([hidden_states, context_pooled], dim=-1)
        gate = self.gate(gate_input)
        
        # Apply gated injection
        output = hidden_states + gate * attn_output
        
        return output


class AdaptiveAttentionInjector(nn.Module):
    """
    Main module for injecting garment features into Flux model
    """
    
    def __init__(self, config: InjectionConfig):
        super().__init__()
        self.config = config
        
        # Create injection modules for each layer
        self.injection_modules = nn.ModuleDict()
        
        # Note: These dimensions should match Flux model dimensions
        # This is a placeholder - actual dimensions depend on Flux config
        flux_hidden_dims = {
            3: 1024,
            6: 1024,
            9: 1024,
            12: 1024,
            15: 1024,
            18: 1024
        }
        
        for layer_idx in config.injection_layers:
            hidden_dim = flux_hidden_dims.get(layer_idx, 1024)
            self.injection_modules[f"layer_{layer_idx}"] = AdaptiveCrossAttention(
                hidden_dim=hidden_dim,
                context_dim=config.cross_attention_dim,
                num_heads=config.num_heads,
                dropout=config.dropout
            )
        
        # Strength modulation
        if config.injection_mode == "adaptive":
            self.strength_modulator = nn.Sequential(
                nn.Linear(config.cross_attention_dim, 128),
                nn.SiLU(),
                nn.Linear(128, len(config.injection_layers)),
                nn.Sigmoid()
            )
        else:
            self.register_buffer(
                "strength_modulator",
                torch.ones(len(config.injection_layers)) * config.injection_strength
            )
    
    def forward(
        self,
        flux_features: Dict[str, torch.Tensor],
        garment_features: Dict[str, torch.Tensor],
        timestep: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Inject garment features into Flux features
        
        Args:
            flux_features: Dictionary of features from Flux model layers
            garment_features: Dictionary of features from garment encoder
            timestep: Optional timestep for time-dependent injection
            
        Returns:
            Updated flux features with garment information injected
        """
        updated_features = {}
        
        # Get injection strengths
        if self.config.injection_mode == "adaptive":
            # Use garment features to determine injection strength
            garment_global = garment_features.get("garment_token", garment_features.get("pooled_features"))
            strengths = self.strength_modulator(garment_global).squeeze(1)  # [B, num_layers]
        else:
            strengths = self.strength_modulator
        
        # Apply injection to each specified layer
        for i, layer_idx in enumerate(self.config.injection_layers):
            layer_key = f"layer_{layer_idx}"
            
            if layer_key in flux_features:
                flux_feat = flux_features[layer_key]
                
                # Get appropriate garment features for this layer
                # Use multi-scale features if available
                if f"layer_{layer_idx}" in garment_features.get("features", {}):
                    garment_feat = garment_features["features"][f"layer_{layer_idx}"]
                else:
                    # Fallback to patch features
                    garment_feat = garment_features.get("patch_features", garment_features["pooled_features"].unsqueeze(1))
                
                # Apply injection module
                injection_module = self.injection_modules[layer_key]
                injected_feat = injection_module(flux_feat, garment_feat)
                
                # Apply strength modulation
                if self.config.injection_mode == "progressive" and timestep is not None:
                    # Reduce injection strength over time
                    time_factor = 1.0 - (timestep / 1000.0).clamp(0, 1)
                    strength = strengths[i] * time_factor
                else:
                    strength = strengths[i] if len(strengths.shape) > 0 else strengths
                
                # Blend original and injected features
                updated_features[layer_key] = flux_feat + strength * (injected_feat - flux_feat)
            else:
                # Pass through unchanged if not injecting
                if layer_key in flux_features:
                    updated_features[layer_key] = flux_features[layer_key]
        
        # Pass through any features we didn't inject into
        for key, value in flux_features.items():
            if key not in updated_features:
                updated_features[key] = value
        
        return updated_features
    
    def prepare_injection_points(self, model: nn.Module) -> Dict[str, Any]:
        """
        Prepare injection points in the target model
        
        Args:
            model: The Flux model to inject into
            
        Returns:
            Dictionary of injection hooks and metadata
        """
        injection_hooks = {}
        
        # This is a placeholder - actual implementation would need to:
        # 1. Identify the correct layers in the Flux model
        # 2. Register forward hooks to capture features
        # 3. Set up mechanism to inject modified features
        
        # For now, return empty dict
        return injection_hooks
    
    @staticmethod
    def create_from_config(config_dict: Dict) -> 'AdaptiveAttentionInjector':
        """Create injector from configuration dictionary"""
        config = InjectionConfig(**config_dict)
        return AdaptiveAttentionInjector(config)