"""
DreamFit Model Wrapper for ComfyUI
Wraps a Flux model to add read/write mechanism for garment feature injection
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Any, Tuple
import copy


class DreamFitModelWrapper(nn.Module):
    """
    Wraps a Flux model to implement DreamFit's read/write mechanism
    """
    
    def __init__(self, base_model, garment_features: Dict[str, torch.Tensor]):
        super().__init__()
        self.base_model = base_model
        self.garment_features = garment_features
        
        # Storage for garment Q, K, V values
        self.garment_storage = {}
        
        # Flag to track if features have been written
        self.features_written = False
        
        # Current mode
        self.current_mode = "normal"
        
        # Determine model dimensions
        # Flux uses hidden_dim=3072 for double stream blocks
        self.garment_dim = 1024  # AnythingDressingEncoder output
        self.hidden_dim = 3072   # Flux hidden dimension
        
        # Initialize projection layers for garment features
        # Project from garment encoder dimension to Flux hidden dimension
        self.garment_to_flux_k_proj = nn.Linear(self.garment_dim, self.hidden_dim, bias=True)
        self.garment_to_flux_v_proj = nn.Linear(self.garment_dim, self.hidden_dim, bias=True)
        
        # Initialize projections to zero (following IP-Adapter style)
        nn.init.zeros_(self.garment_to_flux_k_proj.weight)
        nn.init.zeros_(self.garment_to_flux_k_proj.bias)
        nn.init.zeros_(self.garment_to_flux_v_proj.weight)
        nn.init.zeros_(self.garment_to_flux_v_proj.bias)
        
        # Get device from base model (handle ComfyUI ModelPatcher)
        device = self._get_model_device(base_model)
        self.garment_to_flux_k_proj = self.garment_to_flux_k_proj.to(device)
        self.garment_to_flux_v_proj = self.garment_to_flux_v_proj.to(device)
        
        # Patch the model's forward method
        self._patch_model()
    
    def _get_model_device(self, model):
        """Get device from model, handling ComfyUI ModelPatcher"""
        # Check if this is a ComfyUI ModelPatcher
        if hasattr(model, 'model') and hasattr(model.model, 'device'):
            return model.model.device
        elif hasattr(model, 'device'):
            return model.device
        elif hasattr(model, 'load_device'):
            return model.load_device
        else:
            # Try to get device from first parameter
            try:
                if hasattr(model, 'model'):
                    # It's a ModelPatcher, get the actual model
                    actual_model = model.model
                    if hasattr(actual_model, 'parameters'):
                        return next(actual_model.parameters()).device
                elif hasattr(model, 'parameters'):
                    return next(model.parameters()).device
            except StopIteration:
                pass
            # Default to CPU if we can't determine
            return torch.device('cpu')
    
    def _patch_model(self):
        """Patch the model to intercept forward calls"""
        # Check if this is a ComfyUI ModelPatcher
        if hasattr(self.base_model, 'model'):
            # It's a ModelPatcher, we need to patch the actual model
            self.actual_model = self.base_model.model
            self.is_model_patcher = True
        else:
            # It's a regular model
            self.actual_model = self.base_model
            self.is_model_patcher = False
        
        # Store original forward
        self.original_forward = self.actual_model.forward
        
        # Replace with our forward
        self.actual_model.forward = self._wrapped_forward
    
    def _wrapped_forward(self, *args, **kwargs):
        """Wrapped forward that handles read/write logic"""
        
        if self.current_mode == "write":
            # Store the model output for later injection
            output = self.original_forward(*args, **kwargs)
            self.garment_storage["write_output"] = output.clone() if output is not None else None
            self.features_written = True
            return output
            
        elif self.current_mode == "neg_write":
            # Store negative conditioning output
            output = self.original_forward(*args, **kwargs)
            self.garment_storage["neg_write_output"] = output.clone() if output is not None else None
            return output
            
        elif self.current_mode == "read" and self.features_written:
            # Get the base output
            output = self.original_forward(*args, **kwargs)
            
            # Inject stored garment features
            if "write_output" in self.garment_storage and self.garment_storage["write_output"] is not None:
                stored_output = self.garment_storage["write_output"]
                if output.shape == stored_output.shape:
                    # Blend the outputs - this is where garment features get injected
                    injection_strength = self.garment_features.get("injection_strength", 0.5)
                    output = output + injection_strength * (stored_output - output)
            
            return output
            
        elif self.current_mode == "neg_read" and self.features_written:
            # Similar for negative conditioning
            output = self.original_forward(*args, **kwargs)
            
            if "neg_write_output" in self.garment_storage and self.garment_storage["neg_write_output"] is not None:
                stored_output = self.garment_storage["neg_write_output"]
                if output.shape == stored_output.shape:
                    injection_strength = self.garment_features.get("injection_strength", 0.5)
                    output = output + injection_strength * (stored_output - output)
            
            return output
        
        else:
            # Normal forward pass
            return self.original_forward(*args, **kwargs)
    
    def _forward_write_mode(self, *args, **kwargs):
        """Forward pass in write mode - stores garment features"""
        # In write mode, we need to:
        # 1. Process the input through the model
        # 2. Store the attention features for later use
        
        # Get the actual forward result
        with torch.no_grad():
            # Store current inputs for potential reuse
            self.stored_args = args
            self.stored_kwargs = kwargs
            
            # Process through the model to generate attention features
            output = self.original_forward(*args, **kwargs)
            
            # In DreamFit, the write mode stores attention K,V values
            # Since we can't easily hook into attention layers here,
            # we'll store the output features for later injection
            if self.current_mode == "write":
                self.garment_storage["write_features"] = output
                self.garment_storage["write_args"] = args
                self.garment_storage["write_kwargs"] = kwargs
            elif self.current_mode == "neg_write":
                self.garment_storage["neg_write_features"] = output
                self.garment_storage["neg_write_args"] = args
                self.garment_storage["neg_write_kwargs"] = kwargs
            
            # Mark features as written
            self.features_written = True
            
            return output
    
    def _forward_read_mode(self, *args, **kwargs):
        """Forward pass in read mode - uses stored garment features"""
        # Get the base output
        output = self.original_forward(*args, **kwargs)
        
        # Apply garment influence based on stored features
        if self.features_written and output is not None:
            # Determine which features to use based on mode
            if self.current_mode == "read" and "write_features" in self.garment_storage:
                stored_features = self.garment_storage["write_features"]
                # Add garment influence (simplified version of attention injection)
                # In real DreamFit, this happens inside attention layers
                if stored_features is not None and output.shape == stored_features.shape:
                    # Blend features with adjustable strength
                    injection_strength = self.garment_features.get("injection_strength", 0.5)
                    output = output + injection_strength * stored_features
                    
            elif self.current_mode == "neg_read" and "neg_write_features" in self.garment_storage:
                stored_features = self.garment_storage["neg_write_features"]
                if stored_features is not None and output.shape == stored_features.shape:
                    injection_strength = self.garment_features.get("injection_strength", 0.5)
                    output = output + injection_strength * stored_features
        
        return output
    
    def _prepare_garment_inputs(self, *args, **kwargs):
        """Prepare inputs for garment feature storage with projection"""
        # Extract garment features
        garment_token = self.garment_features.get("garment_token")
        if garment_token is None:
            return {}
        
        # Ensure proper shape [B, L, D] where D=1024
        if garment_token.dim() == 2:
            garment_token = garment_token.unsqueeze(0)
        
        # Project garment features to Flux dimension
        device = garment_token.device
        self.garment_to_flux_k_proj = self.garment_to_flux_k_proj.to(device)
        self.garment_to_flux_v_proj = self.garment_to_flux_v_proj.to(device)
        
        garment_k = self.garment_to_flux_k_proj(garment_token)  # [B, L, 3072]
        garment_v = self.garment_to_flux_v_proj(garment_token)  # [B, L, 3072]
        
        prepared = {
            "garment_token": garment_token,
            "garment_k": garment_k,
            "garment_v": garment_v,
            "patch_features": self.garment_features.get("patch_features"),
            "pooled_features": self.garment_features.get("pooled_features"),
        }
        return prepared
    
    def set_mode(self, mode: str):
        """Set the current mode (write/read/normal)"""
        self.current_mode = mode
    
    def reset(self):
        """Reset the wrapper state"""
        self.features_written = False
        self.garment_storage.clear()
        self.current_mode = "normal"
    
    def forward(self, *args, **kwargs):
        """Forward through the base model"""
        # For ComfyUI compatibility, we need to handle ModelPatcher
        if self.is_model_patcher:
            # Let the ModelPatcher handle the forward call
            return self.base_model(*args, **kwargs)
        else:
            # Direct model forward
            return self.actual_model(*args, **kwargs)
    
    def __getattr__(self, name):
        """Delegate attribute access to base model"""
        if name in ['base_model', 'garment_features', 'garment_storage', 
                    'features_written', 'current_mode', 'original_forward',
                    'stored_args', 'stored_kwargs', 'garment_dim', 'hidden_dim',
                    'garment_to_flux_k_proj', 'garment_to_flux_v_proj',
                    'actual_model', 'is_model_patcher']:
            return super().__getattr__(name)
        return getattr(self.base_model, name)


class DreamFitAttentionProcessor:
    """
    Custom attention processor that handles read/write mechanism
    This would be attached to specific attention layers in the model
    """
    
    def __init__(self):
        self.stored_q = None
        self.stored_k = None
        self.stored_v = None
        self.mode = "normal"
    
    def __call__(self, attn, hidden_states, encoder_hidden_states=None, **kwargs):
        """Process attention with read/write logic"""
        if self.mode == "write":
            # Store Q, K, V from garment features
            # This is simplified - actual implementation would compute from garment features
            self.stored_q = hidden_states  # Placeholder
            self.stored_k = hidden_states  # Placeholder
            self.stored_v = hidden_states  # Placeholder
            return hidden_states  # Return unchanged in write mode
            
        elif self.mode == "read" and self.stored_q is not None:
            # Concatenate stored garment features with current features
            # This is where the magic happens - injecting garment information
            # Actual implementation would properly compute attention with concatenated features
            return hidden_states  # Placeholder
            
        else:
            # Normal attention
            return hidden_states  # Placeholder - would call original attention
    
    def set_mode(self, mode: str):
        """Set processor mode"""
        self.mode = mode
    
    def reset(self):
        """Reset stored values"""
        self.stored_q = None
        self.stored_k = None
        self.stored_v = None
        self.mode = "normal"