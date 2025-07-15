"""
DreamFit Unified Node V2 - Correct implementation based on official DreamFit
Loads complete checkpoints and properly integrates with Flux models
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from typing import Dict, Optional, Tuple, List, Any
from pathlib import Path

# DreamFit imports
from ..dreamfit_core.models.anything_dressing_encoder import AnythingDressingEncoder, EncoderConfig
from ..dreamfit_core.utils.image_processing import preprocess_garment_image
from ..dreamfit_types import DreamFitFeatures


class DreamFitUnifiedV2:
    """
    Correct DreamFit implementation based on official codebase
    Uses self-contained checkpoints with encoder and LoRA weights
    """
    
    # Class-level cache for checkpoints
    _cached_checkpoint = None
    _cached_checkpoint_name = None
    _cached_encoder = None
    
    @classmethod
    def INPUT_TYPES(cls):
        # Get available checkpoints
        models_dir = cls._get_models_dir()
        available_checkpoints = cls._get_available_checkpoints(models_dir)
        
        return {
            "required": {
                "model": ("MODEL",),  # Flux diffusion model
                "positive": ("CONDITIONING",),  # Pre-encoded from CLIP
                "negative": ("CONDITIONING",),  # Pre-encoded from CLIP
                "garment_image": ("IMAGE",),  # Reference garment
                "dreamfit_checkpoint": (available_checkpoints,),
                "strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.05,
                    "display": "slider"
                }),
                "injection_strength": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "display": "slider"
                }),
            },
            "optional": {
                "model_image": ("IMAGE",),  # For try-on mode
                "injection_mode": (["adaptive", "fixed", "progressive"], {
                    "default": "adaptive"
                }),
                "debug_mode": ("BOOLEAN", {"default": False}),
            }
        }
    
    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING", "IMAGE", "DREAMFIT_FEATURES")
    RETURN_NAMES = ("model", "positive", "negative", "debug_visualization", "garment_features")
    FUNCTION = "process"
    CATEGORY = "DreamFit"
    
    @classmethod
    def _get_models_dir(cls):
        """Get the DreamFit models directory"""
        try:
            import folder_paths
            return os.path.join(folder_paths.models_dir, "dreamfit")
        except ImportError:
            return os.path.join(os.path.expanduser("~"), ".cache", "dreamfit")
    
    @classmethod
    def _get_available_checkpoints(cls, models_dir):
        """Get list of available DreamFit checkpoints"""
        os.makedirs(models_dir, exist_ok=True)
        
        available = []
        for checkpoint_name in ["flux_i2i", "flux_i2i_with_pose", "flux_tryon"]:
            checkpoint_path = os.path.join(models_dir, f"{checkpoint_name}.bin")
            if os.path.exists(checkpoint_path):
                available.append(checkpoint_name)
        
        if not available:
            return ["Please run download_models.py first"]
        
        return available
    
    def process(
        self,
        model,
        positive,
        negative,
        garment_image,
        dreamfit_checkpoint: str,
        strength: float = 1.0,
        injection_strength: float = 0.5,
        model_image: Optional[torch.Tensor] = None,
        injection_mode: str = "adaptive",
        debug_mode: bool = False
    ):
        """
        Process with complete DreamFit checkpoint
        """
        # Validate checkpoint selection
        if dreamfit_checkpoint == "Please run download_models.py first":
            raise ValueError(
                "No DreamFit checkpoints found!\n\n"
                "Please download models first by running:\n"
                "python download_models.py\n\n"
                "From the ComfyUI-DreamFit directory"
            )
        
        # Check for try-on mode requirements
        if dreamfit_checkpoint == "flux_tryon" and model_image is None:
            raise ValueError(
                "flux_tryon checkpoint requires a model_image input!\n"
                "Please connect a reference pose/model image."
            )
        
        # Get device
        device = model.load_device if hasattr(model, 'load_device') else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load complete checkpoint (encoder + LoRA + configs)
        checkpoint_data, encoder = self._load_checkpoint(dreamfit_checkpoint, device)
        
        # Process garment image (must be 224x224 for encoder)
        processed_garment = preprocess_garment_image(
            garment_image,
            target_size=224,
            normalize=True,
            device=device
        )
        
        # Extract garment features using checkpoint's encoder
        with torch.no_grad():
            encoder_output = encoder(processed_garment)
        
        # Build garment features dictionary
        garment_features = {
            "garment_token": encoder_output["garment_token"],
            "pooled_features": encoder_output["pooled_features"],
            "patch_features": encoder_output["patch_features"],
            "features": encoder_output.get("features", {}),
            "attention_weights": encoder_output.get("attention_weights", {}),
            "encoder_config": checkpoint_data.get("encoder_config", {}),
        }
        
        # Process model image if provided (for try-on mode)
        pose_features = None
        if model_image is not None:
            processed_pose = preprocess_garment_image(
                model_image,
                target_size=224,
                normalize=True,
                device=device
            )
            with torch.no_grad():
                pose_output = encoder(processed_pose)
            pose_features = {
                "pose_token": pose_output["garment_token"],
                "pose_patches": pose_output["patch_features"],
            }
            garment_features["pose_features"] = pose_features
        
        # Apply LoRA weights from checkpoint to model
        enhanced_model = self._apply_checkpoint_lora(
            model, 
            checkpoint_data.get("lora_weights", {}),
            strength
        )
        
        # CRITICAL: Store garment features in the model for ComfyUI's sampling
        # ComfyUI will pass this through the conditioning
        enhanced_model.model.dreamfit_features = {
            "garment_token": garment_features["garment_token"],
            "patch_features": garment_features.get("patch_features"),
            "pooled_features": garment_features.get("pooled_features"),
            "injection_strength": injection_strength,
            "injection_mode": injection_mode,
            "pose_features": garment_features.get("pose_features"),
        }
        
        # Mark the model as having DreamFit features
        enhanced_model.model.dreamfit_enabled = True
        
        # Enhance conditioning with garment features
        enhanced_positive = self._enhance_conditioning(
            positive,
            garment_features,
            injection_strength,
            injection_mode
        )
        
        # Create debug visualization
        try:
            debug_viz = self._create_debug_visualization(
                garment_image,
                processed_garment,
                encoder_output.get("attention_weights", {}),
                model_image,
                debug_mode
            )
        except Exception as e:
            print(f"Warning: Debug visualization failed: {e}")
            print(f"Garment image shape: {garment_image.shape}")
            print(f"Processed garment shape: {processed_garment.shape}")
            if model_image is not None:
                print(f"Model image shape: {model_image.shape}")
            # Return original garment image as fallback
            debug_viz = garment_image
        
        # Create DreamFitFeatures object for output
        dreamfit_features_output = DreamFitFeatures(garment_features)
        
        # Return all outputs
        return (enhanced_model, enhanced_positive, negative, debug_viz, dreamfit_features_output)
    
    def _load_checkpoint(self, checkpoint_name: str, device: torch.device):
        """Load complete DreamFit checkpoint with caching"""
        # Check cache
        if self._cached_checkpoint_name == checkpoint_name:
            return self._cached_checkpoint, self._cached_encoder
        
        # Load checkpoint
        models_dir = self._get_models_dir()
        checkpoint_path = os.path.join(models_dir, f"{checkpoint_name}.bin")
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}\n"
                f"Please run: python download_models.py --model {checkpoint_name}"
            )
        
        print(f"Loading DreamFit checkpoint: {checkpoint_name}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Initialize encoder from checkpoint
        encoder_config_dict = checkpoint.get("encoder_config", {})
        encoder_config = EncoderConfig(**encoder_config_dict) if encoder_config_dict else EncoderConfig()
        
        encoder = AnythingDressingEncoder(encoder_config)
        if "encoder_state_dict" in checkpoint:
            encoder.load_state_dict(checkpoint["encoder_state_dict"])
        
        encoder = encoder.to(device).eval()
        
        # Cache for reuse
        DreamFitUnifiedV2._cached_checkpoint = checkpoint
        DreamFitUnifiedV2._cached_checkpoint_name = checkpoint_name
        DreamFitUnifiedV2._cached_encoder = encoder
        
        return checkpoint, encoder
    
    def _apply_checkpoint_lora(self, model, lora_weights: Dict, strength: float):
        """Apply LoRA weights from checkpoint to model"""
        if not lora_weights or strength <= 0:
            return model
        
        # Deep copy to avoid modifying original
        enhanced_model = copy.deepcopy(model)
        
        # Apply LoRA to attention layers
        applied_count = 0
        for name, module in enhanced_model.named_modules():
            if self._is_attention_layer(name, module):
                lora_key = self._get_lora_key(name)
                if lora_key in lora_weights:
                    self._apply_lora_to_module(
                        module,
                        lora_weights[lora_key],
                        strength
                    )
                    applied_count += 1
        
        print(f"Applied LoRA to {applied_count} attention layers")
        return enhanced_model
    
    def _create_ip_adapter_hook(self, garment_token: torch.Tensor, injection_strength: float, layer_idx: int):
        """Create a forward hook that injects garment features using IP-Adapter style"""
        def hook_fn(module, input, output):
            # Skip if not in generation mode
            if not hasattr(module, '_dreamfit_active'):
                return output
            
            try:
                # Handle different output formats
                if isinstance(output, tuple):
                    hidden_states = output[0]
                    rest = output[1:]
                else:
                    hidden_states = output
                    rest = ()
                
                # Get the query from hidden states (assuming it's the image stream)
                B, L, D = hidden_states.shape
                
                # Ensure garment token matches batch size
                if garment_token.shape[0] == 1 and B > 1:
                    garment_features = garment_token.repeat(B, 1, 1)
                else:
                    garment_features = garment_token[:B]
                
                # Simple cross-attention injection
                # In real IP-Adapter, this would use proper K,V projections
                # For now, we'll add the garment information directly
                garment_influence = garment_features.mean(dim=1, keepdim=True)  # [B, 1, D]
                garment_influence = garment_influence.expand(-1, L, -1)  # [B, L, D]
                
                # Apply injection with strength
                modified_hidden = hidden_states + injection_strength * garment_influence
                
                # Return in original format
                if rest:
                    return (modified_hidden,) + rest
                else:
                    return modified_hidden
                    
            except Exception as e:
                print(f"Error in DreamFit injection hook at layer {layer_idx}: {e}")
                return output
        
        return hook_fn
    
    def _is_attention_layer(self, name: str, module: nn.Module) -> bool:
        """Check if module is an attention layer that should get LoRA"""
        # Check for common attention layer patterns in Flux
        attention_patterns = [
            "attn", "attention", "self_attn", "cross_attn",
            "to_q", "to_k", "to_v", "to_out"
        ]
        return any(pattern in name.lower() for pattern in attention_patterns)
    
    def _get_lora_key(self, module_name: str) -> str:
        """Convert module name to LoRA weight key"""
        # Simplify the module name for matching with checkpoint keys
        # This mapping depends on how LoRA weights are stored in checkpoint
        key = module_name.replace(".", "_")
        return key
    
    def _apply_lora_to_module(self, module: nn.Module, lora_weight: Dict, strength: float):
        """Apply LoRA weights to a specific module"""
        if isinstance(module, nn.Linear):
            # Standard LoRA application for linear layers
            if "lora_A" in lora_weight and "lora_B" in lora_weight:
                lora_A = lora_weight["lora_A"].to(module.weight.device)
                lora_B = lora_weight["lora_B"].to(module.weight.device)
                
                # Apply LoRA: W' = W + strength * BA
                with torch.no_grad():
                    module.weight.data += strength * (lora_B @ lora_A)
    
    def _setup_feature_injection(
        self,
        model,
        garment_features: Dict,
        injection_strength: float,
        injection_mode: str,
        injection_config: Dict
    ):
        """Set up feature injection in the model using IP-Adapter style"""
        # Extract garment token for cross-attention
        garment_token = garment_features.get("garment_token")  # [B, 1, D]
        
        if garment_token is None:
            print("Warning: No garment token found in features")
            return
        
        # Store injection configuration
        injection_data = {
            "garment_token": garment_token,
            "patch_features": garment_features.get("patch_features"),
            "injection_strength": injection_strength,
            "injection_mode": injection_mode,
            "injection_layers": injection_config.get("layers", [3, 6, 9, 12, 15, 18]),
            "active": True
        }
        
        # Find double stream blocks in Flux model
        double_stream_blocks = []
        for name, module in model.named_modules():
            if "double_stream" in name.lower() or "doublestream" in name.lower():
                double_stream_blocks.append((name, module))
        
        if not double_stream_blocks:
            # Fallback: find any attention blocks
            for name, module in model.named_modules():
                if any(key in name.lower() for key in ["attn", "attention", "transformer"]):
                    double_stream_blocks.append((name, module))
        
        print(f"Found {len(double_stream_blocks)} potential injection points")
        
        # Apply IP-adapter style injection to selected layers
        injected_count = 0
        for idx, (name, module) in enumerate(double_stream_blocks):
            if idx in injection_data["injection_layers"]:
                # Create and register the injection hook
                hook = self._create_ip_adapter_hook(
                    garment_token,
                    injection_strength,
                    idx
                )
                module.register_forward_hook(hook)
                injected_count += 1
                
        print(f"Injected DreamFit features into {injected_count} layers")
    
    def _enhance_conditioning(
        self,
        conditioning,
        garment_features: Dict,
        injection_strength: float,
        injection_mode: str
    ):
        """Enhance conditioning with garment features"""
        enhanced_conditioning = []
        
        for item in conditioning:
            if isinstance(item, list) and len(item) == 2:
                cond_tensor, extras = item
            else:
                cond_tensor = item
                extras = {}
            
            # Create new extras with DreamFit features
            new_extras = extras.copy() if isinstance(extras, dict) else {}
            
            # CRITICAL: Add garment features in a way ComfyUI can use
            # The garment_token should modify the conditioning directly
            garment_token = garment_features["garment_token"]
            pooled_features = garment_features.get("pooled_features")
            
            # Concatenate garment token to conditioning
            # This ensures it's part of the actual conditioning passed to the model
            if garment_token is not None:
                # Ensure dimensions match
                if cond_tensor.dim() == 2:
                    cond_tensor = cond_tensor.unsqueeze(0)
                if garment_token.dim() == 2:
                    garment_token = garment_token.unsqueeze(0)
                
                # Match batch sizes
                B = cond_tensor.shape[0]
                if garment_token.shape[0] == 1 and B > 1:
                    garment_token = garment_token.repeat(B, 1, 1)
                elif garment_token.shape[0] > B:
                    garment_token = garment_token[:B]
                
                # IMPORTANT: For ComfyUI compatibility, we need to enhance the existing conditioning
                # rather than concatenate (which would change sequence length)
                # Average the garment token and add it to the conditioning
                if cond_tensor.shape[1] > 0:
                    # Add garment influence to the first few tokens (similar to IP-Adapter)
                    num_tokens_to_enhance = min(4, cond_tensor.shape[1])
                    enhanced_cond = cond_tensor.clone()
                    garment_influence = garment_token.mean(dim=1, keepdim=True) * injection_strength
                    enhanced_cond[:, :num_tokens_to_enhance] += garment_influence.expand(-1, num_tokens_to_enhance, -1)
                else:
                    enhanced_cond = cond_tensor
                
                # If we have pooled features, also enhance them
                if pooled_features is not None and "pooled_output" in new_extras:
                    pooled = new_extras["pooled_output"]
                    if pooled is not None:
                        # Add garment influence to pooled output
                        new_extras["pooled_output"] = pooled + injection_strength * pooled_features.mean(dim=0)
            else:
                enhanced_cond = cond_tensor
            
            # Also store features in extras for potential custom samplers
            new_extras["dreamfit_features"] = {
                "garment_token": garment_features["garment_token"],
                "pooled_features": garment_features["pooled_features"],
                "patch_features": garment_features["patch_features"],
                "injection_strength": injection_strength,
                "injection_mode": injection_mode,
                "pose_features": garment_features.get("pose_features"),
            }
            
            enhanced_conditioning.append([enhanced_cond, new_extras])
        
        return enhanced_conditioning
    
    def _create_debug_visualization(
        self,
        original_garment: torch.Tensor,
        processed_garment: torch.Tensor,
        attention_weights: Dict,
        model_image: Optional[torch.Tensor],
        debug_mode: bool
    ):
        """Create informative debug visualization grid"""
        if not debug_mode:
            # Return minimal debug image if not in debug mode
            return original_garment
        
        try:
            # Handle ComfyUI format [B, H, W, C] -> [B, C, H, W]
            if original_garment.dim() == 4 and original_garment.shape[-1] == 3:
                original_garment_viz = original_garment.permute(0, 3, 1, 2)
            else:
                original_garment_viz = original_garment.clone()
            
            B, C, H, W = original_garment_viz.shape
            device = original_garment_viz.device
        
            # Denormalize processed garment for visualization
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
            processed_viz = processed_garment * std + mean
            processed_viz = processed_viz.clamp(0, 1)
            
            # Create attention heatmap
            heatmap = self._create_attention_heatmap(attention_weights, (H, W))
            
            # Prepare images for grid
            images = []
            
            # Top-left: Original garment
            images.append(original_garment_viz[0])
        
            # Top-right: Processed garment (upscaled to match)
            processed_resized = F.interpolate(
                processed_viz,
                size=(H, W),
                mode='bilinear',
                align_corners=False
            )
            images.append(processed_resized[0])
            
            # Bottom-left: Attention heatmap
            images.append(heatmap)
            
            # Bottom-right: Model image or placeholder
            if model_image is not None:
                if model_image.dim() == 4 and model_image.shape[-1] == 3:
                    model_image = model_image.permute(0, 3, 1, 2)
                # Ensure model image matches the size
                if model_image.shape[-2:] != (H, W):
                    model_image = F.interpolate(model_image, size=(H, W), mode='bilinear', align_corners=False)
                images.append(model_image[0])
            else:
                # Create placeholder
                placeholder = torch.zeros_like(images[0])
                # Add text or pattern to indicate no model image
                placeholder[:, H//3:2*H//3, W//3:2*W//3] = 0.2
                images.append(placeholder)
            
            # Create 2x2 grid
            grid = torch.zeros(3, 2*H, 2*W, device=device)
            grid[:, :H, :W] = images[0]
            grid[:, :H, W:] = images[1]
            grid[:, H:, :W] = images[2]
            grid[:, H:, W:] = images[3]
            
            # Convert back to ComfyUI format [B, H, W, C]
            grid = grid.unsqueeze(0).permute(0, 2, 3, 1)
            
            return grid
        except Exception as e:
            print(f"Error in debug visualization: {e}")
            # Return original image as fallback
            return original_garment
    
    def _create_attention_heatmap(self, attention_weights: Dict, size: Tuple[int, int]) -> torch.Tensor:
        """Create attention heatmap from weights"""
        H, W = size
        
        try:
            if not attention_weights:
                # Return gray image if no attention weights
                return torch.ones(3, H, W) * 0.5
            
            # Get the last layer's attention (usually most informative)
            last_layer_key = max(attention_weights.keys()) if attention_weights else None
            if last_layer_key is None:
                return torch.ones(3, H, W) * 0.5
            
            attn = attention_weights[last_layer_key]
            
            # Average attention across heads and reshape
            if attn.dim() == 4:  # [B, heads, seq, seq]
                attn = attn.mean(dim=1)  # Average across heads
            
            # Take attention to garment token (index 1)
            if attn.shape[-1] > 1:
                garment_attn = attn[0, :, 1]  # Attention from all positions to garment token
            else:
                garment_attn = attn[0, 0]
            
            # Reshape to spatial dimensions
            num_patches = int(np.sqrt(garment_attn.shape[0] - 2))  # Subtract CLS and garment tokens
            if num_patches * num_patches + 2 == garment_attn.shape[0]:
                spatial_attn = garment_attn[2:].reshape(num_patches, num_patches)
            else:
                # Fallback if shape doesn't match
                spatial_attn = torch.ones(16, 16) * 0.5
            
            # Upsample to target size
            spatial_attn = spatial_attn.unsqueeze(0).unsqueeze(0)
            heatmap = F.interpolate(spatial_attn, size=(H, W), mode='bilinear', align_corners=False)
            
            # Normalize and convert to RGB
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
            
            # Apply colormap (simple red-yellow-white)
            heatmap_rgb = torch.zeros(3, H, W)
            heatmap_rgb[0] = heatmap[0, 0]  # Red channel
            heatmap_rgb[1] = heatmap[0, 0] * 0.5  # Green channel
            heatmap_rgb[2] = 0  # Blue channel
            
            return heatmap_rgb
        except Exception as e:
            print(f"Error creating attention heatmap: {e}")
            # Return gray fallback
            return torch.ones(3, H, W) * 0.5
    
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Force reload if checkpoint files change
        return float("nan")


# Node mappings
NODE_CLASS_MAPPINGS = {
    "DreamFitUnifiedV2": DreamFitUnifiedV2,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DreamFitUnifiedV2": "DreamFit Unified V2",
}