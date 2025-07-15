"""
DreamFit Unified Node - Complete integration for Flux diffusion models
Handles garment feature extraction and conditioning enhancement in one node
"""

import os
import torch
import torch.nn as nn
import copy
from typing import Dict, Optional, Tuple, List, Any
from pathlib import Path

# DreamFit imports
from ..dreamfit_core.models.anything_dressing_encoder import AnythingDressingEncoder, EncoderConfig
from ..dreamfit_core.utils.image_processing import preprocess_garment_image
from ..dreamfit_core.models.lora_adapter import DreamFitLoRAAdapter


class DreamFitUnified:
    """
    Unified DreamFit node that properly integrates with Flux diffusion models
    """
    
    # Class-level cache for encoder to avoid reloading
    _cached_encoder = None
    _cached_model_name = None
    
    @classmethod
    def INPUT_TYPES(cls):
        # Get available models
        models_dir = cls._get_models_dir()
        available_models = cls._get_available_models(models_dir)
        
        return {
            "required": {
                "model": ("MODEL",),  # Flux diffusion model
                "positive": ("CONDITIONING",),  # Already encoded by CLIP
                "negative": ("CONDITIONING",),  # Already encoded by CLIP
                "garment_image": ("IMAGE",),
                "dreamfit_model": (available_models,),
                "strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.05,
                    "display": "slider"
                }),
            },
            "optional": {
                "model_image": ("IMAGE",),  # For try-on mode
                "injection_strength": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "display": "slider"
                }),
                "injection_mode": (["adaptive", "fixed", "progressive"], {
                    "default": "adaptive"
                }),
            }
        }
    
    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING", "IMAGE")
    RETURN_NAMES = ("model", "positive", "negative", "debug_garment")
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
    def _get_available_models(cls, models_dir):
        """Get list of available DreamFit models"""
        os.makedirs(models_dir, exist_ok=True)
        
        available_models = []
        for model_name in ["flux_i2i", "flux_i2i_with_pose", "flux_tryon"]:
            model_path = os.path.join(models_dir, f"{model_name}.bin")
            if os.path.exists(model_path):
                available_models.append(model_name)
        
        if not available_models:
            return ["Please run download_models.py first"]
        
        return available_models
    
    def process(
        self,
        model,
        positive,
        negative,
        garment_image,
        dreamfit_model: str,
        strength: float = 1.0,
        model_image: Optional[torch.Tensor] = None,
        injection_strength: float = 0.5,
        injection_mode: str = "adaptive"
    ):
        """
        Process garment and enhance model/conditioning
        """
        # Validate model selection
        if dreamfit_model == "Please run download_models.py first":
            raise ValueError(
                "No DreamFit models found!\n\n"
                "Please download models first by running:\n"
                "python download_models.py\n\n"
                "From the ComfyUI-DreamFit directory"
            )
        
        # Check for try-on mode requirements
        if dreamfit_model == "flux_tryon" and model_image is None:
            raise ValueError(
                "flux_tryon model requires a model_image input!\n"
                "Please connect a reference pose/model image."
            )
        
        # Get device
        device = model.load_device if hasattr(model, 'load_device') else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load DreamFit components
        encoder, lora_weights, config = self._load_dreamfit_model(dreamfit_model, device)
        
        # Process garment image (resize to 224x224 is required!)
        processed_garment = preprocess_garment_image(
            garment_image,
            target_size=224,  # REQUIRED by Anything-Dressing Encoder
            normalize=True,
            device=device
        )
        
        # Extract garment features
        with torch.no_grad():
            encoder_output = encoder(processed_garment)
        
        garment_features = {
            "garment_token": encoder_output["garment_token"],
            "pooled_features": encoder_output["pooled_features"], 
            "patch_features": encoder_output["patch_features"],
            "features": encoder_output.get("features", {}),
        }
        
        # Process model image if provided
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
                "pose_token": pose_output["garment_token"],  # Reuse same encoder
                "pose_patches": pose_output["patch_features"],
            }
        
        # Enhance model with LoRA if available
        enhanced_model = self._enhance_model(model, lora_weights, strength)
        
        # Store DreamFit components in model for samplers
        enhanced_model.dreamfit_components = {
            "garment_features": garment_features,
            "pose_features": pose_features,
            "injection_config": {
                "strength": injection_strength,
                "mode": injection_mode,
                "layers": config.get("injection_layers", [3, 6, 9, 12, 15, 18])
            },
            "model_type": dreamfit_model,
            "encoder": encoder,
        }
        
        # Enhance conditioning with garment features
        enhanced_positive = self._enhance_conditioning(
            positive,
            garment_features,
            pose_features,
            injection_strength,
            injection_mode,
            config
        )
        
        # Create debug output (denormalized for visualization)
        debug_garment = self._create_debug_output(processed_garment)
        
        return (enhanced_model, enhanced_positive, negative, debug_garment)
    
    def _load_dreamfit_model(self, model_name: str, device: torch.device):
        """Load DreamFit checkpoint with caching"""
        # Check cache
        if self._cached_model_name == model_name and self._cached_encoder is not None:
            encoder = self._cached_encoder
        else:
            # Load checkpoint
            models_dir = self._get_models_dir()
            model_path = os.path.join(models_dir, f"{model_name}.bin")
            
            if not os.path.exists(model_path):
                raise FileNotFoundError(
                    f"Model file not found: {model_path}\n"
                    f"Please run: python download_models.py --model {model_name}"
                )
            
            print(f"Loading DreamFit model: {model_name}")
            checkpoint = torch.load(model_path, map_location=device)
            
            # Initialize encoder
            encoder_config_dict = checkpoint.get("encoder_config", {})
            encoder_config = EncoderConfig(**encoder_config_dict) if encoder_config_dict else EncoderConfig()
            
            encoder = AnythingDressingEncoder(encoder_config)
            if "encoder_state_dict" in checkpoint:
                encoder.load_state_dict(checkpoint["encoder_state_dict"])
            
            encoder = encoder.to(device).eval()
            
            # Cache for next use
            DreamFitUnified._cached_encoder = encoder
            DreamFitUnified._cached_model_name = model_name
        
        # Extract other components
        lora_weights = checkpoint.get("lora_weights", {}) or checkpoint.get("adapter_state", {}).get("lora_weights", {})
        config = checkpoint.get("config", checkpoint.get("model_config", checkpoint.get("attention_injection", {})))
        
        return encoder, lora_weights, config
    
    def _enhance_model(self, model, lora_weights: Dict, strength: float):
        """Apply LoRA enhancement to model"""
        if not lora_weights or strength <= 0:
            return model
        
        # Deep copy to avoid modifying original
        enhanced_model = copy.deepcopy(model)
        
        # Apply LoRA (simplified - real implementation would be more sophisticated)
        # This is a placeholder as full LoRA implementation requires knowing
        # the exact model architecture
        try:
            # Store LoRA info for potential use by samplers
            enhanced_model.dreamfit_lora = {
                "weights": lora_weights,
                "strength": strength
            }
        except Exception as e:
            print(f"Warning: Could not apply LoRA: {e}")
        
        return enhanced_model
    
    def _enhance_conditioning(
        self,
        conditioning,
        garment_features: Dict,
        pose_features: Optional[Dict],
        injection_strength: float,
        injection_mode: str,
        config: Dict
    ):
        """Enhance conditioning with DreamFit features"""
        enhanced_conditioning = []
        
        # ComfyUI conditioning format: [(tensor, extras_dict)]
        for item in conditioning:
            if isinstance(item, list) and len(item) == 2:
                cond_tensor, extras = item
            else:
                # Handle other formats
                cond_tensor = item
                extras = {}
            
            # Create new extras with DreamFit features
            new_extras = extras.copy() if isinstance(extras, dict) else {}
            
            # Add DreamFit features
            new_extras["dreamfit_features"] = {
                "garment_token": garment_features["garment_token"],
                "pooled_features": garment_features["pooled_features"],
                "patch_features": garment_features["patch_features"],
                "features": garment_features.get("features", {}),
                "injection_config": {
                    "strength": injection_strength,
                    "mode": injection_mode,
                    "layers": config.get("injection_layers", [3, 6, 9, 12, 15, 18])
                }
            }
            
            # Add pose features if available
            if pose_features is not None:
                new_extras["dreamfit_features"]["pose_features"] = pose_features
            
            enhanced_conditioning.append([cond_tensor, new_extras])
        
        return enhanced_conditioning
    
    def _create_debug_output(self, processed_garment: torch.Tensor):
        """Create debug visualization of processed garment"""
        # Denormalize
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(processed_garment.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(processed_garment.device)
        
        debug_image = processed_garment * std + mean
        debug_image = debug_image.clamp(0, 1)
        
        # Convert to ComfyUI format [B, H, W, C]
        debug_image = debug_image.permute(0, 2, 3, 1)
        
        return debug_image
    
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Force reload if model files change
        return float("nan")


# Node mappings
NODE_CLASS_MAPPINGS = {
    "DreamFitUnified": DreamFitUnified,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DreamFitUnified": "DreamFit Unified",
}