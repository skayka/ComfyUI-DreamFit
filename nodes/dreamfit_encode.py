"""
DreamFit Encode Node
Encodes garment images and text prompts into conditioning for Flux generation
"""

import torch
from typing import Dict, Any, Optional, Tuple
import numpy as np

# DreamFit imports
from ..dreamfit_core.models import AnythingDressingEncoder
from ..dreamfit_core.models.anything_dressing_encoder import EncoderConfig
from ..dreamfit_core.utils import preprocess_garment_image, PromptEnhancer


class DreamFitEncode:
    """
    Encode garment images and text prompts for DreamFit generation
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dreamfit_model": ("DREAMFIT_MODEL",),
                "encoder": ("DREAMFIT_ENCODER",),
                "garment_image": ("IMAGE",),
                "text_prompt": ("STRING", {
                    "default": "A person wearing the garment",
                    "multiline": True
                }),
            },
            "optional": {
                "model_image": ("IMAGE",),  # Optional pose/model reference
                "garment_mask": ("MASK",),
                "enhance_prompt": ("BOOLEAN", {"default": True}),
                "use_lmm": ("BOOLEAN", {"default": False}),
                "injection_strength": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05
                }),
                "injection_mode": (["adaptive", "fixed", "progressive"], {
                    "default": "adaptive"
                }),
            }
        }
    
    RETURN_TYPES = ("DREAMFIT_CONDITIONING", "STRING", "IMAGE")
    RETURN_NAMES = ("conditioning", "enhanced_prompt", "processed_garment")
    FUNCTION = "encode"
    CATEGORY = "DreamFit"
    
    def encode(
        self,
        dreamfit_model: Dict,
        encoder: AnythingDressingEncoder,
        garment_image: torch.Tensor,
        text_prompt: str,
        model_image: Optional[torch.Tensor] = None,
        garment_mask: Optional[torch.Tensor] = None,
        enhance_prompt: bool = True,
        use_lmm: bool = False,
        injection_strength: float = 0.5,
        injection_mode: str = "adaptive"
    ) -> Tuple[Dict, str, torch.Tensor]:
        """
        Encode garment and prompt into conditioning
        
        Args:
            dreamfit_model: The loaded DreamFit model dict
            encoder: The Anything-Dressing Encoder
            garment_image: Input garment image [B, H, W, C]
            text_prompt: Text description
            model_image: Optional model/pose reference image
            garment_mask: Optional garment mask
            enhance_prompt: Whether to enhance the prompt
            use_lmm: Whether to use LMM enhancement (placeholder)
            injection_strength: Strength of feature injection
            injection_mode: Mode for injection (adaptive/fixed/progressive)
            
        Returns:
            Tuple of (conditioning_dict, enhanced_prompt, processed_garment)
        """
        device = encoder.device if hasattr(encoder, 'device') else torch.device('cpu')
        
        # Move encoder to device if needed
        if hasattr(encoder, 'to'):
            encoder = encoder.to(device)
        
        # Preprocess garment image
        # ComfyUI images are [B, H, W, C] in range [0, 1]
        processed_garment = preprocess_garment_image(
            garment_image,
            target_size=224,
            normalize=True,
            device=device
        )
        
        # Extract garment features using encoder
        with torch.no_grad():
            encoder_output = encoder(processed_garment)
        
        # Extract different feature representations
        garment_features = {
            "garment_token": encoder_output["garment_token"],
            "pooled_features": encoder_output["pooled_features"],
            "patch_features": encoder_output["patch_features"],
            "features": encoder_output["features"],
            "attention_weights": encoder_output.get("attention_weights", {})
        }
        
        # Enhance prompt if requested
        if enhance_prompt:
            prompt_enhancer = PromptEnhancer()
            
            # Extract garment info from prompt
            garment_info = prompt_enhancer.extract_garment_info(text_prompt)
            
            # Create garment description from features (placeholder)
            garment_desc = None
            if not garment_info["garment_types"]:
                garment_desc = "fashionable garment"
            
            enhanced_prompt = prompt_enhancer.enhance_prompt(
                text_prompt,
                garment_description=garment_desc,
                use_lmm=use_lmm
            )
        else:
            enhanced_prompt = text_prompt
        
        # Process model/pose image if provided
        if model_image is not None:
            from ..dreamfit_core.utils import preprocess_pose_image
            processed_pose = preprocess_pose_image(
                model_image,
                target_size=224,
                device=device
            )
        else:
            processed_pose = None
        
        # Create conditioning dictionary
        conditioning = {
            "type": "dreamfit",
            "garment_features": garment_features,
            "text_prompt": enhanced_prompt,
            "injection_config": {
                "injection_strength": injection_strength,
                "injection_mode": injection_mode,
                "injection_layers": dreamfit_model["config"].get("injection_layers", [3, 6, 9, 12, 15, 18])
            },
            "model_config": dreamfit_model["config"],
            "adapter_state": dreamfit_model.get("adapter_state", {}),
        }
        
        # Add pose conditioning if available
        if processed_pose is not None:
            conditioning["pose_features"] = {
                "pose_image": processed_pose,
                "use_pose": True
            }
        
        # Add mask if provided
        if garment_mask is not None:
            conditioning["garment_mask"] = garment_mask
        
        # Convert processed image back to ComfyUI format for preview
        preview_image = processed_garment.cpu()
        # Denormalize
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        preview_image = preview_image * std + mean
        preview_image = preview_image.clamp(0, 1)
        # Convert to [B, H, W, C]
        preview_image = preview_image.permute(0, 2, 3, 1)
        
        return (conditioning, enhanced_prompt, preview_image)


class DreamFitConditioningInfo:
    """
    Display information about DreamFit conditioning
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("DREAMFIT_CONDITIONING",),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("info",)
    FUNCTION = "get_info"
    CATEGORY = "DreamFit/Utils"
    
    def get_info(self, conditioning: Dict) -> Tuple[str]:
        """Get information about conditioning"""
        info_lines = [
            "DreamFit Conditioning Info:",
            f"Type: {conditioning.get('type', 'unknown')}",
            f"Text Prompt: {conditioning.get('text_prompt', 'N/A')}",
            f"Injection Mode: {conditioning['injection_config']['injection_mode']}",
            f"Injection Strength: {conditioning['injection_config']['injection_strength']}",
            f"Injection Layers: {conditioning['injection_config']['injection_layers']}",
            f"Has Pose: {'pose_features' in conditioning}",
            f"Has Mask: {'garment_mask' in conditioning}",
        ]
        
        # Add garment feature info
        if "garment_features" in conditioning:
            gf = conditioning["garment_features"]
            info_lines.extend([
                "\nGarment Features:",
                f"  - Garment Token Shape: {gf['garment_token'].shape}",
                f"  - Pooled Features Shape: {gf['pooled_features'].shape}",
                f"  - Patch Features Shape: {gf['patch_features'].shape}",
                f"  - Multi-scale Features: {list(gf['features'].keys())}",
            ])
        
        return ("\n".join(info_lines),)


# Node mappings
NODE_CLASS_MAPPINGS = {
    "DreamFitEncode": DreamFitEncode,
    "DreamFitConditioningInfo": DreamFitConditioningInfo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DreamFitEncode": "DreamFit Encode",
    "DreamFitConditioningInfo": "DreamFit Conditioning Info",
}